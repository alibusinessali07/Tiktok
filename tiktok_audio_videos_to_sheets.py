"""
TikTok Audio Videos to Google Sheets Scraper

This script reads a Google Sheet with Song Name and TikTok Audio URL columns,
scrapes videos from each audio page, and creates output sheets with video data.
After writing each output sheet, it sorts rows by upload_date (Z->A), detects
views/saves spikes, and colors spike rows green.

Required pip packages:
- playwright
- google-auth
- google-api-python-client
- google-auth-httplib2
- tqdm

Prerequisites:
- auto_auth.json (service account credentials) in the same directory
- spotify_secret.json (Spotify API client credentials) in the same directory
- Persistent Chrome profile already logged into TikTok with extension installed
- Run: playwright install chromium

SAFE SUGGESTIONS (not applied - may affect stability or behavior):
- Run in headless mode: may break extension injection or captcha flows.
- Parallelism across songs: multiple tabs could trigger rate limiting or captcha.
- Block stylesheets: may break layout/extension shadow DOM visibility.
- Disable JS on third-party domains: risk of breaking TikTok functionality.
- More aggressive settle time reduction: may miss extension metadata injection.
"""

import os
import re
import json
import logging
import time
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple, Any, Callable
from urllib.parse import urljoin
from collections import OrderedDict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright
import requests
from tqdm import tqdm

# Input / auth
DEFAULT_INPUT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1bWlvp89BtbsMYs4wiW-XFeUB8RK8bPGwHz_YfQXHunU/edit?usp=sharing"
)
DEFAULT_TEST_INPUT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1AGFFwTjEU0A2UalFaMHBRCjDWUALe2-kLRqbw7YvyYY/edit?gid=0#gid=0"
)


def _env_bool(var_name: str, default: bool = False) -> bool:
    raw = str(os.getenv(var_name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


TEST_MODE_ENABLED = _env_bool("ENABLE_TESTING_MODE", True) #Dont change this
PROD_INPUT_SHEET_URL = os.getenv("INPUT_SHEET_URL", DEFAULT_INPUT_SHEET_URL)
TEST_INPUT_SHEET_URL = os.getenv("TEST_INPUT_SHEET_URL", DEFAULT_TEST_INPUT_SHEET_URL)
INPUT_SHEET_URL = TEST_INPUT_SHEET_URL if TEST_MODE_ENABLED else PROD_INPUT_SHEET_URL
AUTH_FILE = Path(__file__).parent / "auto_auth.json"
SPOTIFY_SECRET_FILE = Path(__file__).parent / "spotify_secret.json"
PROFILE_DIR = Path(os.getenv("CHROME_EXTENSION_PROFILE", "./chrome_profile_tiktok_sorter")).resolve()

# Input sheet expected columns:
# A: Song Name | B: Spotify Audio URL | C: Spotify Release Date | D: Tiktok Audio URL | E: Output Link
INPUT_COL_SONG_NAME = 0
INPUT_COL_TIKTOK_AUDIO_URL = 3
INPUT_COL_OUTPUT_LINK = 4
INPUT_HEADER_SONG_NAME = "song name"
INPUT_HEADER_TIKTOK_AUDIO_URLS = {"tiktok audio url", "tik tok audio url"}
INPUT_HEADER_OUTPUT_LINK = "output link"
INPUT_COL_SPOTIFY_AUDIO_URL = 1
INPUT_COL_SPOTIFY_RELEASE_DATE = 2
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_HTTP_TIMEOUT_SEC = 20
SPOTIFY_FORCE_REFRESH_EXISTING_RELEASE_DATES = False

# Scraping thresholds
def _format_threshold_label(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return f"{value:,}"

PROD_VIEWS_THRESHOLD = 100
TEST_VIEWS_THRESHOLD = 10000
VIEWS_THRESHOLD = TEST_VIEWS_THRESHOLD if TEST_MODE_ENABLED else PROD_VIEWS_THRESHOLD
VIEWS_THRESHOLD_LABEL = _format_threshold_label(VIEWS_THRESHOLD)
VIDEOS_KEPT_FIELD = "videos_kept_ge_threshold"
MAX_SCROLLS = 200000
STAGNATION_SCROLLS = 3
SORT_SETTLE_SECONDS = 0.25
BUFFER_SCROLLS_AFTER_THRESHOLD = 1
DEBUG_SCROLL = False  # When False, scroll/debug logs are silent
SONG_SCRAPE_PROMPT_INTERVAL_SEC = 15 * 60
TAIL_READY_POLL_INTERVAL_SEC = 0.1
TAIL_READY_NUDGE_EVERY_POLLS = 50
TAIL_ROWS_APPEAR_STALL_POLLS = 80

# DOM selectors (music grid)
BASE_BODY = "#app div.e1pgfmdu0"
THREE_COL = "#main-content-single_song"
MUSIC_LIST = "#music-item-list"
GRID_ITEMS = '#music-item-list [id^="grid-item-container-"]'
ANCHOR_IN_ITEM = 'div[data-e2e="music-item"] a[href*="/video/"]'
META_IN_ANCHOR = "ov-ext-meta"
# Full selectors (grid item -> anchor -> meta)
ANCHORS_FULL = f'{MUSIC_LIST} [id^="grid-item-container-"] {ANCHOR_IN_ITEM}'
METAS_FULL = f'{ANCHORS_FULL} > {META_IN_ANCHOR}'
EXTENSION_WAIT_TIMEOUT_MS = 60000  # 60s for slow extension injection
EXTENSION_WAIT_LOG_INTERVAL_SEC = 2

# Precompiled JS (eval once per call, no f-string rebuild)
_GET_GRID_COUNTS_JS = """(args) => {
    const items = document.querySelectorAll(args.gridItemsSel);
    const anchors = document.querySelectorAll(args.anchorsFull);
    const metas = document.querySelectorAll(args.metasFull);
    return { items: items.length, anchors: anchors.length, metas: metas.length };
}"""
_WAIT_METAS_JS = "([sel, n]) => document.querySelectorAll(sel).length >= n"
_WAIT_CONTAINERS_JS = "sels => sels.every(s => document.querySelector(s))"
_WAIT_CONTAINERS_ARGS = ["#app", BASE_BODY, THREE_COL, MUSIC_LIST]
_WAIT_ALL_READY_JS = """(args) => {
    if (!args.sels.every(s => document.querySelector(s))) return false;
    return document.querySelectorAll(args.gridSel).length >= args.n
        && document.querySelectorAll(args.anchorsSel).length >= args.n;
}"""
_GRID_COUNT_JS = "sel => document.querySelectorAll(sel).length"
_CHECK_NEW_ITEMS_JS = """(args) => {
    const g = (root, n) => { const el = root.querySelector("div > div:nth-child(" + n + ") > div > div.t"); return el && el.innerText ? el.innerText.trim() : ""; };
    const items = document.querySelectorAll(args.gridItemsSel);
    const out = [];
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const anchor = item.querySelector(args.anchorSel);
        if (!anchor) continue;
        const meta = anchor.querySelector(args.metaSel);
        if (!meta) continue;
        const root = meta.shadowRoot || meta;
        out.push({ views: g(root, 8) });
    }
    return out;
}"""
_SCRAPE_ROWS_JS = """(args) => {
    const g = (root, n) => { const el = root.querySelector("div > div:nth-child(" + n + ") > div > div.t"); return el && el.innerText ? el.innerText.trim() : ""; };
    const items = document.querySelectorAll(args.gridItemsSel);
    const out = [];
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const anchor = item.querySelector(args.anchorSel);
        if (!anchor) continue;
        const href = anchor.getAttribute("href") || "";
        if (!href) continue;
        const meta = anchor.querySelector(args.metaSel);
        if (!meta) continue;
        const root = meta.shadowRoot || meta;
        const dEl = root.querySelector("div.d");
        const duration = dEl && dEl.innerText ? dEl.innerText.trim() : "";
        out.push({
            href: href.startsWith("http") ? href : "https://www.tiktok.com" + href,
            saves: g(root, 2), shares: g(root, 3), likes: g(root, 4), comments: g(root, 5),
            upload_date: g(root, 6), upload_time: g(root, 7), views: g(root, 8),
            duration: duration
        });
    }
    return { rows: out, totalCount: items.length };
}"""

# Output schema
OUTPUT_COLUMNS = [
    "video_link",
    "username",
    "profile_link",
    "views",
    "likes",
    "comments",
    "saves",
    "shares_reposts",
    "duration",
    "upload_date",
    "upload_time",
    "upload_datetime",
]
OUTPUT_COL_IDX_VIDEO_LINK = OUTPUT_COLUMNS.index("video_link")
OUTPUT_COL_IDX_VIEWS = OUTPUT_COLUMNS.index("views")
OUTPUT_COL_IDX_SAVES = OUTPUT_COLUMNS.index("saves")
OUTPUT_COL_IDX_UPLOAD_DATE = OUTPUT_COLUMNS.index("upload_date")
OUTPUT_COL_IDX_UPLOAD_TIME = OUTPUT_COLUMNS.index("upload_time")
OUTPUT_COL_IDX_UPLOAD_DATETIME = OUTPUT_COLUMNS.index("upload_datetime")
OUTPUT_COLUMNS_COUNT = len(OUTPUT_COLUMNS)
SPIKE_Q1_QUANTILE = 0.25
SPIKE_Q3_QUANTILE = 0.90
PLOT_DATA_SHEET_TITLE = "Daily Plot Data"
PLOT_CHART_SHEET_TITLE = "Daily Plot"

# Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

# Logging / runtime
LOGS_DIR = Path(__file__).parent / "logs"
CAIRO_TZ = ZoneInfo("Africa/Cairo")
_USERNAME_RE = re.compile(r"@([^/]+)/video/")


class CairoTimeFormatter(logging.Formatter):
    """Format log timestamps in Africa/Cairo timezone (YYYY-MM-DD HH:MM:SS)."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        ct = dt.datetime.fromtimestamp(record.created, tz=CAIRO_TZ)
        fmt = datefmt or getattr(self, "datefmt", None) or "%Y-%m-%d %H:%M:%S"
        return ct.strftime(fmt)


def setup_logging(log_file_path: Path) -> None:
    """
    Configure logging: file only, Cairo timestamps. No console output.
    Suppress noisy loggers (asyncio, playwright, urllib3).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = CairoTimeFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    root.addHandler(fh)
    for name in ("asyncio", "playwright", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class SongSkippedByUserError(Exception):
    """Raised when user chooses to skip a long-running song scrape."""


# ============================================================================
# Google Sheets API Functions
# ============================================================================

def extract_spreadsheet_id(url: str) -> str:
    """Extract spreadsheet ID from Google Sheets URL (/spreadsheets/d/{ID}/...)."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError(f"Could not extract spreadsheet ID from URL: {url}")
    return match.group(1)


def get_google_clients() -> Tuple:
    """Build Sheets and Drive API clients from service account (AUTH_FILE). Return (sheets_svc, drive_svc)."""
    if not AUTH_FILE.exists():
        raise FileNotFoundError(f"Service account file not found: {AUTH_FILE}")
    
    creds = service_account.Credentials.from_service_account_file(
        str(AUTH_FILE),
        scopes=SCOPES
    )
    
    sheets_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    return sheets_service, drive_service


_SPOTIFY_TOKEN_CACHE: Dict[str, Any] = {"access_token": "", "expires_at": 0.0}


def _load_spotify_credentials() -> Tuple[str, str]:
    if not SPOTIFY_SECRET_FILE.exists():
        raise FileNotFoundError(f"Spotify secret file not found: {SPOTIFY_SECRET_FILE}")
    raw = json.loads(SPOTIFY_SECRET_FILE.read_text(encoding="utf-8"))
    client_id = str(raw.get("client_id", "")).strip()
    client_secret = str(raw.get("client_secret", "")).strip()
    if not client_id or not client_secret:
        raise ValueError("spotify_secret.json must include non-empty client_id and client_secret")
    return client_id, client_secret


def _get_spotify_access_token(force_refresh: bool = False) -> str:
    now = time.time()
    token = str(_SPOTIFY_TOKEN_CACHE.get("access_token", "") or "")
    expires_at = float(_SPOTIFY_TOKEN_CACHE.get("expires_at", 0.0) or 0.0)
    if (not force_refresh) and token and expires_at > now + 30:
        return token

    client_id, client_secret = _load_spotify_credentials()
    response = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=SPOTIFY_HTTP_TIMEOUT_SEC,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Spotify token request failed: {response.status_code} {response.text[:200]}")

    payload = response.json()
    access_token = str(payload.get("access_token", "")).strip()
    expires_in = int(payload.get("expires_in", 3600) or 3600)
    if not access_token:
        raise RuntimeError("Spotify token response missing access_token")

    _SPOTIFY_TOKEN_CACHE["access_token"] = access_token
    _SPOTIFY_TOKEN_CACHE["expires_at"] = now + max(60, expires_in)
    return access_token


def _extract_spotify_resource(spotify_url: str) -> Optional[Tuple[str, str]]:
    s = str(spotify_url or "").strip()
    if not s:
        return None

    m = re.search(r"open\.spotify\.com/(track|album)/([A-Za-z0-9]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)

    m = re.match(r"spotify:(track|album):([A-Za-z0-9]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)

    return None


def _normalize_spotify_release_date(release_date: str, precision: str) -> str:
    rd = str(release_date or "").strip()
    p = str(precision or "").strip().lower()
    if not rd:
        return ""
    # Spotify can return YYYY, YYYY-MM, or YYYY-MM-DD.
    # Normalize to full date for consistent sheet values.
    if p == "year" and re.fullmatch(r"\d{4}", rd):
        return f"{rd}-01-01"
    if p == "month" and re.fullmatch(r"\d{4}-\d{2}", rd):
        return f"{rd}-01"
    return rd


def _fetch_spotify_release_date(spotify_url: str) -> Optional[str]:
    resource = _extract_spotify_resource(spotify_url)
    if not resource:
        return None
    resource_type, resource_id = resource

    endpoint = "tracks" if resource_type == "track" else "albums"

    def _do_request(access_token: str) -> requests.Response:
        return requests.get(
            f"{SPOTIFY_API_BASE}/{endpoint}/{resource_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=SPOTIFY_HTTP_TIMEOUT_SEC,
        )

    token = _get_spotify_access_token(force_refresh=False)
    response = _do_request(token)
    if response.status_code == 401:
        token = _get_spotify_access_token(force_refresh=True)
        response = _do_request(token)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", "1") or "1")
        time.sleep(max(1, min(retry_after, 5)))
        response = _do_request(token)

    if response.status_code != 200:
        logger.warning(
            "Spotify API request failed for %s (%s): %s",
            spotify_url,
            resource_type,
            response.status_code,
        )
        return None

    payload = response.json()
    if resource_type == "track":
        album = payload.get("album", {}) or {}
        release_date = str(album.get("release_date", "")).strip()
        precision = str(album.get("release_date_precision", "")).strip()
    else:
        release_date = str(payload.get("release_date", "")).strip()
        precision = str(payload.get("release_date_precision", "")).strip()

    normalized = _normalize_spotify_release_date(release_date, precision)
    return normalized or None


def update_spotify_release_dates(sheets_svc, spreadsheet_id: str) -> int:
    """
    For each input row:
      - Read Spotify URL from column B
      - Resolve release date via Spotify API
      - Write release date to column C
    Returns number of rows updated.
    """
    result = sheets_svc.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="A:C",
    ).execute()
    values = result.get("values", [])
    if not values:
        logger.info("Spotify release date update skipped: input sheet empty")
        return 0

    release_cache: Dict[str, Optional[str]] = {}
    updates: List[Dict[str, Any]] = []

    for row_index, row in enumerate(values[1:], start=2):
        spotify_url = str(row[INPUT_COL_SPOTIFY_AUDIO_URL]).strip() if len(row) > INPUT_COL_SPOTIFY_AUDIO_URL else ""
        current_release = str(row[INPUT_COL_SPOTIFY_RELEASE_DATE]).strip() if len(row) > INPUT_COL_SPOTIFY_RELEASE_DATE else ""
        if not spotify_url:
            continue
        if current_release and not SPOTIFY_FORCE_REFRESH_EXISTING_RELEASE_DATES:
            continue

        if spotify_url in release_cache:
            release_date = release_cache[spotify_url]
        else:
            try:
                release_date = _fetch_spotify_release_date(spotify_url)
            except Exception as e:
                logger.warning("Spotify lookup failed for row %s: %s", row_index, e)
                release_date = None
            release_cache[spotify_url] = release_date

        if not release_date:
            continue

        updates.append({"range": f"C{row_index}", "values": [[release_date]]})

    if not updates:
        logger.info("Spotify release date update: no rows needed updates")
        return 0

    chunk_size = 100
    for i in range(0, len(updates), chunk_size):
        chunk = updates[i:i + chunk_size]
        sheets_svc.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "USER_ENTERED", "data": chunk},
        ).execute()

    logger.info("Spotify release dates updated for %s row(s)", len(updates))
    return len(updates)


def get_input_rows(sheets_svc, spreadsheet_id: str) -> List[Dict]:
    """
    Read input rows using current schema:
      A Song Name, B Spotify Audio URL, C Spotify Release Date,
      D Tiktok Audio URL, E Output Link.
    Return list of:
      {song_name, audio_url, spotify_audio_url, spotify_release_date, row_index}
    where row_index is 1-based.
    """
    return _get_input_rows_impl(sheets_svc, spreadsheet_id)


def _get_input_rows_impl(sheets_svc, spreadsheet_id: str) -> List[Dict]:
    try:
        result = sheets_svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="A:E"  # Read Song Name through Output Link
        ).execute()
        
        values = result.get("values", [])
        if not values:
            logger.warning("No data found in input sheet")
            return []

        header = values[0] if values else []
        header_norm = [str(h).strip().lower() for h in header]
        if len(header_norm) > INPUT_COL_SONG_NAME and header_norm[INPUT_COL_SONG_NAME] != INPUT_HEADER_SONG_NAME:
            logger.warning(
                "Input header A1 expected '%s' but found '%s'",
                INPUT_HEADER_SONG_NAME,
                header[INPUT_COL_SONG_NAME] if len(header) > INPUT_COL_SONG_NAME else "",
            )
        if len(header_norm) > INPUT_COL_TIKTOK_AUDIO_URL and header_norm[INPUT_COL_TIKTOK_AUDIO_URL] not in INPUT_HEADER_TIKTOK_AUDIO_URLS:
            logger.warning(
                "Input header D1 expected one of %s but found '%s'",
                sorted(INPUT_HEADER_TIKTOK_AUDIO_URLS),
                header[INPUT_COL_TIKTOK_AUDIO_URL] if len(header) > INPUT_COL_TIKTOK_AUDIO_URL else "",
            )
        
        rows = []
        for idx, row in enumerate(values[1:], start=2):
            if (
                len(row) <= INPUT_COL_TIKTOK_AUDIO_URL
                or not str(row[INPUT_COL_SONG_NAME]).strip()
                or not str(row[INPUT_COL_TIKTOK_AUDIO_URL]).strip()
            ):
                continue  # Skip empty rows
            
            rows.append({
                "song_name": str(row[INPUT_COL_SONG_NAME]).strip(),
                "audio_url": str(row[INPUT_COL_TIKTOK_AUDIO_URL]).strip(),
                "spotify_audio_url": str(row[INPUT_COL_SPOTIFY_AUDIO_URL]).strip() if len(row) > INPUT_COL_SPOTIFY_AUDIO_URL else "",
                "spotify_release_date": str(row[INPUT_COL_SPOTIFY_RELEASE_DATE]).strip() if len(row) > INPUT_COL_SPOTIFY_RELEASE_DATE else "",
                "row_index": idx
            })
        
        logger.info("Found %s input rows", len(rows))
        return rows

    except HttpError as e:
        logger.error("Error reading input sheet: %s", e)
        raise


def ensure_output_link_column(sheets_svc, spreadsheet_id: str) -> None:
    """Ensure header row has 'Output Link' in column E."""
    _ensure_output_link_column_impl(sheets_svc, spreadsheet_id)


def _ensure_output_link_column_impl(sheets_svc, spreadsheet_id: str) -> None:
    try:
        # Read header row
        result = sheets_svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="1:1"  # First row only
        ).execute()
        
        headers = result.get("values", [[]])[0] if result.get("values") else []
        
        # Ensure Output Link is at column E (index 4)
        while len(headers) <= INPUT_COL_OUTPUT_LINK:
            headers.append("")

        current = str(headers[INPUT_COL_OUTPUT_LINK]).strip()
        if current.lower() != INPUT_HEADER_OUTPUT_LINK:
            headers[INPUT_COL_OUTPUT_LINK] = "Output Link"
            sheets_svc.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range="1:1",
                valueInputOption="RAW",
                body={"values": [headers]}
            ).execute()
            
            logger.info("Set 'Output Link' header in column E")
        else:
            logger.info("'Output Link' header already exists in column E")
            
    except HttpError as e:
        logger.error("Error ensuring output link column: %s", e)
        raise


def create_output_spreadsheet(
    drive_svc,
    sheets_svc,
    title: str
) -> Tuple[str, str]:
    """Create sheet with OUTPUT_COLUMNS header, set anyone-with-link writer. Return (id, url)."""
    return _create_output_spreadsheet_impl(drive_svc, sheets_svc, title)


def _create_output_spreadsheet_impl(
    drive_svc,
    sheets_svc,
    title: str
) -> Tuple[str, str]:
    try:
        # Create spreadsheet
        spreadsheet = {
            "properties": {"title": title}
        }
        
        result = sheets_svc.spreadsheets().create(body=spreadsheet).execute()
        spreadsheet_id = result["spreadsheetId"]
        spreadsheet_url = result["spreadsheetUrl"]
        
        # Write header row
        sheets_svc.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": [OUTPUT_COLUMNS]}
        ).execute()

        # Format D:H as comma-separated numbers and set a header filter (A:last col)
        _format_output_count_columns(sheets_svc, spreadsheet_id, apply_filter=True)
        
        # Set sharing: anyone with link can edit
        permission = {
            "type": "anyone",
            "role": "writer"
        }
        drive_svc.permissions().create(
            fileId=spreadsheet_id,
            body=permission
        ).execute()
        
        logger.info("Created output spreadsheet: %s", spreadsheet_url)
        return spreadsheet_id, spreadsheet_url
        
    except HttpError as e:
        logger.error("Error creating output spreadsheet: %s", e)
        raise


def _format_output_count_columns(
    sheets_svc,
    spreadsheet_id: str,
    apply_filter: bool = False,
) -> None:
    """Apply output sheet formats (D:H counts, I duration, J date, K time, L datetime)."""
    try:
        meta = sheets_svc.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets(properties(sheetId))",
        ).execute()
        sheets = meta.get("sheets", [])
        if not sheets:
            return
        sheet_id = sheets[0].get("properties", {}).get("sheetId")
        if sheet_id is None:
            return

        requests = [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"frozenRowCount": 1},
                    },
                    "fields": "gridProperties.frozenRowCount",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,   # skip header row
                        "startColumnIndex": 3, # D
                        "endColumnIndex": 8,   # H (exclusive)
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "NUMBER",
                                "pattern": "#,##0",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,   # skip header row
                        "startColumnIndex": 8, # I
                        "endColumnIndex": 9,   # I (exclusive)
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "TIME",
                                "pattern": "[mm]:ss",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,   # skip header row
                        "startColumnIndex": 9, # J
                        "endColumnIndex": 10,  # J (exclusive)
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "DATE",
                                "pattern": "m/d/yyyy",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,    # skip header row
                        "startColumnIndex": 10, # K
                        "endColumnIndex": 11,   # K (exclusive)
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "TIME",
                                "pattern": "h:mm:ss AM/PM",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,    # skip header row
                        "startColumnIndex": OUTPUT_COL_IDX_UPLOAD_DATETIME,
                        "endColumnIndex": OUTPUT_COL_IDX_UPLOAD_DATETIME + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "DATE_TIME",
                                "pattern": "m/d/yyyy h:mm:ss AM/PM",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            },
        ]

        if apply_filter:
            requests.append(
                {
                    "setBasicFilter": {
                        "filter": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 0,      # include header row
                                "startColumnIndex": 0,   # A
                                "endColumnIndex": len(OUTPUT_COLUMNS),  # through last output column
                            }
                        }
                    }
                }
            )

        sheets_svc.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        ).execute()
    except Exception as e:
        logger.warning("Could not apply output number/date/time formats: %s", e)


def append_rows_to_output_sheet(
    sheets_svc,
    spreadsheet_id: str,
    rows: List[List]
) -> None:
    """Append rows to output sheet (A2, INSERT_ROWS). No-op if rows empty."""
    _append_rows_to_output_sheet_impl(sheets_svc, spreadsheet_id, rows)


def _append_rows_to_output_sheet_impl(
    sheets_svc,
    spreadsheet_id: str,
    rows: List[List]
) -> None:
    if not rows:
        logger.warning("No rows to append")
        return
    
    try:
        sheets_svc.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="A2",  # Start after header
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows}
        ).execute()
        # Re-apply output formats after INSERT_ROWS so inserted rows keep number/date/time rendering.
        _format_output_count_columns(sheets_svc, spreadsheet_id)
        
    except HttpError as e:
        logger.error("Error appending rows: %s", e)
        raise


def postprocess_output_sheet(sheets_svc, spreadsheet_id: str) -> int:
    """
    Post-process one generated output sheet:
    1) Sort rows by upload_datetime descending (Z->A).
    2) Detect spikes in views/saves using IQR-style thresholds.
    3) Mark spike rows green.

    Returns number of marked rows.
    """
    return _postprocess_output_sheet_impl(sheets_svc, spreadsheet_id)


def _get_primary_sheet_props(sheets_svc, spreadsheet_id: str) -> Tuple[int, str]:
    meta = sheets_svc.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(sheetId,title))",
    ).execute()
    sheets = meta.get("sheets", [])
    if not sheets:
        raise ValueError(f"No sheets found in spreadsheet {spreadsheet_id}")
    props = sheets[0].get("properties", {})
    sheet_id = props.get("sheetId")
    title = props.get("title", "Sheet1")
    if sheet_id is None:
        raise ValueError(f"Missing sheetId in spreadsheet {spreadsheet_id}")
    return int(sheet_id), str(title)


def _read_output_rows_unformatted(
    sheets_svc,
    spreadsheet_id: str,
    sheet_title: str,
) -> List[List[Any]]:
    safe_title = sheet_title.replace("'", "''")
    last_col = _column_index_to_a1_label(OUTPUT_COLUMNS_COUNT - 1)
    result = sheets_svc.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{safe_title}'!A2:{last_col}",
        valueRenderOption="UNFORMATTED_VALUE",
        dateTimeRenderOption="SERIAL_NUMBER",
    ).execute()
    return result.get("values", [])


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", "").strip()
    if not s:
        return None
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", ".", "-.", ".-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _quantile_linear(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    q = max(0.0, min(1.0, q))
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _serial_to_iso_datetime(value: Any) -> str:
    serial = _to_float(value)
    if serial is None:
        s = str(value or "").strip()
        return s
    try:
        serial_day = int(serial)
        frac = float(serial - serial_day)
        dt_value = dt.datetime.combine(_SHEETS_EPOCH_DATE + dt.timedelta(days=serial_day), dt.time()) + dt.timedelta(days=frac)
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def _contiguous_ranges(sorted_indices: List[int]) -> List[Tuple[int, int]]:
    if not sorted_indices:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = prev = idx
    ranges.append((start, prev + 1))
    return ranges


def _column_index_to_a1_label(col_idx_0_based: int) -> str:
    """Convert 0-based column index to A1 column label (0->A, 25->Z, 26->AA)."""
    if col_idx_0_based < 0:
        raise ValueError("Column index must be >= 0")
    n = col_idx_0_based + 1
    chars: List[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def create_daily_plot_tabs(sheets_svc, spreadsheet_id: str) -> None:
    """
    Create/refresh the chart tab in one output spreadsheet.
    Chart series source ranges point directly to the primary output sheet.
    """
    _create_daily_plot_tabs_impl(sheets_svc, spreadsheet_id)


def _create_daily_plot_tabs_impl(sheets_svc, spreadsheet_id: str) -> None:
    primary_sheet_id, primary_sheet_title = _get_primary_sheet_props(sheets_svc, spreadsheet_id)
    rows = _read_output_rows_unformatted(sheets_svc, spreadsheet_id, primary_sheet_title)
    if not rows:
        logger.info("Daily plot skipped (%s): no data rows", spreadsheet_id)
        return

    meta = sheets_svc.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(sheetId,title))",
    ).execute()
    existing_ids: Dict[str, int] = {}
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {})
        sheet_id = props.get("sheetId")
        title = str(props.get("title") or "")
        if sheet_id is None or not title:
            continue
        existing_ids[title] = int(sheet_id)

    for reserved_title in (PLOT_DATA_SHEET_TITLE, PLOT_CHART_SHEET_TITLE):
        if existing_ids.get(reserved_title) == primary_sheet_id:
            raise ValueError(f"Cannot create '{reserved_title}' tab because primary sheet uses this title")

    setup_requests: List[Dict[str, Any]] = []
    for title in (PLOT_DATA_SHEET_TITLE, PLOT_CHART_SHEET_TITLE):
        existing_id = existing_ids.get(title)
        if existing_id is not None:
            setup_requests.append({"deleteSheet": {"sheetId": existing_id}})

    setup_requests.append({"addSheet": {"properties": {"title": PLOT_CHART_SHEET_TITLE}}})

    sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": setup_requests},
    ).execute()

    refreshed = sheets_svc.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(sheetId,title))",
    ).execute()
    chart_sheet_id: Optional[int] = None
    for sheet in refreshed.get("sheets", []):
        props = sheet.get("properties", {})
        title = str(props.get("title") or "")
        sheet_id = props.get("sheetId")
        if sheet_id is None:
            continue
        if title == PLOT_CHART_SHEET_TITLE:
            chart_sheet_id = int(sheet_id)

    if chart_sheet_id is None:
        raise ValueError("Failed to resolve sheet ID for daily plot chart tab")

    end_row = len(rows) + 1  # include header row from primary sheet
    chart_request = {
        "addChart": {
            "chart": {
                "spec": {
                    "title": f"Views and Saves over Upload DateTime ({primary_sheet_title})",
                    "basicChart": {
                        "chartType": "LINE",
                        "legendPosition": "TOP_LEGEND",
                        "headerCount": 1,
                        "axis": [
                            {"position": "BOTTOM_AXIS", "title": "Upload DateTime"},
                            {"position": "LEFT_AXIS", "title": "Views"},
                            {"position": "RIGHT_AXIS", "title": "Saves"},
                        ],
                        "domains": [
                            {
                                "domain": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": primary_sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": end_row,
                                                "startColumnIndex": OUTPUT_COL_IDX_UPLOAD_DATETIME,
                                                "endColumnIndex": OUTPUT_COL_IDX_UPLOAD_DATETIME + 1,
                                            }
                                        ]
                                    }
                                }
                            }
                        ],
                        "series": [
                            {
                                "series": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": primary_sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": end_row,
                                                "startColumnIndex": OUTPUT_COL_IDX_VIEWS,
                                                "endColumnIndex": OUTPUT_COL_IDX_VIEWS + 1,
                                            }
                                        ]
                                    }
                                },
                                "targetAxis": "LEFT_AXIS",
                            },
                            {
                                "series": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": primary_sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": end_row,
                                                "startColumnIndex": OUTPUT_COL_IDX_SAVES,
                                                "endColumnIndex": OUTPUT_COL_IDX_SAVES + 1,
                                            }
                                        ]
                                    }
                                },
                                "targetAxis": "RIGHT_AXIS",
                            },
                        ],
                    },
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {"sheetId": chart_sheet_id, "rowIndex": 0, "columnIndex": 0},
                        "offsetXPixels": 20,
                        "offsetYPixels": 20,
                        "widthPixels": 1200,
                        "heightPixels": 600,
                    }
                },
            }
        }
    }
    sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [chart_request]},
    ).execute()

    logger.info(
        "Created daily plot tab in %s using canonical upload_datetime column (%s rows)",
        spreadsheet_id,
        len(rows),
    )


def _postprocess_output_sheet_impl(sheets_svc, spreadsheet_id: str) -> int:
    sheet_id, sheet_title = _get_primary_sheet_props(sheets_svc, spreadsheet_id)

    rows_before_sort = _read_output_rows_unformatted(sheets_svc, spreadsheet_id, sheet_title)
    if not rows_before_sort:
        logger.info("Post-process skipped (%s): no data rows", spreadsheet_id)
        return 0

    # Sort uploaded rows by upload_datetime (column L) descending = Z->A.
    sort_request = {
        "sortRange": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1,  # skip header
                "endRowIndex": 1 + len(rows_before_sort),
                "startColumnIndex": 0,
                "endColumnIndex": OUTPUT_COLUMNS_COUNT,
            },
            "sortSpecs": [
                {
                    "dimensionIndex": OUTPUT_COL_IDX_UPLOAD_DATETIME,
                    "sortOrder": "DESCENDING",
                }
            ],
        }
    }
    sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [sort_request]},
    ).execute()
    logger.info("Sorted output sheet by upload_datetime Z->A: %s", spreadsheet_id)

    rows_after_sort = _read_output_rows_unformatted(sheets_svc, spreadsheet_id, sheet_title)
    if not rows_after_sort:
        logger.info("Post-process skipped after sort (%s): no rows", spreadsheet_id)
        return 0

    records: List[Dict[str, Any]] = []
    for row_idx_1_based, row in enumerate(rows_after_sort, start=2):
        video_link = str(row[OUTPUT_COL_IDX_VIDEO_LINK]).strip() if len(row) > OUTPUT_COL_IDX_VIDEO_LINK else ""
        views = _to_float(row[OUTPUT_COL_IDX_VIEWS] if len(row) > OUTPUT_COL_IDX_VIEWS else 0) or 0.0
        saves = _to_float(row[OUTPUT_COL_IDX_SAVES] if len(row) > OUTPUT_COL_IDX_SAVES else 0) or 0.0
        upload_datetime = row[OUTPUT_COL_IDX_UPLOAD_DATETIME] if len(row) > OUTPUT_COL_IDX_UPLOAD_DATETIME else ""
        if not video_link:
            continue
        records.append(
            {
                "row_idx_1_based": row_idx_1_based,
                "video_link": video_link,
                "views": float(views),
                "saves": float(saves),
                "upload_datetime": upload_datetime,
            }
        )

    if not records:
        logger.info("Post-process skipped (%s): no valid records with video_link", spreadsheet_id)
        return 0

    views_vals = [r["views"] for r in records]
    saves_vals = [r["saves"] for r in records]

    views_q1 = _quantile_linear(views_vals, SPIKE_Q1_QUANTILE)
    views_q3 = _quantile_linear(views_vals, SPIKE_Q3_QUANTILE)
    views_threshold = views_q3 + 1.5 * (views_q3 - views_q1)

    saves_q1 = _quantile_linear(saves_vals, SPIKE_Q1_QUANTILE)
    saves_q3 = _quantile_linear(saves_vals, SPIKE_Q3_QUANTILE)
    saves_threshold = saves_q3 + 1.5 * (saves_q3 - saves_q1)

    spike_records: List[Dict[str, Any]] = []
    for rec in records:
        views_spike = rec["views"] > views_threshold
        saves_spike = rec["saves"] > saves_threshold
        if not (views_spike or saves_spike):
            continue
        if views_spike and saves_spike:
            spike_type = "views_and_saves"
        elif views_spike:
            spike_type = "views_only"
        else:
            spike_type = "saves_only"
        rec2 = dict(rec)
        rec2["spike_type"] = spike_type
        spike_records.append(rec2)

    logger.info(
        "Spike thresholds for %s: views>%s, saves>%s",
        spreadsheet_id,
        int(round(views_threshold)),
        int(round(saves_threshold)),
    )

    if not spike_records:
        logger.info("No spikes detected in output sheet: %s", spreadsheet_id)
        return 0

    for rec in spike_records:
        logger.info(
            "SPIKE [%s] %s | %s | views=%s | saves=%s",
            rec["spike_type"],
            _serial_to_iso_datetime(rec["upload_datetime"]),
            rec["video_link"],
            int(round(rec["views"])),
            int(round(rec["saves"])),
        )

    color = {"red": 0.85, "green": 0.97, "blue": 0.85}
    row_indices_0_based = sorted({rec["row_idx_1_based"] - 1 for rec in spike_records})
    requests = []
    for start_row, end_row in _contiguous_ranges(row_indices_0_based):
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row,
                        "endRowIndex": end_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": OUTPUT_COLUMNS_COUNT,
                    },
                    "cell": {"userEnteredFormat": {"backgroundColor": color}},
                    "fields": "userEnteredFormat.backgroundColor",
                }
            }
        )

    sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()

    logger.info("Marked %s spike row(s) green in %s", len(row_indices_0_based), spreadsheet_id)
    return len(row_indices_0_based)


_OUTPUT_LINK_QUEUE: List[Tuple[str, int, str]] = []  # (spreadsheet_id, row_index, output_url)
_OUTPUT_LINK_BATCH_SIZE = 5


def _flush_output_link_queue(sheets_svc) -> None:
    """Flush queued output link writes via batchUpdate. Reduces API round-trips."""
    global _OUTPUT_LINK_QUEUE
    if not _OUTPUT_LINK_QUEUE:
        return
    # Group by spreadsheet_id (typically one input sheet for entire run)
    by_sheet: Dict[str, List[Tuple[int, str]]] = {}
    for sid, row_index, output_url in _OUTPUT_LINK_QUEUE:
        by_sheet.setdefault(sid, []).append((row_index, output_url))
    _OUTPUT_LINK_QUEUE = []
    for spreadsheet_id, updates in by_sheet.items():
        try:
            data = [
                {"range": "E%d" % row_index, "values": [[output_url]]}
                for row_index, output_url in updates
            ]
            sheets_svc.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"valueInputOption": "RAW", "data": data}
            ).execute()
        except HttpError as e:
            logger.error("Error batch writing output links: %s", e)
            raise


def queue_output_link(
    sheets_svc,
    spreadsheet_id: str,
    row_index: int,
    output_url: str,
    force_flush: bool = False
) -> None:
    """Queue output link write; flush when batch size reached or force_flush."""
    _OUTPUT_LINK_QUEUE.append((spreadsheet_id, row_index, output_url))
    if len(_OUTPUT_LINK_QUEUE) >= _OUTPUT_LINK_BATCH_SIZE or force_flush:
        _flush_output_link_queue(sheets_svc)


def write_output_link_back(
    sheets_svc,
    spreadsheet_id: str,
    row_index: int,
    output_url: str
) -> None:
    """Write output_url to column E at row_index. Uses queue for batching."""
    queue_output_link(sheets_svc, spreadsheet_id, row_index, output_url, force_flush=False)


# ============================================================================
# Browser Launch Functions
# ============================================================================

def launch_browser_with_profile() -> Tuple[Playwright, BrowserContext, Page]:
    """Launch Chrome persistent context (PROFILE_DIR), extensions enabled. Return (playwright, context, page)."""
    playwright = sync_playwright().start()
    
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        channel="chrome",
        headless=False,
        viewport=None,  # No viewport emulation
        args=[
            "--window-position=0,0",
        ],
        ignore_default_args=[
            "--enable-automation",
            "--disable-extensions",  # Prevent Playwright from disabling extensions
            "--no-sandbox",  # Remove sandbox flag to avoid warnings
        ],
    )
    
    # Get existing page or create new one
    if context.pages:
        page = context.pages[0]
    else:
        page = context.new_page()
    
    # Block image/media/font; also block known heavy analytics/tracking (avoid analytics. to not match tiktok.com/analytics)
    _ANALYTICS_DOMAINS = (
        "google-analytics.com", "googletagmanager.com", "doubleclick.net",
        "googleadservices.com", "facebook.com/tr", "connect.facebook.net",
    )

    def _route_handler(route):
        req = route.request
        url = (req.url or "").lower()
        if req.resource_type in ("image", "media", "font"):
            route.abort()
            return
        if any(d in url for d in _ANALYTICS_DOMAINS):
            route.abort()
            return
        route.continue_()

    page.route("**/*", _route_handler)

    logger.info("Browser launched (profile: %s)", PROFILE_DIR)
    return playwright, context, page


# ============================================================================
# Data Parsing Functions
# ============================================================================

_PARSE_COUNT_CACHE: OrderedDict[str, int] = OrderedDict()
_PARSE_COUNT_CACHE_MAX = 1024
_LEADING_SHEET_PREFIX_RE = re.compile(
    r"^[\s\u00A0\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069\uFEFF'`\u2018\u2019\u201B\u2032\u00B4]+"
)
_SHEETS_EPOCH_DATE = dt.date(1899, 12, 30)


def parse_count(text: str) -> int:
    """Parse count string (e.g. 49.9K, 1.2M, 12,345) to int. Cached with FIFO eviction."""
    if not text:
        return 0
    key = text.replace(",", "").strip().upper()
    if not key:
        return 0
    cache = _PARSE_COUNT_CACHE
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    multiplier = 1
    k = key
    if k.endswith("K"):
        multiplier = 1000
        k = k[:-1]
    elif k.endswith("M"):
        multiplier = 1000000
        k = k[:-1]
    try:
        value = float(k)
        result = int(value * multiplier)
    except ValueError:
        logger.warning("Could not parse count: %s", text)
        result = 0
    while len(cache) >= _PARSE_COUNT_CACHE_MAX:
        cache.popitem(last=False)
    cache[key] = result
    return result


def extract_username_from_url(video_url: str) -> Optional[str]:
    """Extract @username from TikTok video URL; return without @ or None."""
    match = _USERNAME_RE.search(video_url)
    return match.group(1) if match else None


def normalize_video_url(url: str) -> str:
    """Normalize href to absolute https://www.tiktok.com URL."""
    if url.startswith("http"):
        return url
    return urljoin("https://www.tiktok.com", url)


def normalize_sheet_text(value: Any) -> str:
    """Normalize text fields written to Sheets; strips leading quote/marker prefixes."""
    if value is None:
        return ""
    s = str(value).replace("\u00A0", " ").strip()
    if not s:
        return ""
    s = _LEADING_SHEET_PREFIX_RE.sub("", s)
    return s.strip()


def parse_duration_to_day_fraction(value: Any) -> Any:
    """Parse duration text (m:ss / mm:ss / hh:mm:ss) to Sheets day fraction."""
    s = normalize_sheet_text(value)
    if not s:
        return ""
    m = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        return ""
    a = int(m.group(1))
    b = int(m.group(2))
    c = m.group(3)
    if c is None:
        minutes = a
        seconds = b
        if seconds >= 60:
            return ""
        total_seconds = minutes * 60 + seconds
    else:
        hours = a
        minutes = b
        seconds = int(c)
        if minutes >= 60 or seconds >= 60:
            return ""
        total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds / 86400.0


def parse_date_to_serial(value: Any) -> Any:
    """Parse date text to Google Sheets serial day number."""
    s = normalize_sheet_text(value)
    if not s:
        return ""
    formats = (
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    )
    parsed_date = None
    for fmt in formats:
        try:
            parsed_date = dt.datetime.strptime(s, fmt).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        return ""
    return float((parsed_date - _SHEETS_EPOCH_DATE).days)


def parse_time_to_day_fraction(value: Any) -> Any:
    """Parse time text to Google Sheets day fraction."""
    s = normalize_sheet_text(value)
    if not s:
        return ""
    s = s.replace(".", "").upper()
    formats = (
        "%I:%M:%S %p",
        "%I:%M %p",
        "%I:%M:%S%p",
        "%I:%M%p",
        "%H:%M:%S",
        "%H:%M",
    )
    parsed_time = None
    for fmt in formats:
        try:
            parsed_time = dt.datetime.strptime(s, fmt).time()
            break
        except ValueError:
            continue
    if parsed_time is None:
        return ""
    total_seconds = parsed_time.hour * 3600 + parsed_time.minute * 60 + parsed_time.second
    return total_seconds / 86400.0


def combine_date_and_time_serial(upload_date: Any, upload_time: Any) -> Any:
    """Combine Sheets serial date + time fraction into one datetime serial."""
    date_serial = _to_float(upload_date)
    if date_serial is None:
        return ""
    time_fraction = _to_float(upload_time)
    if time_fraction is None:
        time_fraction = 0.0
    if time_fraction >= 1.0 or time_fraction <= -1.0:
        time_fraction = time_fraction - int(time_fraction)
    return float(int(date_serial) + float(time_fraction))


# ============================================================================
# Scraping Functions
# ============================================================================

_FIND_SCROLLER_JS = """
    () => {
      const candidates = [
        document.scrollingElement,
        document.documentElement,
        document.body,
        ...Array.from(document.querySelectorAll('#app, #app *'))
      ].filter(Boolean);
      let scroller = candidates.find(el => {
        try {
          const st = getComputedStyle(el);
          return (st.overflowY === 'auto' || st.overflowY === 'scroll')
            && el.scrollHeight > el.clientHeight + 50;
        } catch { return false; }
      });
      return scroller || document.scrollingElement || document.documentElement;
    }
"""
_SCROLL_EL_JS = """
    (el) => {
      if (!el) return { before: 0, after: 0, height: 0, client: 0 };
      const before = el.scrollTop;
      const height = el.scrollHeight;
      const client = el.clientHeight;
      el.scrollTop = height;
      return { before, after: el.scrollTop, height, client };
    }
"""
_NUDGE_SCROLL_EL_JS = """
    (el) => {
      if (!el) return { before: 0, after: 0, height: 0, client: 0 };
      const before = el.scrollTop;
      const height = el.scrollHeight || 0;
      const client = el.clientHeight || 0;
      const up = Math.max(120, Math.floor(client * 0.25));
      const down = Math.max(480, Math.floor(client * 0.9));
      el.scrollTop = Math.max(0, before - up);
      el.scrollTop = Math.min(height, el.scrollTop + down);
      return { before, after: el.scrollTop, height, client };
    }
"""


def get_scroller_handle(page) -> Any:
    """Cached scroller element; recompute if None or stale. Same selection logic as before."""
    h = getattr(page, "_scroller_handle", None)
    if h is not None:
        return h
    handle = page.evaluate_handle(_FIND_SCROLLER_JS)
    setattr(page, "_scroller_handle", handle)
    return handle


def scroll_to_bottom(page):
    """
    Scrolls the *actual* scrolling element to bottom (no mouse).
    Uses cached scroller; returns {before, after, height, client} for stuck detection.
    """
    handle = get_scroller_handle(page)
    try:
        return page.evaluate(_SCROLL_EL_JS, handle)
    except Exception:
        setattr(page, "_scroller_handle", None)
        handle = page.evaluate_handle(_FIND_SCROLLER_JS)
        setattr(page, "_scroller_handle", handle)
        return page.evaluate(_SCROLL_EL_JS, handle)


def nudge_scroll(page):
    """
    Small up/down nudge on the active scroller to trigger lazy-load handlers
    when direct "jump to bottom" reports no movement.
    """
    handle = get_scroller_handle(page)
    try:
        return page.evaluate(_NUDGE_SCROLL_EL_JS, handle)
    except Exception:
        setattr(page, "_scroller_handle", None)
        handle = page.evaluate_handle(_FIND_SCROLLER_JS)
        setattr(page, "_scroller_handle", handle)
        return page.evaluate(_NUDGE_SCROLL_EL_JS, handle)


def _get_grid_counts(page: Page) -> Dict[str, int]:
    """Return {items, anchors, metas} counts via GRID_ITEMS / ANCHORS_FULL / METAS_FULL."""
    out = page.evaluate(_GET_GRID_COUNTS_JS, _GET_GRID_COUNTS_ARGS)
    return {"items": out["items"], "anchors": out["anchors"], "metas": out["metas"]}


def _save_wait_diagnostics(page: Page, ts: int) -> None:
    """Save screenshot, HTML, and DOM dumps on wait failure."""
    screenshot_path = f"debug_wait_failed_{ts}.png"
    html_path = f"debug_wait_failed_{ts}.html"
    list_path = f"debug_music_item_list_{ts}.html"
    item_path = f"debug_first_item_{ts}.html"
    anchor_path = f"debug_first_anchor_{ts}.html"
    
    try:
        page.screenshot(path=screenshot_path, full_page=True)
        html = page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html[:200000])
    except Exception as e:
        logger.error("Failed to save screenshot/HTML: %s", e)
    
    js = f"""
        () => {{
            const list = document.querySelector({json.dumps(MUSIC_LIST)});
            const listHtml = list ? list.outerHTML.substring(0, 50000) : '';
            const firstItem = document.querySelector({json.dumps(GRID_ITEMS)});
            const firstItemHtml = firstItem ? firstItem.outerHTML : '';
            const firstAnchor = document.querySelector({json.dumps(ANCHORS_FULL)});
            const firstAnchorHtml = firstAnchor ? firstAnchor.outerHTML : '';
            return {{ listHtml, firstItemHtml, firstAnchorHtml }};
        }}
    """
    try:
        d = page.evaluate(js)
        for path, content in [
            (list_path, d.get("listHtml") or ""),
            (item_path, d.get("firstItemHtml") or ""),
            (anchor_path, d.get("firstAnchorHtml") or ""),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
    except Exception as e:
        logger.error("Failed to save DOM dumps: %s", e)
    
    logger.error(
        "Wait failed. URL=%s title=%s | screenshot=%s html=%s | list=%s first_item=%s first_anchor=%s",
        page.url, page.title(),
        screenshot_path, html_path, list_path, item_path, anchor_path,
    )


def wait_for_grid_and_extension(
    page: Page,
    timeout: int = 30000,
    min_items: int = 6,
    min_metas: int = 1,
) -> str:
    """
    Robust wait:
    - Avoid TikTok's unstable css-* selectors.
    - Wait for main song container + music list + grid items.
    - Then wait for extension's ov-ext-meta injection.
    """
    logger.info("Waiting for grid and extension.")
    try:
        # 1–3) Wait for containers + grid items + anchors (one round-trip)
        page.wait_for_function(
            _WAIT_ALL_READY_JS,
            arg={"sels": _WAIT_CONTAINERS_ARGS, "gridSel": GRID_ITEMS, "anchorsSel": ANCHORS_FULL, "n": min_items},
            timeout=timeout,
        )
        logger.info("Grid container, items and anchors loaded (>= %s)", min_items)

        # 4) Wait for extension metadata (ov-ext-meta) with progress logging
        logger.info("Waiting for extension metadata (>= %s ov-ext-meta).", min_metas)
        ext_timeout_sec = EXTENSION_WAIT_TIMEOUT_MS / 1000.0
        log_interval = EXTENSION_WAIT_LOG_INTERVAL_SEC
        t0 = time.time()
        last_log = -log_interval

        while (time.time() - t0) < ext_timeout_sec:
            elapsed = time.time() - t0
            if elapsed - last_log >= log_interval:
                counts = _get_grid_counts(page)
                logger.info(
                    "[extension wait] %.0fs | items=%s anchors=%s metas=%s",
                    elapsed, counts["items"], counts["anchors"], counts["metas"],
                )
                last_log = elapsed

            try:
                page.wait_for_function(
                    _WAIT_METAS_JS,
                    arg=[METAS_FULL, min_metas],
                    timeout=min(log_interval * 1000, max(1000, (ext_timeout_sec - elapsed) * 1000)),
                )
                logger.info("Extension metadata loaded")
                return MUSIC_LIST
            except Exception:
                pass

            time.sleep(0.1)  # minimal; was 0.5s, reduced to avoid per-track waste

        ts = int(time.time())
        _save_wait_diagnostics(page, ts)
        raise Exception(
            "Extension metadata (ov-ext-meta) not detected within %s s. "
            "Confirm extension is enabled. Diagnostics: debug_wait_failed_%s.*, "
            "debug_music_item_list_%s.html, debug_first_item_%s.html, debug_first_anchor_%s.html"
            % (ext_timeout_sec, ts, ts, ts, ts)
        )

    except Exception as e:
        # Save diagnostics on any failure (same behavior as your current code)
        if "Extension metadata" in str(e):
            raise
        ts = int(time.time())
        _save_wait_diagnostics(page, ts)
        logger.error("Wait failed: %s", e)
        raise


_GET_COUNT_LAST_VIEWS_JS = """(args) => {
    const items = document.querySelectorAll(args.gridItemsSel);
    const count = items.length;
    if (count === 0) return { count: 0, lastViewsText: null, debug: null };
    const lastItem = items[items.length - 1];
    const anchor = lastItem.querySelector(args.anchorSel);
    if (!anchor) return { count, lastViewsText: null, debug: args.debug ? { hadAnchor: false, hadMeta: false } : null };
    const meta = anchor.querySelector(args.metaSel);
    if (!meta) return { count, lastViewsText: null, debug: args.debug ? { hadAnchor: true, hadMeta: false } : null };
    const root = meta.shadowRoot || meta;
    const viewsEl = root.querySelector("div > div:nth-child(8) > div > div.t");
    const lastViewsText = viewsEl && viewsEl.innerText ? viewsEl.innerText : null;
    const debug = args.debug ? { hadAnchor: true, hadMeta: true, viewsSelectorFound: !!viewsEl, lastViewsText } : null;
    return { count, lastViewsText, debug };
}"""


def get_grid_count_and_last_views(page: Page) -> Tuple[int, Optional[int]]:
    """
    Single JS evaluate: returns (grid_count, last_item_views).
    Uses same DOM path as get_last_grid_item_views (meta -> nth-child(8) -> div.t).
    """
    try:
        result = page.evaluate(_GET_COUNT_LAST_VIEWS_JS, _GET_COUNT_LAST_VIEWS_ARGS)
        count = int(result.get("count", 0))
        raw = result.get("lastViewsText")
        views = parse_count(raw) if raw is not None else None
        if DEBUG_SCROLL and result.get("debug"):
            dbg = result["debug"]
            if views is None and dbg.get("hadMeta"):
                logger.info("[DEBUG] get_grid_count_and_last_views: count=%s views=None hadAnchor=%s hadMeta=%s viewsSelectorFound=%s lastViewsText=%s",
                    count, dbg.get("hadAnchor"), dbg.get("hadMeta"), dbg.get("viewsSelectorFound"), repr(dbg.get("lastViewsText")))
        return count, views
    except Exception as e:
        if DEBUG_SCROLL:
            logger.info("[DEBUG] get_grid_count_and_last_views exception: %s", e)
        return 0, None


def get_last_grid_item_views(page: Page, root_selector: str = MUSIC_LIST) -> Optional[int]:
    """Get views from the last grid item (ANCHOR_IN_ITEM, META_IN_ANCHOR). Returns None if not found."""
    _, views = get_grid_count_and_last_views(page)
    return views


_CHECK_NEW_ARGS = {"gridItemsSel": GRID_ITEMS, "anchorSel": ANCHOR_IN_ITEM, "metaSel": META_IN_ANCHOR}
_GET_GRID_COUNTS_ARGS = {"gridItemsSel": GRID_ITEMS, "anchorsFull": ANCHORS_FULL, "metasFull": METAS_FULL}
_GET_COUNT_LAST_VIEWS_ARGS = {"gridItemsSel": GRID_ITEMS, "anchorSel": ANCHOR_IN_ITEM, "metaSel": META_IN_ANCHOR, "debug": DEBUG_SCROLL}


def check_new_items_below_threshold(
    page: Page,
    prev_count: int,
    threshold: int,
) -> Tuple[int, int, bool, int]:
    """
    Returns (newCount, delta, allNewBelow, maxNewViews).
    Uses the SAME extraction as scrape_loaded_items (g(8) for views).
    allNewBelow=True only if every new item has views < threshold.
    Missing views -> allNewBelow=False.
    """
    rows = page.evaluate(_CHECK_NEW_ITEMS_JS, _CHECK_NEW_ARGS)
    new_count = len(rows)
    delta = max(0, new_count - prev_count)
    all_new_below = True
    max_new_views = 0
    for i in range(prev_count, new_count):
        views_text = rows[i].get("views", "") or ""
        if not views_text:
            all_new_below = False
            continue
        v = parse_count(views_text)
        max_new_views = max(max_new_views, v)
        if v >= threshold:
            all_new_below = False
    return new_count, delta, all_new_below, max_new_views


_WAIT_COUNT_JS = "([sel, prev]) => document.querySelectorAll(sel).length > prev"


def _noop_watchdog() -> None:
    return None


def wait_for_new_items_and_extension_settle(
    page: Page,
    prev_count: int,
    wait_timeout_ms: int = 5000,
    settle_seconds: float = SORT_SETTLE_SECONDS,
    watchdog_cb: Optional[Callable[[], None]] = None,
) -> bool:
    """
    Wait until TikTok loads more GRID_ITEMS than prev_count.
    After items arrive, wait for new extractable tail rows, then wait
    indefinitely until extension views are ready for those rows.
    If extractable tail rows never appear and counts stay stalled, return False
    so the caller can continue end-of-list detection.
    Returns True if new items loaded, else False.
    """
    watchdog = watchdog_cb or _noop_watchdog
    try:
        baseline_rows = len(page.evaluate(_CHECK_NEW_ITEMS_JS, _CHECK_NEW_ARGS))
    except Exception:
        baseline_rows = 0
    loaded = False
    try:
        page.wait_for_function(
            _WAIT_COUNT_JS,
            arg=[GRID_ITEMS, prev_count],
            timeout=wait_timeout_ms,
        )
        loaded = True
    except Exception:
        loaded = False

    if loaded:
        polls = 0
        no_progress_polls = 0
        new_rows_seen = False
        try:
            last_grid_count = int(page.evaluate(_GRID_COUNT_JS, GRID_ITEMS))
        except Exception:
            last_grid_count = prev_count
        while True:
            watchdog()
            try:
                grid_count_now = int(page.evaluate(_GRID_COUNT_JS, GRID_ITEMS))
            except Exception:
                grid_count_now = last_grid_count
            try:
                rows = page.evaluate(_CHECK_NEW_ITEMS_JS, _CHECK_NEW_ARGS)
                row_count = len(rows)
                if row_count <= baseline_rows:
                    ready = False
                else:
                    new_rows_seen = True
                    ready = True
                    for i in range(baseline_rows, row_count):
                        views_text = rows[i].get("views", "") or ""
                        if not str(views_text).strip():
                            ready = False
                            break
            except Exception:
                ready = False
                row_count = baseline_rows
            if ready:
                break

            if not new_rows_seen:
                progressed = row_count > baseline_rows or grid_count_now > last_grid_count
                if progressed:
                    no_progress_polls = 0
                else:
                    no_progress_polls += 1
                    if no_progress_polls >= TAIL_ROWS_APPEAR_STALL_POLLS:
                        if DEBUG_SCROLL:
                            logger.info(
                                "[DEBUG] tail rows stalled: prev_count=%s baseline_rows=%s grid_now=%s polls=%s",
                                prev_count, baseline_rows, grid_count_now, no_progress_polls
                            )
                        return False
                last_grid_count = max(last_grid_count, grid_count_now)

            polls += 1
            if polls % TAIL_READY_NUDGE_EVERY_POLLS == 0:
                try:
                    nudge_scroll(page)
                except Exception:
                    pass
            time.sleep(TAIL_READY_POLL_INTERVAL_SEC)
        time.sleep(settle_seconds)
    return loaded


def scroll_until_views_below_threshold(
    page: Page,
    root_selector: str,
    threshold: int = VIEWS_THRESHOLD,
    max_scrolls: int = MAX_SCROLLS,
    watchdog_cb: Optional[Callable[[], None]] = None,
):
    """
    Scrolls until a full new scroll loads new items AND all newly loaded items
    are below the views threshold.
    """

    threshold_reached = False
    watchdog = watchdog_cb or _noop_watchdog
    max_count_seen = 0
    iterations_no_increase = 0
    no_movement_streak = 0
    prev_count = int(page.evaluate(_GRID_COUNT_JS, GRID_ITEMS))

    for scroll_num in range(max_scrolls):
        watchdog()
        try:
            scroll_ret = scroll_to_bottom(page)
            if DEBUG_SCROLL:
                logger.info("[DEBUG] scroll_to_bottom scroll=%s ret=%s", scroll_num, scroll_ret)
        except Exception as ex:
            if DEBUG_SCROLL:
                logger.info("[DEBUG] scroll_to_bottom scroll=%s exception=%s", scroll_num, ex)
            continue

        scroll_moved = scroll_ret.get("after") != scroll_ret.get("before")
        if not scroll_moved:
            # If items increased despite no scroll delta, continue scanning.
            new_count, _ = get_grid_count_and_last_views(page)
            if new_count > prev_count:
                prev_count = new_count
                no_movement_streak = 0
                continue

            # Nudge to fire lazy-load handlers that may not trigger on no-op scroll.
            try:
                nudge_scroll(page)
            except Exception:
                pass

            loaded_after_nudge = wait_for_new_items_and_extension_settle(
                page, prev_count=prev_count, wait_timeout_ms=800, settle_seconds=0.1, watchdog_cb=watchdog
            )
            if loaded_after_nudge:
                new_count, delta, all_new_below, max_new_views = check_new_items_below_threshold(
                    page, prev_count, threshold
                )
                prev_count = new_count
                no_movement_streak = 0
                if delta > 0 and all_new_below:
                    threshold_reached = True
                    logger.info("Stopping: new batch delta=%s, maxNewViews=%s < threshold=%s",
                        delta, max_new_views, threshold)
                    break
                continue

            new_count, _ = get_grid_count_and_last_views(page)
            if new_count > prev_count:
                prev_count = new_count
                no_movement_streak = 0
                continue

            no_movement_streak += 1
            if no_movement_streak >= STAGNATION_SCROLLS:
                prev_count = new_count
                logger.info(
                    "Stopping: scroll at bottom after %s no-move checks, newCount=%s",
                    no_movement_streak, new_count
                )
                threshold_reached = True
                break
            continue

        no_movement_streak = 0

        loaded = wait_for_new_items_and_extension_settle(page, prev_count=prev_count, watchdog_cb=watchdog)
        if loaded:
            new_count, delta, all_new_below, max_new_views = check_new_items_below_threshold(
                page, prev_count, threshold
            )
        else:
            new_count, last_views = get_grid_count_and_last_views(page)
            delta = max(0, new_count - prev_count)
            all_new_below = False if delta > 0 else True
            max_new_views = last_views if last_views is not None else 0

        prev_count = new_count

        if DEBUG_SCROLL:
            if scroll_num % 25 == 0 and scroll_num > 0:
                logger.info("[DEBUG] scroll progress: iteration %s", scroll_num)
            logger.info("[DEBUG] post-wait scroll=%s loaded=%s delta=%s allNewBelow=%s maxNewViews=%s scroll_moved=%s",
                scroll_num, loaded, delta, all_new_below, max_new_views, scroll_moved)

        if delta == 0:
            iterations_no_increase += 1
            if new_count > max_count_seen:
                max_count_seen = new_count
                iterations_no_increase = 0
            if DEBUG_SCROLL and iterations_no_increase >= 10:
                logger.info("[DEBUG] stagnation: count=%s unchanged for %s iterations scroll=%s",
                    new_count, iterations_no_increase, scroll_num)
            continue

        if new_count > max_count_seen:
            max_count_seen = new_count
            iterations_no_increase = 0

        if delta > 0 and all_new_below:
            threshold_reached = True
            logger.info("Stopping: new batch delta=%s, maxNewViews=%s < threshold=%s",
                delta, max_new_views, threshold)
            break

    # -------------------------
    # Buffer scrolling phase
    # -------------------------
    if threshold_reached:
        logger.info(
            "Buffer scrolling starting (%s extra scrolls, resets if >= threshold found).",
            BUFFER_SCROLLS_AFTER_THRESHOLD,
        )

        buffer_left = BUFFER_SCROLLS_AFTER_THRESHOLD

        while buffer_left > 0:
            watchdog()
            if prev_count == 0:
                break

            try:
                scroll_ret = scroll_to_bottom(page)
            except Exception:
                continue

            scroll_moved = scroll_ret.get("after") != scroll_ret.get("before")
            if not scroll_moved:
                break
            loaded = wait_for_new_items_and_extension_settle(page, prev_count=prev_count, watchdog_cb=watchdog)
            if loaded:
                new_count, delta, all_new_below, max_new_views = check_new_items_below_threshold(
                    page, prev_count, threshold
                )
            else:
                new_count, last_views = get_grid_count_and_last_views(page)
                delta = max(0, new_count - prev_count)
                all_new_below = False if delta > 0 else True
                max_new_views = last_views if last_views is not None else 0
            prev_count = new_count

            if DEBUG_SCROLL:
                logger.info("[DEBUG] buffer post-wait buffer_left=%s loaded=%s delta=%s allNewBelow=%s maxNewViews=%s",
                    buffer_left, loaded, delta, all_new_below, max_new_views)

            if delta > 0 and not all_new_below:
                buffer_left = BUFFER_SCROLLS_AFTER_THRESHOLD
            else:
                buffer_left -= 1

        logger.info("Buffer scrolling done.")


def scrape_loaded_items(
    page: Page,
    root_selector: str = MUSIC_LIST,
    threshold: int = VIEWS_THRESHOLD
) -> Tuple[List[List], int]:
    """
    Scrape loaded grid items. Single evaluate. Returns (output rows in OUTPUT_COLUMNS order, grid count).
    Builds rows directly to avoid intermediate dict churn.
    """
    rows = page.evaluate(_SCRAPE_ROWS_JS, _CHECK_NEW_ARGS)
    rows_list = rows["rows"]
    grid_count = rows["totalCount"]

    output_rows = []
    profile_prefix = "https://www.tiktok.com/@"
    for idx, row in enumerate(rows_list):
        try:
            video_link = row["href"]
            username = extract_username_from_url(video_link)
            if not username:
                continue
            views = parse_count(row["views"])
            if views < threshold:
                continue
            duration = parse_duration_to_day_fraction(row.get("duration", ""))
            upload_date = parse_date_to_serial(row.get("upload_date", ""))
            upload_time = parse_time_to_day_fraction(row.get("upload_time", ""))
            upload_datetime = combine_date_and_time_serial(upload_date, upload_time)
            output_rows.append([
                video_link,
                username,
                profile_prefix + username,
                views,
                parse_count(row["likes"]),
                parse_count(row["comments"]),
                parse_count(row["saves"]),
                parse_count(row["shares"]),
                duration,
                upload_date,
                upload_time,
                upload_datetime,
            ])
        except Exception as e:
            logger.warning("Error scraping item %s: %s", idx, e)

    logger.info("Scraped %s videos (>= %s) from %s grid items", len(output_rows), threshold, grid_count)
    return output_rows, grid_count


_BLOCKER_KEYWORDS = ("verify", "captcha", "enable cookies", "something went wrong")


def check_tiktok_blocker(page: Page) -> bool:
    """Return True if body text contains verify/captcha/cookies/something went wrong."""
    try:
        page_text = page.inner_text("body").lower()
        return any(kw in page_text for kw in _BLOCKER_KEYWORDS)
    except Exception:
        return False


def process_audio_url(
    page: Page,
    audio_url: str,
    song_name: str
) -> Tuple[List[List], int]:
    """Process audio URL: navigate, scroll, scrape. Return (output rows, grid_count)."""
    logger.info("Processing: %s", song_name)
    next_watchdog_prompt_at = time.monotonic() + SONG_SCRAPE_PROMPT_INTERVAL_SEC

    def _scrape_watchdog() -> None:
        nonlocal next_watchdog_prompt_at
        now = time.monotonic()
        if now < next_watchdog_prompt_at:
            return

        prompt = (
            f"Song '{song_name}' has been scraping for over 15 minutes. "
            "Continue waiting for this song? [y/n]: "
        )
        while True:
            answer = input(prompt).strip().lower()
            if answer in ("y", "yes"):
                next_watchdog_prompt_at = time.monotonic() + SONG_SCRAPE_PROMPT_INTERVAL_SEC
                logger.warning("Continuing scrape for %s for another 15-minute window.", song_name)
                return
            if answer in ("n", "no"):
                msg = f"Skipped by user after 15-minute scrape window: {song_name}"
                logger.warning(msg)
                raise SongSkippedByUserError(msg)
            print("Please enter 'y' or 'n'.")

    try:
        _scrape_watchdog()
        page.goto(audio_url, wait_until="domcontentloaded", timeout=60000)
        setattr(page, "_scroller_handle", None)

        if check_tiktok_blocker(page):
            logger.warning("TikTok blocker/captcha detected")
            input("TikTok may be blocking. Fix it in the open browser, then press Enter...")

        _scrape_watchdog()
        root_selector = wait_for_grid_and_extension(page)
        if DEBUG_SCROLL:
            counts = _get_grid_counts(page)
            logger.info("[DEBUG] After wait_for_grid_and_extension: url=%s root=%s items=%s anchors=%s metas=%s",
                page.url[:80], root_selector, counts["items"], counts["anchors"], counts["metas"])
            logger.info("[DEBUG] Before scroll_until_views_below_threshold")

        scroll_until_views_below_threshold(page, root_selector, watchdog_cb=_scrape_watchdog)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] After scroll_until_views_below_threshold returned")

        time.sleep(0.1)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] Before scrape_loaded_items")

        output_rows, grid_count = scrape_loaded_items(page, root_selector)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] After scrape_loaded_items: rows=%s grid_count=%s", len(output_rows), grid_count)
        return output_rows, grid_count

    except SongSkippedByUserError:
        raise

    except Exception as e:
        logger.error("Error processing %s: %s", song_name, e)
        # Check if it's a captcha/block issue
        if "captcha" in str(e).lower() or "blocked" in str(e).lower():
            logger.warning("Possible captcha or blocking detected")
            input("Please resolve any captcha/blocking in the browser, then press Enter to continue...")
        return [], 0


# ============================================================================
# Main Function
# ============================================================================

def _format_elapsed(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def main():
    """Main execution function."""
    log_ts = dt.datetime.now(CAIRO_TZ).strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"tiktok_scrape_{log_ts}.log"
    setup_logging(log_path)
    logger.info("="*70)
    logger.info("TikTok Audio Videos to Sheets Scraper")
    logger.info("="*70)
    logger.info("Log file: %s", log_path)
    logger.info("Testing mode enabled: %s", TEST_MODE_ENABLED)
    logger.info(
        "Input sheet URL source: %s",
        "TEST_INPUT_SHEET_URL" if TEST_MODE_ENABLED else "INPUT_SHEET_URL",
    )

    run_stats: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    # Build Google API clients
    try:
        sheets_svc, drive_svc = get_google_clients()
    except Exception as e:
        logger.error("Failed to initialize Google clients: %s", e)
        return

    # Extract input spreadsheet ID
    try:
        input_spreadsheet_id = extract_spreadsheet_id(INPUT_SHEET_URL)
        logger.info("Input spreadsheet ID: %s", input_spreadsheet_id)
    except Exception as e:
        logger.error("Failed to extract spreadsheet ID: %s", e)
        return

    # Ensure output link column exists
    try:
        ensure_output_link_column(sheets_svc, input_spreadsheet_id)
    except Exception as e:
        logger.error("Failed to ensure output link column: %s", e)
        return

    # Update Spotify release dates (B -> C) before processing TikTok URLs.
    try:
        updated_release_dates = update_spotify_release_dates(sheets_svc, input_spreadsheet_id)
        logger.info("Spotify release date rows updated: %s", updated_release_dates)
    except Exception as e:
        logger.warning("Spotify release date update failed; continuing: %s", e)

    # Get input rows
    try:
        input_rows = get_input_rows(sheets_svc, input_spreadsheet_id)
    except Exception as e:
        logger.error("Failed to get input rows: %s", e)
        return

    if not input_rows:
        logger.warning("No input rows to process")
        return

    # Filter out empty URLs for tqdm iteration
    rows_to_process = [r for r in input_rows if r.get("audio_url")]
    if len(rows_to_process) < len(input_rows):
        logger.warning("Skipping %s row(s) with empty audio URL", len(input_rows) - len(rows_to_process))
    if not rows_to_process:
        logger.warning("No rows to process")
        return

    # Launch browser (reuse for all rows)
    try:
        playwright, context, page = launch_browser_with_profile()
    except Exception as e:
        logger.error("Failed to launch browser: %s", e)
        return

    try:
        pbar = tqdm(
            rows_to_process,
            desc="Processing songs",
            unit="song",
            dynamic_ncols=True,
        )
        for input_row in pbar:
            song_name = input_row["song_name"]
            audio_url = input_row["audio_url"]
            row_index = input_row["row_index"]
            start_ts = time.perf_counter()
            stat: Dict[str, Any] = {
                "song_name": song_name,
                "audio_url": audio_url,
                "row_index": row_index,
                "start_ts": start_ts,
                "end_ts": 0.0,
                "elapsed_seconds": 0.0,
                "grid_items_loaded_count": 0,
                VIDEOS_KEPT_FIELD: 0,
                "spike_rows_marked": 0,
                "output_sheet_url": "",
                "status": "failed",
                "error_message": "",
            }
            try:
                output_rows, grid_count = process_audio_url(page, audio_url, song_name)
                stat["grid_items_loaded_count"] = grid_count
                n_videos = len(output_rows)
                stat[VIDEOS_KEPT_FIELD] = n_videos

                if not output_rows:
                    stat["status"] = "no_videos"
                    stat["end_ts"] = time.perf_counter()
                    stat["elapsed_seconds"] = stat["end_ts"] - start_ts
                    run_stats.append(stat)
                    pbar.set_postfix_str("%s | 0 v | %ss" % (song_name[:18], int(stat["elapsed_seconds"])))
                    continue

                output_title = f"{song_name} - TikTok Audio Videos"
                output_spreadsheet_id, output_url = create_output_spreadsheet(
                    drive_svc, sheets_svc, output_title
                )
                append_rows_to_output_sheet(sheets_svc, output_spreadsheet_id, output_rows)
                stat["spike_rows_marked"] = postprocess_output_sheet(sheets_svc, output_spreadsheet_id)
                write_output_link_back(sheets_svc, input_spreadsheet_id, row_index, output_url)
                try:
                    create_daily_plot_tabs(sheets_svc, output_spreadsheet_id)
                except Exception as chart_err:
                    logger.warning(
                        "Could not create daily plot tabs for %s: %s",
                        output_spreadsheet_id,
                        chart_err,
                    )

                stat["output_sheet_url"] = output_url
                stat["status"] = "success"
                stat["end_ts"] = time.perf_counter()
                stat["elapsed_seconds"] = stat["end_ts"] - start_ts
                run_stats.append(stat)
                pbar.set_postfix_str("%s | %s v | %ss" % (song_name[:18], n_videos, int(stat["elapsed_seconds"])))
                logger.info("Completed %s: %s videos -> %s", song_name, n_videos, output_url)

            except SongSkippedByUserError as err:
                stat["status"] = "failed"
                stat["error_message"] = str(err)[:200]
                stat["end_ts"] = time.perf_counter()
                stat["elapsed_seconds"] = stat["end_ts"] - start_ts
                run_stats.append(stat)
                pbar.set_postfix_str("%s | FAIL | %ss" % (song_name[:18], int(stat["elapsed_seconds"])))
                logger.error("Skipped %s by user: %s", song_name, err)

            except Exception as err:
                stat["status"] = "failed"
                stat["error_message"] = str(err)[:200]
                stat["end_ts"] = time.perf_counter()
                stat["elapsed_seconds"] = stat["end_ts"] - start_ts
                run_stats.append(stat)
                pbar.set_postfix_str("%s | FAIL | %ss" % (song_name[:18], int(stat["elapsed_seconds"])))
                logger.error("Failed %s: %s", song_name, err)

    finally:
        _flush_output_link_queue(sheets_svc)
        logger.info("Closing browser...")
        context.close()
        playwright.stop()

    total_elapsed = time.perf_counter() - total_start
    n_ok = sum(1 for s in run_stats if s["status"] == "success")
    n_fail = sum(1 for s in run_stats if s["status"] == "failed")
    n_no_videos = sum(1 for s in run_stats if s["status"] == "no_videos")
    total_videos = sum(s[VIDEOS_KEPT_FIELD] for s in run_stats)
    total_spike_rows_marked = sum(int(s.get("spike_rows_marked", 0) or 0) for s in run_stats)
    avg_time = total_elapsed / len(run_stats) if run_stats else 0.0
    slowest = sorted(run_stats, key=lambda s: s["elapsed_seconds"], reverse=True)[:3]

    lines = [
        "",
        "=" * 70,
        "FINAL REPORT",
        "=" * 70,
        "Total songs processed: %s" % len(run_stats),
        "  Success: %s  |  No videos: %s  |  Failed: %s" % (n_ok, n_no_videos, n_fail),
        "Total runtime: %s" % _format_elapsed(total_elapsed),
        "Total videos (>= %s): %s" % (VIEWS_THRESHOLD_LABEL, total_videos),
        "Total spike rows marked green: %s" % total_spike_rows_marked,
        "Avg time per song: %s" % _format_elapsed(avg_time),
        "Slowest 3:",
    ]
    for s in slowest:
        lines.append("  - %s  %s  (%s videos)  %s" % (
            s["song_name"][:40], _format_elapsed(s["elapsed_seconds"]),
            s[VIDEOS_KEPT_FIELD], s["output_sheet_url"] or s.get("error_message", "")[:50]
        ))
    lines.append("")
    for s in run_stats:
        lines.append("[%s] %s  %s  %s videos  %s spikes  %s  %s" % (
            s["status"],
            s["song_name"][:35],
            _format_elapsed(s["elapsed_seconds"]),
            s[VIDEOS_KEPT_FIELD],
            int(s.get("spike_rows_marked", 0) or 0),
            s["output_sheet_url"] or "-",
            (" " + s["error_message"][:60]) if s["error_message"] else "",
        ))
    lines.append("=" * 70)
    report = "\n".join(lines)

    print(report)
    logger.info("\n%s", report)


if __name__ == "__main__":
    main()
