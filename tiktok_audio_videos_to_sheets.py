"""
TikTok Audio Videos to Google Sheets Scraper

This script reads a Google Sheet with Song Name and TikTok Audio URL columns,
scrapes videos from each audio page, and creates output sheets with video data.

Required pip packages:
- playwright
- google-auth
- google-api-python-client
- google-auth-httplib2
- tqdm

Prerequisites:
- auto_auth.json (service account credentials) in the same directory
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
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin
from collections import OrderedDict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright
from tqdm import tqdm

# Input / auth
INPUT_SHEET_URL = os.getenv(
    "INPUT_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/1bWlvp89BtbsMYs4wiW-XFeUB8RK8bPGwHz_YfQXHunU/edit?usp=sharing"
)
AUTH_FILE = Path(__file__).parent / "auto_auth.json"
PROFILE_DIR = Path(os.getenv("CHROME_EXTENSION_PROFILE", "./chrome_profile_tiktok_sorter")).resolve()

# Scraping thresholds
def _format_threshold_label(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return f"{value:,}"


VIEWS_THRESHOLD = 10_000
VIEWS_THRESHOLD_LABEL = _format_threshold_label(VIEWS_THRESHOLD)
VIDEOS_KEPT_FIELD = "videos_kept_ge_threshold"
MAX_SCROLLS = 2000
STAGNATION_SCROLLS = 10
SORT_SETTLE_SECONDS = 0.25
BUFFER_SCROLLS_AFTER_THRESHOLD = 1
DEBUG_SCROLL = False  # When False, scroll/debug logs are silent

# DOM selectors (music grid)
BASE_BODY = "#app div.e1pgfmdu0"
THREE_COL = "#main-content-single_song"
MUSIC_LIST = "#music-item-list"
GRID_ITEMS = '#music-item-list [id^="grid-item-container-"]'
ANCHOR_IN_ITEM = "div.css-ghnkqr-7937d88b--DivContainer-7937d88b--StyledDivContainerV2.eip9vuq0 > div > div > a"
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
]

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


def get_input_rows(sheets_svc, spreadsheet_id: str) -> List[Dict]:
    """Read A:B from first sheet. Return list of {song_name, audio_url, row_index} (row_index 1-based)."""
    return _get_input_rows_impl(sheets_svc, spreadsheet_id)


def _get_input_rows_impl(sheets_svc, spreadsheet_id: str) -> List[Dict]:
    try:
        result = sheets_svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="A:B"  # Read columns A and B
        ).execute()
        
        values = result.get("values", [])
        if not values:
            logger.warning("No data found in input sheet")
            return []
        
        rows = []
        for idx, row in enumerate(values[1:], start=2):
            if len(row) < 2 or not row[0] or not row[1]:
                continue  # Skip empty rows
            
            rows.append({
                "song_name": row[0].strip(),
                "audio_url": row[1].strip(),
                "row_index": idx
            })
        
        logger.info("Found %s input rows", len(rows))
        return rows

    except HttpError as e:
        logger.error("Error reading input sheet: %s", e)
        raise


def ensure_output_link_column(sheets_svc, spreadsheet_id: str) -> None:
    """Ensure 'Output Link' exists in header (column C); append if missing."""
    _ensure_output_link_column_impl(sheets_svc, spreadsheet_id)


def _ensure_output_link_column_impl(sheets_svc, spreadsheet_id: str) -> None:
    try:
        # Read header row
        result = sheets_svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="1:1"  # First row only
        ).execute()
        
        headers = result.get("values", [[]])[0] if result.get("values") else []
        
        # Check if "Output Link" exists
        if "Output Link" not in headers:
            # Append "Output Link" to header row
            headers.append("Output Link")
            
            sheets_svc.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range="1:1",
                valueInputOption="RAW",
                body={"values": [headers]}
            ).execute()
            
            logger.info("Added 'Output Link' column to input sheet")
        else:
            logger.info("'Output Link' column already exists")
            
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
        
    except HttpError as e:
        logger.error("Error appending rows: %s", e)
        raise


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
                {"range": "C%d" % row_index, "values": [[output_url]]}
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
    """Write output_url to column C at row_index. Uses queue for batching."""
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


def wait_for_new_items_and_extension_settle(
    page: Page,
    prev_count: int,
    wait_timeout_ms: int = 5000,
    settle_seconds: float = SORT_SETTLE_SECONDS,
) -> bool:
    """
    Wait until TikTok loads more GRID_ITEMS than prev_count.
    Adaptive settle: only sleep settle_seconds when new items actually arrived.
    Returns True if new items loaded, else False.
    """
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
        time.sleep(settle_seconds)
    return loaded


def scroll_until_views_below_threshold(
    page: Page,
    root_selector: str,
    threshold: int = VIEWS_THRESHOLD,
    max_scrolls: int = MAX_SCROLLS,
):
    """
    Scrolls until a full new scroll loads new items AND all newly loaded items
    are below the views threshold.
    """

    threshold_reached = False
    max_count_seen = 0
    iterations_no_increase = 0
    prev_count = int(page.evaluate(_GRID_COUNT_JS, GRID_ITEMS))

    for scroll_num in range(max_scrolls):
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
            new_count, _ = get_grid_count_and_last_views(page)
            prev_count = new_count
            logger.info("Stopping: scroll at bottom (no movement), newCount=%s", new_count)
            threshold_reached = True
            break

        loaded = wait_for_new_items_and_extension_settle(page, prev_count=prev_count)
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
            if prev_count == 0:
                break

            try:
                scroll_ret = scroll_to_bottom(page)
            except Exception:
                continue

            scroll_moved = scroll_ret.get("after") != scroll_ret.get("before")
            if not scroll_moved:
                break
            loaded = wait_for_new_items_and_extension_settle(page, prev_count=prev_count)
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
            output_rows.append([
                video_link,
                username,
                profile_prefix + username,
                views,
                parse_count(row["likes"]),
                parse_count(row["comments"]),
                parse_count(row["saves"]),
                parse_count(row["shares"]),
                row.get("duration", ""),
                row.get("upload_date", ""),
                row.get("upload_time", ""),
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
    try:
        page.goto(audio_url, wait_until="domcontentloaded", timeout=60000)
        setattr(page, "_scroller_handle", None)

        if check_tiktok_blocker(page):
            logger.warning("TikTok blocker/captcha detected")
            input("TikTok may be blocking. Fix it in the open browser, then press Enter...")

        root_selector = wait_for_grid_and_extension(page)
        if DEBUG_SCROLL:
            counts = _get_grid_counts(page)
            logger.info("[DEBUG] After wait_for_grid_and_extension: url=%s root=%s items=%s anchors=%s metas=%s",
                page.url[:80], root_selector, counts["items"], counts["anchors"], counts["metas"])
            logger.info("[DEBUG] Before scroll_until_views_below_threshold")

        scroll_until_views_below_threshold(page, root_selector)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] After scroll_until_views_below_threshold returned")

        time.sleep(0.1)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] Before scrape_loaded_items")

        output_rows, grid_count = scrape_loaded_items(page, root_selector)

        if DEBUG_SCROLL:
            logger.info("[DEBUG] After scrape_loaded_items: rows=%s grid_count=%s", len(output_rows), grid_count)
        return output_rows, grid_count
        
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
                write_output_link_back(sheets_svc, input_spreadsheet_id, row_index, output_url)

                stat["output_sheet_url"] = output_url
                stat["status"] = "success"
                stat["end_ts"] = time.perf_counter()
                stat["elapsed_seconds"] = stat["end_ts"] - start_ts
                run_stats.append(stat)
                pbar.set_postfix_str("%s | %s v | %ss" % (song_name[:18], n_videos, int(stat["elapsed_seconds"])))
                logger.info("Completed %s: %s videos -> %s", song_name, n_videos, output_url)

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
        lines.append("[%s] %s  %s  %s videos  %s  %s" % (
            s["status"],
            s["song_name"][:35],
            _format_elapsed(s["elapsed_seconds"]),
            s[VIDEOS_KEPT_FIELD],
            s["output_sheet_url"] or "-",
            (" " + s["error_message"][:60]) if s["error_message"] else "",
        ))
    lines.append("=" * 70)
    report = "\n".join(lines)

    print(report)
    logger.info("\n%s", report)


if __name__ == "__main__":
    main()
