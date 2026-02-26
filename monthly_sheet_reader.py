# !pip install google-api-python-client google-auth

"""
Monthly payments aggregator for Deepnote.

Reads all tabs from the source payments spreadsheet, normalizes TikTok profiles
and Song Title values, aggregates metrics by (Month, Member), and rebuilds the
destination "Monthly" tab on every run using a safe temp-tab rotation.
"""

import os
import re
import time
import traceback
import json
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


# =============================================================================
# CONFIGURATION
# =============================================================================

# SOURCE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1sKoJT1tyH3OFOG0ZQQpzOpJbPK7owxk075-ZtX0Qbrg/edit?usp=sharing"
# SOURCE_SPREADSHEET_ID = "1sKoJT1tyH3OFOG0ZQQpzOpJbPK7owxk075-ZtX0Qbrg"

SOURCE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1EXrm0FbudIu44LmgSKXZt2lf96XYdOuwyRWf-6yhXus/edit?usp=sharing"
SOURCE_SPREADSHEET_ID = "1EXrm0FbudIu44LmgSKXZt2lf96XYdOuwyRWf-6yhXus"

DEST_SHEET_URL = "https://docs.google.com/spreadsheets/d/1RS29y36hwIIrgavJXH6NbI1SCqSr4ha4DsXwErS3aPc/edit?usp=sharing"
DEST_SPREADSHEET_ID = "1RS29y36hwIIrgavJXH6NbI1SCqSr4ha4DsXwErS3aPc"
DEST_MONTHLY_TAB_NAME = "Monthly"

# Column mapping (0-based index). Headers assumed on row 1; data from row 2.
COLUMNS = {
    "Name": 0,
    "Song Title": 1,
    "Type": 2,
    "Type of Compensation": 3,
    "TikTok Profile": 4,
    "Paypal/Crypto": 5,
    "Price per edit": 6,
    "# edits / # views": 7,
    "Total amount (USD)": 8,
    "Genre": 9,
    "Requested by": 10,
    "Payment Status": 11,
    "Payment date": 12,
}

PAYMENT_STATUS_PAID = "PAID"
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "auto_auth.json")
DEBUG_OUTPUT_DIR = "logs/monthly_debug"
WRITE_DEBUG_ARTIFACTS = True

OUTPUT_HEADERS = [
    "Month",
    "Member",
    "# of Songs",
    "# of pages commissioned",
    "# of UNIQUE pages commissioned",
    "# of NEW UNIQUE pages commissioned",
    "# of Videos Posted",
    "Average videos per editor",
    "Total Spent",
    "Average cost per video",
]


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

_INVISIBLE_CHARS_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_TIKTOK_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?tiktok\.com/@[\w._-]+",
    re.IGNORECASE,
)
_AT_USER_PATTERN = re.compile(r"@[\w._-]+")
_URL_TOKEN_PATTERN = re.compile(
    r"(?:https?://[^\s,;\"'<>]+|www\.[^\s,;\"'<>]+|(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s,;\"'<>]+)",
    re.IGNORECASE,
)
_HANDLE_TOKEN_PATTERN = re.compile(r"(?<![\w.])@[\w._-]{2,}", re.IGNORECASE)
_TRACKING_QS_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "igshid", "si", "feature", "fbclid", "gclid",
}
DEBUG_CSV_FIELDS = [
    "row_id",
    "tab_name",
    "source_row_number",
    "payment_date_raw",
    "strict_month_key",
    "strict_month_source",
    "fallback_month_key",
    "fallback_month_source",
    "month_key",
    "month_source",
    "requested_by",
    "profile_platform",
    "profile_key",
    "profile_key_final",
    "profile_canonical_url",
    "profile_canonical_handle",
    "merge_alias_to",
    "merge_applied",
    "videos_posted_num",
    "total_spent_num",
    "finished_filter_pass",
]


def normalize_profile_link(link: str) -> str:
    """Normalize raw profile cell text for consistent parsing."""
    if not link:
        return ""
    s = str(link)
    s = _INVISIBLE_CHARS_RE.sub("", s)
    s = s.replace("\u00A0", " ")
    return s.strip().strip("\"'").strip()


def normalize_username(raw_username: str) -> Optional[str]:
    """Normalize username to lowercase without leading @."""
    if raw_username is None:
        return None
    uname = str(raw_username).strip().strip("\"'").strip().lower().lstrip("@")
    return uname or None


def normalize_profile_url(raw: str) -> Optional[str]:
    """Backward-compatible TikTok-only canonicalizer wrapper."""
    ident = canonicalize_profile_token(raw)
    if ident.get("platform") == "tiktok":
        return ident.get("canonical_url")
    return None


def _clean_url_token(token: str) -> Optional[str]:
    t = normalize_profile_link(token)
    if not t:
        return None
    if t.lower().startswith("www."):
        t = "https://" + t
    elif "://" not in t and re.match(r"^[a-z0-9-]+\.[a-z]{2,}/", t, re.IGNORECASE):
        t = "https://" + t
    if "://" not in t:
        return None
    return t


def _normalized_query(query: str) -> str:
    if not query:
        return ""
    pairs = parse_qsl(query, keep_blank_values=True)
    filtered = [(k, v) for k, v in pairs if k.lower() not in _TRACKING_QS_KEYS]
    return urlencode(filtered, doseq=True)


def _normalize_generic_url(url: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (canonical_url, host, path_lower_no_trailing_slash) or None.
    """
    try:
        p = urlparse(url)
    except Exception:
        return None
    host = (p.netloc or "").lower().strip()
    if not host:
        return None
    host = host[4:] if host.startswith("www.") else host
    path = (p.path or "").strip()
    path = re.sub(r"/+", "/", path).rstrip("/")
    if not path:
        path = "/"
    query = _normalized_query(p.query or "")
    canonical = urlunparse(("https", host, path, "", query, ""))
    return canonical, host, path.lower()


def _first_path_segment(path: str) -> str:
    segs = [s for s in (path or "").split("/") if s]
    return segs[0] if segs else ""


def extract_profile_tokens(cell: Any) -> List[str]:
    """
    Extract candidate profile tokens from a cell.
    Priority extraction: URLs first, then @handles, then text fragments fallback.
    """
    if cell is None:
        return []
    raw = normalize_profile_link(str(cell))
    if not raw:
        return []

    tokens: List[str] = []
    seen: Set[str] = set()

    for m in _URL_TOKEN_PATTERN.finditer(raw):
        tok = normalize_profile_link(m.group(0))
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for m in _HANDLE_TOKEN_PATTERN.finditer(raw):
        tok = normalize_profile_link(m.group(0))
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    if tokens:
        return tokens

    for part in re.split(r"[\n,;|]+", raw):
        t = normalize_profile_link(part)
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)
    return tokens


def canonicalize_profile_token(token: str) -> Dict[str, Any]:
    """
    Canonicalize a token into a unified identity structure:
    {platform, key, canonical_url, canonical_handle, source_type}
    """
    tok = normalize_profile_link(token)
    if not tok:
        return {
            "platform": "text",
            "key": "",
            "canonical_url": "",
            "canonical_handle": "",
            "source_type": "text",
        }

    # Handle-only token (@name) – keep as text handle bucket for now.
    if tok.startswith("@") and len(tok) > 1:
        h = normalize_username(tok) or ""
        return {
            "platform": "text",
            "key": f"text:{h}" if h else "",
            "canonical_url": "",
            "canonical_handle": h,
            "source_type": "handle",
        }

    url_tok = _clean_url_token(tok)
    if url_tok:
        norm = _normalize_generic_url(url_tok)
        if norm:
            canonical_url, host, path_lower = norm
            seg0 = _first_path_segment(path_lower)

            # TikTok
            if host.endswith("tiktok.com"):
                handle = normalize_username(seg0) if seg0.startswith("@") else None
                if handle:
                    return {
                        "platform": "tiktok",
                        "key": f"tiktok:{handle}",
                        "canonical_url": f"https://www.tiktok.com/@{handle}",
                        "canonical_handle": handle,
                        "source_type": "url",
                    }

            # Instagram
            if host.endswith("instagram.com"):
                reserved = {"p", "reel", "reels", "tv", "explore", "accounts"}
                handle = normalize_username(seg0) if seg0 and seg0 not in reserved else None
                if handle:
                    return {
                        "platform": "instagram",
                        "key": f"instagram:{handle}",
                        "canonical_url": f"https://www.instagram.com/{handle}",
                        "canonical_handle": handle,
                        "source_type": "url",
                    }

            # YouTube
            if host.endswith("youtube.com") or host == "youtu.be":
                handle = normalize_username(seg0) if seg0.startswith("@") else None
                if handle:
                    return {
                        "platform": "youtube",
                        "key": f"youtube:{handle}",
                        "canonical_url": f"https://www.youtube.com/@{handle}",
                        "canonical_handle": handle,
                        "source_type": "url",
                    }
                segs = [s for s in path_lower.split("/") if s]
                if len(segs) >= 2 and segs[0] in {"channel", "user", "c"}:
                    ch = normalize_username(segs[1]) or segs[1]
                    return {
                        "platform": "youtube",
                        "key": f"youtube:{segs[0]}:{ch}",
                        "canonical_url": canonical_url,
                        "canonical_handle": ch,
                        "source_type": "url",
                    }

            # Generic URL domain fallback.
            generic_key = f"domain:{host}:{path_lower or '/'}"
            return {
                "platform": f"domain:{host}",
                "key": generic_key,
                "canonical_url": canonical_url,
                "canonical_handle": normalize_username(seg0) if seg0 else "",
                "source_type": "url",
            }

    # Non-url text fallback
    txt = re.sub(r"\s+", " ", tok.casefold()).strip()
    return {
        "platform": "text",
        "key": f"text:{txt}" if txt else "",
        "canonical_url": "",
        "canonical_handle": "",
        "source_type": "text",
    }


def pick_primary_profile_identity(cell: Any) -> Optional[Dict[str, Any]]:
    """
    Pick a deterministic primary identity from a profile cell.
    URL tokens preferred, then text; platform priority among URLs:
    tiktok > instagram > youtube > other domains > text.
    """
    tokens = extract_profile_tokens(cell)
    if not tokens:
        return None
    identities = []
    for i, t in enumerate(tokens):
        ident = canonicalize_profile_token(t)
        if ident.get("key"):
            identities.append((i, ident))
    if not identities:
        return None

    def _prio(ident: Dict[str, Any]) -> int:
        p = ident.get("platform", "")
        s = ident.get("source_type", "")
        if p == "tiktok":
            return 0
        if p == "instagram":
            return 1
        if p == "youtube":
            return 2
        if s == "url":
            return 3
        return 4

    identities.sort(key=lambda x: (_prio(x[1]), x[0]))
    out = dict(identities[0][1])
    out["tokens_raw"] = tokens
    out["tokens_canonical"] = [it[1].get("key", "") for it in identities]
    return out


def normalize_tiktok_profile_cell_preserve_format(cell: Any) -> str:
    """
    Normalize TikTok URLs in a cell to canonical profile URLs, while preserving
    separators/format.
    """
    # Backward-compat wrapper; keep original behavior for TikTok URL replacement.
    if cell is None:
        return ""
    raw = str(cell)
    if not raw:
        return ""
    result = raw
    for m in reversed(list(_TIKTOK_URL_PATTERN.finditer(raw))):
        match_str = m.group(0)
        canon = normalize_profile_url(match_str)
        if canon:
            start, end = m.span()
            result = result[:start] + canon + result[end:]
    return result


def _split_tiktok_profiles(cell: Any) -> List[str]:
    """
    Split profile cell that may contain multiple values into URL/@username tokens.
    """
    # Backward-compat wrapper.
    return extract_profile_tokens(cell)


def canonical_primary_tiktok_profile(cell: Any) -> Optional[str]:
    """Backward-compat wrapper returning TikTok canonical URL when possible."""
    ident = pick_primary_profile_identity(cell)
    if not ident:
        return None
    if ident.get("platform") == "tiktok":
        return ident.get("canonical_url")
    return None


def normalize_song_title(value: Any) -> str:
    """
    Conservative Song Title normalization:
    trim, remove invisible chars/NBSP, collapse whitespace, casefold.
    """
    if value is None:
        return ""
    s = str(value)
    s = _INVISIBLE_CHARS_RE.sub("", s)
    s = s.replace("\u00A0", " ")
    s = s.strip().strip("\"'").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


# =============================================================================
# RATE LIMITING + API WRAPPERS
# =============================================================================

class RateLimiter:
    """Simple per-minute limiter + retry wrapper for Sheets API calls."""

    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []

    def _clean_old_requests(self) -> None:
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]

    def _wait_if_needed(self) -> None:
        self._clean_old_requests()
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest = min(self.request_times)
            wait_time = 60 - (time.time() - oldest) + 1
            if wait_time > 0:
                print(f"    [Rate limit] Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self._clean_old_requests()

    def _record(self) -> None:
        self.request_times.append(time.time())
        time.sleep(0.1)

    def execute_with_retry(
        self,
        request_func,
        operation_name: str,
        max_retries: int = 4,
    ):
        self._wait_if_needed()
        for attempt in range(max_retries):
            try:
                result = request_func()
                self._record()
                return result
            except (TimeoutError, OSError) as e:
                if attempt < max_retries - 1:
                    wait = max(3, (2 ** attempt) + 2)
                    print(f"    [Retry] {operation_name}: {str(e)[:80]}... waiting {wait}s")
                    time.sleep(wait)
                    self.request_times = []
                else:
                    return None
            except Exception as e:
                err = str(e)
                if "429" in err or "RATE_LIMIT" in err or "Quota exceeded" in err:
                    if attempt < max_retries - 1:
                        wait = max(2, (2 ** attempt) + 1)
                        print(f"    [Retry] {operation_name}: {err[:80]}... waiting {wait}s")
                        time.sleep(wait)
                        self.request_times = []
                    else:
                        return None
                else:
                    raise
        return None


_read_limiter = RateLimiter(max_requests_per_minute=50)
_write_limiter = RateLimiter(max_requests_per_minute=50)
_batch_limiter = RateLimiter(max_requests_per_minute=50)


def setup_sheets_service():
    """Create Sheets API service using service-account credentials."""
    if not os.path.isfile(CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"Credentials file not found: {CREDENTIALS_PATH}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or add auto_auth.json."
        )
    creds = Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds)


def _sheets_get_metadata(service, spreadsheet_id: str, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    def _req():
        return service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

    out = _read_limiter.execute_with_retry(_req, f"get metadata {spreadsheet_id[:12]}...")
    if out is None:
        stats["api_failures"] += 1
    return out


def _sheets_get_values(
    service,
    spreadsheet_id: str,
    range_name: str,
    stats: Dict[str, Any],
    value_render_option: str = "FORMATTED_VALUE",
    date_time_render_option: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    def _req():
        params: Dict[str, Any] = {
            "spreadsheetId": spreadsheet_id,
            "range": range_name,
            "valueRenderOption": value_render_option,
        }
        if date_time_render_option:
            params["dateTimeRenderOption"] = date_time_render_option
        return service.spreadsheets().values().get(**params).execute()

    out = _read_limiter.execute_with_retry(_req, f"get {range_name}")
    if out is None:
        stats["api_failures"] += 1
    return out


def _sheets_update_values_safe(
    service,
    spreadsheet_id: str,
    range_name: str,
    values: List[List[Any]],
    stats: Dict[str, Any],
    value_input_option: str = "USER_ENTERED",
) -> bool:
    def _req():
        return service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body={"values": values},
        ).execute()

    out = _write_limiter.execute_with_retry(_req, f"update {range_name}")
    if out is None:
        stats["api_failures"] += 1
    return out is not None


def _sheets_batch_update_safe(
    service,
    spreadsheet_id: str,
    requests: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    def _req():
        return service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        ).execute()

    out = _batch_limiter.execute_with_retry(_req, f"batch update ({len(requests)} req)")
    if out is None:
        stats["api_failures"] += 1
    return out


def _get_sheet_id_by_tab_name(
    service,
    spreadsheet_id: str,
    tab_name: str,
    stats: Dict[str, Any],
) -> Optional[int]:
    meta = _sheets_get_metadata(service, spreadsheet_id, stats)
    if not meta:
        return None
    for s in meta.get("sheets", []):
        if s.get("properties", {}).get("title") == tab_name:
            return s["properties"].get("sheetId")
    return None

# =============================================================================
# PARSING / DATA PREP
# =============================================================================

def row_has_data(row: List[Any]) -> bool:
    return any(cell is not None and str(cell).strip() for cell in (row or []))


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_payment_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    num = _parse_number(value)
    if num is not None and num > 0:
        try:
            excel_epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
            return excel_epoch + timedelta(days=float(num))
        except (ValueError, TypeError, OverflowError):
            pass

    parts = s.split("/")
    if len(parts) == 3:
        try:
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 100:
                y += 2000 if y < 50 else 1900
            return datetime(y, m, d)
        except (TypeError, ValueError):
            pass

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s[:10], fmt)
        except (TypeError, ValueError):
            continue
    return None


def _month_key(dt: datetime) -> str:
    return dt.strftime("%B %Y")


def ensure_debug_dir(run_id: str) -> Path:
    """Create and return a run-specific debug directory."""
    p = Path(DEBUG_OUTPUT_DIR) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_debug_json(path: Path, obj: Any) -> None:
    """Write debug json artifact."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_debug_csv(path: Path, rows: List[Dict[str, Any]], field_order: List[str]) -> None:
    """Write debug CSV artifact with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_tab_month(tab_name: str) -> Optional[datetime]:
    """
    Parse month/year from flexible tab title text.
    Supports examples:
      - January 2026
      - 🔒 January 2026
      - January 2026 - Campaigns
      - Campaigns January 2026
    Returns first day of month on success.
    """
    if not tab_name:
        return None
    raw = str(tab_name).strip()
    if not raw:
        return None
    s = raw.lower()
    month_aliases = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month_pattern = (
        r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|"
        r"july|jul|august|aug|september|sept|sep|october|oct|november|nov|december|dec)\b"
    )
    year_pattern = r"\b(20\d{2}|19\d{2})\b"

    m_month = re.search(month_pattern, s, re.IGNORECASE)
    if not m_month:
        return None
    month_token = m_month.group(1).lower()
    month_num = month_aliases.get(month_token)
    if not month_num:
        return None

    years = list(re.finditer(year_pattern, s))
    if not years:
        return None
    month_pos = m_month.start()
    best_year = min(years, key=lambda y: abs(y.start() - month_pos))
    year_num = int(best_year.group(1))
    if year_num < 1900 or year_num > 2100:
        return None
    return datetime(year_num, month_num, 1)


def _debug_rows_projection(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select debug-friendly row fields for artifact dumps."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "row_id": r.get("row_id"),
                "tab_name": r.get("tab_name"),
                "source_row_number": r.get("source_row_number"),
                "payment_date_raw": r.get("payment_date_raw"),
                "strict_month_key": r.get("strict_month_key"),
                "strict_month_source": r.get("strict_month_source"),
                "fallback_month_key": r.get("fallback_month_key"),
                "fallback_month_source": r.get("fallback_month_source"),
                "month_key": r.get("month_key"),
                "month_source": r.get("month_source"),
                "requested_by": r.get("requested_by"),
                "profile_platform": r.get("profile_platform"),
                "profile_key": r.get("profile_key"),
                "profile_key_final": r.get("profile_key_final"),
                "profile_canonical_url": r.get("profile_canonical_url"),
                "profile_canonical_handle": r.get("profile_canonical_handle"),
                "merge_alias_to": r.get("merge_alias_to"),
                "merge_applied": r.get("merge_applied"),
                "videos_posted_num": r.get("videos_posted_num"),
                "total_spent_num": r.get("total_spent_num"),
                "finished_filter_pass": r.get("finished_filter_pass"),
            }
        )
    return out


def resolve_row_month(
    payment_date_cell: Any,
    tab_name: str,
    mode: str = "strict",
) -> Tuple[Optional[datetime], str, str, bool]:
    """
    Resolve row month with priority:
    1) Payment date month if parseable
    2) Tab month fallback when payment date is missing/invalid

    Returns:
      (month_start_dt, month_key, source, date_tab_mismatch)
      source in {'payment_date', 'tab_name', 'none'}
    """
    payment_dt = _parse_payment_date(payment_date_cell)
    tab_month_dt = parse_tab_month(tab_name)
    mismatch = False

    if payment_dt and tab_month_dt:
        mismatch = (payment_dt.year != tab_month_dt.year) or (payment_dt.month != tab_month_dt.month)

    if mode == "tab_forced" and tab_month_dt:
        return tab_month_dt, _month_key(tab_month_dt), "tab_name", mismatch

    if payment_dt:
        month_start = datetime(payment_dt.year, payment_dt.month, 1)
        return month_start, _month_key(month_start), "payment_date", mismatch

    if tab_month_dt:
        return tab_month_dt, _month_key(tab_month_dt), "tab_name", mismatch

    return None, "", "none", mismatch


def _first_day_of_current_month(now: Optional[datetime] = None) -> datetime:
    """Return first day of current month in local runtime time."""
    base = now or datetime.now()
    return datetime(base.year, base.month, 1)


def month_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return month label to count mapping for diagnostics."""
    counter = Counter()
    for r in rows:
        counter[r["month_key"]] += 1
    return dict(
        sorted(
            counter.items(),
            key=lambda kv: datetime.strptime(kv[0], "%B %Y"),
            reverse=True,
        )
    )


def filter_finished_month_rows(
    rows: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], int, List[str], List[str]]:
    """
    Keep only rows from finished months (month_start_dt < first day of current month).
    Returns (filtered_rows, dropped_count, included_months_desc, excluded_months_desc).
    """
    current_month_start = _first_day_of_current_month(now)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    included_months: Set[str] = set()
    excluded_months: Set[str] = set()

    for row in rows:
        month_start = row["month_start_dt"]
        if month_start < current_month_start:
            kept.append(row)
            included_months.add(row["month_key"])
        else:
            dropped += 1
            excluded_months.add(row["month_key"])

    def _month_sort_desc(months: Set[str]) -> List[str]:
        return sorted(
            months,
            key=lambda m: datetime.strptime(m, "%B %Y"),
            reverse=True,
        )

    return kept, dropped, _month_sort_desc(included_months), _month_sort_desc(excluded_months)


def recompute_month_fields(
    rows: List[Dict[str, Any]],
    mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Recompute month fields in existing normalized rows.
    Returns (rebuilt_rows, source_counters).
    """
    counters = {"payment_date": 0, "tab_name": 0, "none": 0, "mismatch": 0}
    rebuilt: List[Dict[str, Any]] = []

    for row in rows:
        month_start_dt, month_key, month_source, mismatch = resolve_row_month(
            row.get("payment_date_raw"),
            row.get("tab_name", ""),
            mode=mode,
        )
        if mismatch:
            counters["mismatch"] += 1
        counters[month_source] = counters.get(month_source, 0) + 1
        if month_source == "none" or month_start_dt is None:
            continue

        r = dict(row)
        r["month_start_dt"] = month_start_dt
        r["month_key"] = month_key
        r["month_source"] = month_source
        if mode == "tab_forced":
            r["fallback_month_key"] = month_key
            r["fallback_month_source"] = month_source
        rebuilt.append(r)

    return rebuilt, counters


def _has_consecutive_run(nums: List[int], run_len: int = 3) -> bool:
    if len(nums) < run_len:
        return False
    s = sorted(set(nums))
    streak = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            streak += 1
            if streak >= run_len:
                return True
        else:
            streak = 1
    return False


def apply_conservative_drag_merge(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, int]]:
    """
    Conservative merge for drag-generated suffix sequences.
    Scope: same (month_key, requested_by, profile_platform), handle-based keys only.
    """
    alias_map: Dict[str, str] = {}
    stats = {"drag_merge_groups": 0, "drag_merge_aliases_applied": 0}

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r.get("month_key", ""), r.get("requested_by", ""), r.get("profile_platform", ""))].append(r)

    for (_month, _member, platform), grp in grouped.items():
        if platform not in {"tiktok", "instagram", "youtube"}:
            continue

        by_base: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # base -> [(profile_key, suffix_num)]
        counts: Counter = Counter()
        first_seen: Dict[str, int] = {}
        order = 0

        for r in grp:
            k = r.get("profile_key", "")
            h = (r.get("profile_canonical_handle") or "").lower()
            if not k or not h:
                continue
            counts[k] += 1
            if k not in first_seen:
                first_seen[k] = order
                order += 1
            m = re.match(r"^(.*?)(\d{1,4})$", h)
            if not m:
                continue
            base = m.group(1)
            suf = int(m.group(2))
            if not base:
                continue
            by_base[base].append((k, suf))

        for base, pairs in by_base.items():
            nums = [p[1] for p in pairs]
            if not _has_consecutive_run(nums, run_len=3):
                continue
            unique_keys = sorted(set(p[0] for p in pairs))
            if len(unique_keys) < 2:
                continue
            stats["drag_merge_groups"] += 1
            winner = max(unique_keys, key=lambda kk: (counts.get(kk, 0), -first_seen.get(kk, 999999)))
            for loser in unique_keys:
                if loser != winner:
                    alias_map[loser] = winner

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        src_key = rr.get("profile_key", "")
        dst_key = alias_map.get(src_key, src_key)
        rr["profile_key_final"] = dst_key
        rr["merge_applied"] = dst_key != src_key
        rr["merge_alias_to"] = dst_key if dst_key != src_key else ""
        if rr["merge_applied"]:
            stats["drag_merge_aliases_applied"] += 1
        out.append(rr)
    return out, alias_map, stats


def load_all_tabs_paid_rows(
    service,
    spreadsheet_id: str,
    stats: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Read all tabs A:M and return normalized PAID rows only.
    """
    meta = _sheets_get_metadata(service, spreadsheet_id, stats)
    if not meta:
        raise RuntimeError("Could not load source spreadsheet metadata.")
    tab_names = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if not tab_names:
        return [], {}

    dropped = defaultdict(int)
    rows_out: List[Dict[str, Any]] = []

    for tab_order, tab_name in enumerate(tab_names):
        range_name = f"'{tab_name}'!A:M"
        resp = _sheets_get_values(
            service,
            spreadsheet_id,
            range_name,
            stats,
            value_render_option="UNFORMATTED_VALUE",
            date_time_render_option="SERIAL_NUMBER",
        )
        if not resp:
            print(f"  [WARN] Could not read tab '{tab_name}', skipping")
            dropped["tab_read_failure"] += 1
            continue
        values = resp.get("values", [])
        if len(values) <= 1:
            continue

        for row_idx_1based, row in enumerate(values[1:], start=2):
            stats["rows_read"] += 1
            if not row_has_data(row):
                dropped["empty_row"] += 1
                continue

            def _get(col_name: str) -> Any:
                i = COLUMNS[col_name]
                return row[i] if i < len(row) else ""

            status = str(_get("Payment Status") or "").strip().upper()
            if status != PAYMENT_STATUS_PAID:
                dropped["non_paid_status"] += 1
                continue

            requested_by = str(_get("Requested by") or "").strip()
            if not requested_by:
                dropped["missing_requested_by"] += 1
                continue

            month_start_dt, month_key, month_source, date_tab_mismatch = resolve_row_month(
                _get("Payment date"),
                tab_name,
                mode="strict",
            )
            if date_tab_mismatch:
                stats["rows_with_date_tab_mismatch"] += 1
            if month_source == "payment_date":
                stats["rows_with_valid_payment_date"] += 1
            elif month_source == "tab_name":
                stats["rows_with_tab_month_fallback"] += 1
            else:
                dropped["invalid_payment_date"] += 1
                stats["rows_dropped_no_usable_month"] += 1
                continue

            profile_ident = pick_primary_profile_identity(_get("TikTok Profile"))
            if not profile_ident or not profile_ident.get("key"):
                dropped["invalid_profile_identity"] += 1
                continue
            source_type = profile_ident.get("source_type", "")
            if source_type == "url":
                stats["rows_with_url_profile"] += 1
            else:
                stats["rows_with_text_profile"] += 1
            p = profile_ident.get("platform", "text")
            platform_counts = stats.setdefault("platform_counts", {})
            platform_counts[p] = platform_counts.get(p, 0) + 1

            song_norm = normalize_song_title(_get("Song Title"))
            videos_num = _parse_number(_get("# edits / # views")) or 0.0
            spent_num = _parse_number(_get("Total amount (USD)")) or 0.0

            rows_out.append(
                {
                    "row_id": f"{tab_name}:{row_idx_1based}",
                    "tab_name": tab_name,
                    "tab_order": tab_order,
                    "source_row_number": row_idx_1based,
                    "requested_by": requested_by,
                    "song_title_norm": song_norm,
                    "profile_platform": profile_ident.get("platform", "text"),
                    "profile_key": profile_ident.get("key", ""),
                    "profile_key_final": profile_ident.get("key", ""),
                    "profile_canonical_url": profile_ident.get("canonical_url", ""),
                    "profile_canonical_handle": profile_ident.get("canonical_handle", ""),
                    "profile_tokens_raw": profile_ident.get("tokens_raw", []),
                    "profile_tokens_canonical": profile_ident.get("tokens_canonical", []),
                    "payment_date_dt": _parse_payment_date(_get("Payment date")),
                    "payment_date_raw": _get("Payment date"),
                    "month_key": month_key,
                    "month_start_dt": month_start_dt,
                    "month_source": month_source,
                    "strict_month_key": month_key,
                    "strict_month_source": month_source,
                    "fallback_month_key": "",
                    "fallback_month_source": "",
                    "merge_alias_to": "",
                    "merge_applied": False,
                    "finished_filter_pass": "none",
                    "videos_posted_num": videos_num,
                    "total_spent_num": spent_num,
                }
            )
            stats["rows_kept"] += 1

    stats["tabs_scanned"] = len(tab_names)
    return rows_out, dict(dropped)


def build_first_paid_dictionaries(
    rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, datetime], Dict[str, str], Dict[str, str]]:
    """
    Build first-paid dictionaries by final canonical profile key.
    Tie-breaker for same profile/date uses first (tab_order, source_row_number).
    """
    first_seen: Dict[str, Tuple[datetime, int, int, str]] = {}
    for r in rows:
        profile = r.get("profile_key_final") or r.get("profile_key") or ""
        if not profile:
            continue
        dt = r.get("payment_date_dt") or r["month_start_dt"]
        tord = r["tab_order"]
        srow = r["source_row_number"]
        owner = r["requested_by"]
        existing = first_seen.get(profile)
        if existing is None or (dt, tord, srow) < (existing[0], existing[1], existing[2]):
            first_seen[profile] = (dt, tord, srow, owner)

    first_paid_date_by_profile = {k: v[0] for k, v in first_seen.items()}
    first_paid_month_by_profile = {k: _month_key(v[0]) for k, v in first_seen.items()}
    first_paid_owner_by_profile = {k: v[3] for k, v in first_seen.items()}
    return first_paid_date_by_profile, first_paid_month_by_profile, first_paid_owner_by_profile


def aggregate_monthly_metrics(
    rows: List[Dict[str, Any]],
    first_paid_month_by_profile: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Aggregate metrics by (month_key, requested_by).
    Returns sorted member rows and sorted month order labels.
    """
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    month_to_dt: Dict[str, datetime] = {}

    for r in rows:
        month_key = r["month_key"]
        member = r["requested_by"]
        key = (month_key, member)
        if key not in groups:
            groups[key] = {
                "Month": month_key,
                "Member": member,
                "_songs": set(),
                "_profiles": set(),
                "_rows": 0,
                "_videos_sum": 0.0,
                "_spent_sum": 0.0,
            }
        g = groups[key]
        song_key = r["song_title_norm"]
        if song_key:
            g["_songs"].add(song_key)
        profile_key = r.get("profile_key_final") or r.get("profile_key") or ""
        if profile_key:
            g["_profiles"].add(profile_key)
        g["_rows"] += 1
        g["_videos_sum"] += float(r["videos_posted_num"] or 0.0)
        g["_spent_sum"] += float(r["total_spent_num"] or 0.0)
        month_to_dt.setdefault(month_key, r["month_start_dt"])

    out_rows: List[Dict[str, Any]] = []
    for (month_key, member), g in groups.items():
        unique_profiles = g["_profiles"]
        unique_pages_count = len(unique_profiles)
        videos = g["_videos_sum"]
        spent = g["_spent_sum"]
        new_unique = {
            p for p in unique_profiles if first_paid_month_by_profile.get(p) == month_key
        }
        out_rows.append(
            {
                "Month": month_key,
                "Member": member,
                "# of Songs": len(g["_songs"]),
                "# of pages commissioned": g["_rows"],
                "# of UNIQUE pages commissioned": unique_pages_count,
                "# of NEW UNIQUE pages commissioned": len(new_unique),
                "# of Videos Posted": videos,
                "Average videos per editor": (videos / unique_pages_count) if unique_pages_count else 0.0,
                "Total Spent": spent,
                "Average cost per video": (spent / videos) if videos else 0.0,
            }
        )

    months_sorted = [
        m for m, _ in sorted(month_to_dt.items(), key=lambda kv: kv[1], reverse=True)
    ]

    month_pos = {m: i for i, m in enumerate(months_sorted)}
    out_rows.sort(key=lambda r: (month_pos[r["Month"]], r["Member"].casefold()))
    return out_rows, months_sorted


def _to_int_if_whole(v: float) -> Any:
    return int(v) if isinstance(v, float) and v.is_integer() else v


def _to_int_rounded(v: Any) -> int:
    """Force integer output for metrics that must display as whole numbers."""
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def build_output_values(
    aggregated_rows: List[Dict[str, Any]],
    months_sorted: List[str],
) -> Tuple[List[List[Any]], List[int], List[int], List[Tuple[int, int]]]:
    """
    Build monthly table blocks with separators.
    Returns (values, header_rows_1based, total_rows_1based, block_ranges_1based).
    """
    by_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in aggregated_rows:
        by_month[r["Month"]].append(r)

    values: List[List[Any]] = []
    header_rows: List[int] = []
    total_rows: List[int] = []
    block_ranges: List[Tuple[int, int]] = []

    if not months_sorted:
        header_rows.append(1)
        return [OUTPUT_HEADERS, ["No finished months to report.", "", "", "", "", "", "", "", "", ""]], header_rows, total_rows, block_ranges

    for month in months_sorted:
        members = sorted(by_month.get(month, []), key=lambda r: r["Member"].casefold())
        if not members:
            continue

        block_start = len(values) + 1
        header_rows.append(len(values) + 1)
        values.append(OUTPUT_HEADERS)

        month_spent_total = 0.0
        for r in members:
            month_spent_total += float(r["Total Spent"] or 0.0)
            values.append(
                [
                    r["Month"],
                    r["Member"],
                    _to_int_rounded(r["# of Songs"]),
                    _to_int_rounded(r["# of pages commissioned"]),
                    _to_int_rounded(r["# of UNIQUE pages commissioned"]),
                    _to_int_rounded(r["# of NEW UNIQUE pages commissioned"]),
                    _to_int_rounded(r["# of Videos Posted"]),
                    float(r["Average videos per editor"]),
                    float(r["Total Spent"]),
                    float(r["Average cost per video"]),
                ]
            )

        total_rows.append(len(values) + 1)
        values.append([month, "TOTAL", "", "", "", "", "", "", month_spent_total, ""])
        block_end = len(values)
        block_ranges.append((block_start, block_end))
        values.append([""] * len(OUTPUT_HEADERS))

    return values, header_rows, total_rows, block_ranges

# =============================================================================
# DESTINATION WRITING
# =============================================================================

def _column_to_a1(col_1_based: int) -> str:
    n = col_1_based
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _build_format_requests(
    sheet_id: int,
    row_count: int,
    header_rows_1based: List[int],
    total_rows_1based: List[int],
    block_ranges_1based: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []

    if row_count > 0:
        reqs.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": row_count,
                        "startColumnIndex": 2,
                        "endColumnIndex": 7,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "0"}}},
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )
        reqs.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": row_count,
                        "startColumnIndex": 7,
                        "endColumnIndex": 8,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}}},
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )
        reqs.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": row_count,
                        "startColumnIndex": 8,
                        "endColumnIndex": 10,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": {"type": "CURRENCY", "pattern": "$#,##0.00"}}},
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )

    for r in header_rows_1based:
        reqs.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r - 1,
                        "endRowIndex": r,
                        "startColumnIndex": 0,
                        "endColumnIndex": 10,
                    },
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                    "fields": "userEnteredFormat.textFormat.bold",
                }
            }
        )

    for r in total_rows_1based:
        reqs.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r - 1,
                        "endRowIndex": r,
                        "startColumnIndex": 0,
                        "endColumnIndex": 10,
                    },
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                    "fields": "userEnteredFormat.textFormat.bold",
                }
            }
        )

    # Draw borders + banded table style for each month block.
    for start_row, end_row in block_ranges_1based:
        reqs.append(
            {
                "updateBorders": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row - 1,
                        "endRowIndex": end_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": 10,
                    },
                    "top": {"style": "SOLID", "width": 1},
                    "bottom": {"style": "SOLID", "width": 1},
                    "left": {"style": "SOLID", "width": 1},
                    "right": {"style": "SOLID", "width": 1},
                    "innerHorizontal": {"style": "SOLID", "width": 1},
                    "innerVertical": {"style": "SOLID", "width": 1},
                }
            }
        )
        reqs.append(
            {
                "addBanding": {
                    "bandedRange": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": start_row - 1,
                            "endRowIndex": end_row,
                            "startColumnIndex": 0,
                            "endColumnIndex": 10,
                        },
                        "rowProperties": {
                            "headerColor": {"red": 0.86, "green": 0.92, "blue": 0.98},
                            "firstBandColor": {"red": 0.96, "green": 0.98, "blue": 1.0},
                            "secondBandColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        },
                    }
                }
            }
        )

    reqs.append(
        {
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": 10,
                }
            }
        }
    )
    return reqs


def rebuild_monthly_tab(
    service,
    spreadsheet_id: str,
    values: List[List[Any]],
    header_rows_1based: List[int],
    total_rows_1based: List[int],
    block_ranges_1based: List[Tuple[int, int]],
    stats: Dict[str, Any],
) -> None:
    """Create temp tab, write/format, delete old Monthly, rename temp to Monthly."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_title = f"__monthly_build_{ts}"
    row_count = max(100, len(values) + 10)
    col_count = max(10, len(OUTPUT_HEADERS))

    add_req = {
        "addSheet": {
            "properties": {
                "title": temp_title,
                "gridProperties": {"rowCount": row_count, "columnCount": col_count},
            }
        }
    }
    add_resp = _sheets_batch_update_safe(service, spreadsheet_id, [add_req], stats)
    if not add_resp:
        raise RuntimeError("Failed to create temporary destination tab.")

    replies = add_resp.get("replies", [])
    if not replies or "addSheet" not in replies[0]:
        raise RuntimeError("Unexpected addSheet response; could not resolve temp sheet ID.")
    temp_sheet_id = replies[0]["addSheet"]["properties"]["sheetId"]

    last_col = _column_to_a1(len(OUTPUT_HEADERS))
    last_row = max(1, len(values))
    range_name = f"'{temp_title}'!A1:{last_col}{last_row}"
    if not _sheets_update_values_safe(service, spreadsheet_id, range_name, values, stats):
        raise RuntimeError("Failed to write data to temporary destination tab.")

    fmt_requests = _build_format_requests(
        sheet_id=temp_sheet_id,
        row_count=len(values),
        header_rows_1based=header_rows_1based,
        total_rows_1based=total_rows_1based,
        block_ranges_1based=block_ranges_1based,
    )
    if _sheets_batch_update_safe(service, spreadsheet_id, fmt_requests, stats) is None:
        raise RuntimeError("Failed to format temporary destination tab.")

    old_monthly_sheet_id = _get_sheet_id_by_tab_name(service, spreadsheet_id, DEST_MONTHLY_TAB_NAME, stats)
    finalize_reqs: List[Dict[str, Any]] = []
    if old_monthly_sheet_id is not None:
        finalize_reqs.append({"deleteSheet": {"sheetId": old_monthly_sheet_id}})
    finalize_reqs.append(
        {
            "updateSheetProperties": {
                "properties": {"sheetId": temp_sheet_id, "title": DEST_MONTHLY_TAB_NAME},
                "fields": "title",
            }
        }
    )
    if _sheets_batch_update_safe(service, spreadsheet_id, finalize_reqs, stats) is None:
        raise RuntimeError("Failed to rotate destination Monthly tab (delete/rename).")


# =============================================================================
# MAIN
# =============================================================================

def main() -> Dict[str, Any]:
    """
    Entry point.
    Returns run statistics for notebook/debug usage.
    """
    run_start = time.perf_counter()
    start_dt = datetime.now()
    debug_run_id = start_dt.strftime("%Y%m%d_%H%M%S")
    debug_dir = ensure_debug_dir(debug_run_id) if WRITE_DEBUG_ARTIFACTS else None
    stats: Dict[str, Any] = {
        "start_time": start_dt.isoformat(timespec="seconds"),
        "rows_read": 0,
        "rows_kept": 0,
        "tabs_scanned": 0,
        "api_failures": 0,
        "month_blocks_written": 0,
        "member_rows_written": 0,
        "unique_profiles": 0,
        "months_detected_before_filter": 0,
        "months_after_finished_filter": 0,
        "rows_dropped_current_month": 0,
        "rows_with_valid_payment_date": 0,
        "rows_with_tab_month_fallback": 0,
        "rows_with_date_tab_mismatch": 0,
        "rows_dropped_no_usable_month": 0,
        "finished_filter_cutoff_month": "",
        "rows_with_url_profile": 0,
        "rows_with_text_profile": 0,
        "platform_counts": {},
        "used_tab_month_zero_finished_fallback": False,
        "rows_recovered_by_tab_month_fallback": 0,
        "months_after_fallback": 0,
        "strict_finished_rows": 0,
        "fallback_finished_rows": 0,
        "drag_merge_groups": 0,
        "drag_merge_aliases_applied": 0,
        "debug_run_id": debug_run_id,
        "debug_output_dir": str(debug_dir) if debug_dir else "",
    }

    print("=" * 72)
    print("  MONTHLY PAYMENTS AGGREGATOR - RUN START")
    print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print(f"  Source sheet: {SOURCE_SPREADSHEET_ID}")
    print(f"  Destination sheet: {DEST_SPREADSHEET_ID}")

    try:
        print("\n" + "-" * 72)
        print("  STEP 1: Setup Sheets service")
        print("-" * 72)
        service = setup_sheets_service()

        print("\n" + "-" * 72)
        print("  STEP 2: Load and normalize PAID rows from all source tabs")
        print("-" * 72)
        rows, dropped = load_all_tabs_paid_rows(service, SOURCE_SPREADSHEET_ID, stats)
        print(f"  Tabs scanned: {stats['tabs_scanned']}")
        print(f"  Rows read: {stats['rows_read']}")
        print(f"  Rows kept (PAID + valid): {stats['rows_kept']}")
        if dropped:
            print("  Dropped breakdown:")
            for k, v in sorted(dropped.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    - {k}: {v}")

        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_csv(
                debug_dir / "01_rows_normalized.csv",
                _debug_rows_projection(rows),
                DEBUG_CSV_FIELDS,
            )

        months_before_map = month_distribution(rows)
        stats["months_detected_before_filter"] = len(months_before_map)
        if months_before_map:
            print("  Month distribution before finished-month filter:")
            for m, c in months_before_map.items():
                print(f"    - {m}: {c}")
        print("  Month resolution diagnostics:")
        print(f"    - rows_with_valid_payment_date: {stats['rows_with_valid_payment_date']}")
        print(f"    - rows_with_tab_month_fallback: {stats['rows_with_tab_month_fallback']}")
        print(f"    - rows_with_date_tab_mismatch: {stats['rows_with_date_tab_mismatch']}")
        print(f"    - rows_dropped_no_usable_month: {stats['rows_dropped_no_usable_month']}")
        print("  Profile identity diagnostics:")
        print(f"    - rows_with_url_profile: {stats['rows_with_url_profile']}")
        print(f"    - rows_with_text_profile: {stats['rows_with_text_profile']}")
        print(f"    - platform_counts: {stats.get('platform_counts', {})}")
        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_json(debug_dir / "month_counts_strict.json", months_before_map)

        print("\n" + "-" * 72)
        print("  STEP 3: Keep only finished months (exclude current month)")
        print("-" * 72)
        cutoff_dt = _first_day_of_current_month()
        stats["finished_filter_cutoff_month"] = _month_key(cutoff_dt)
        print(f"  Finished-month cutoff: {stats['finished_filter_cutoff_month']} (excluded and newer)")
        rows_before_finished_filter = list(rows)
        rows, dropped_current, included_months, excluded_months = filter_finished_month_rows(rows)
        for r in rows:
            r["finished_filter_pass"] = "strict"
        stats["strict_finished_rows"] = len(rows)
        stats["rows_dropped_current_month"] = dropped_current
        stats["months_after_finished_filter"] = len(included_months)
        print(f"  Rows kept after finished-month filter: {len(rows)}")
        print(f"  Rows dropped as current/future month: {dropped_current}")
        if included_months:
            print(f"  Included months (newest->oldest): {included_months}")
        if excluded_months:
            print(f"  Excluded months (current/future): {excluded_months}")

        months_after_map = month_distribution(rows)
        if months_after_map:
            print("  Month distribution after finished-month filter:")
            for m, c in months_after_map.items():
                print(f"    - {m}: {c}")
        elif stats["rows_kept"] > 0:
            print("  [DEBUG] No finished-month rows kept. Sample row month sources:")
            for sample in rows_before_finished_filter[:5]:
                print(
                    f"    - tab={sample.get('tab_name')} "
                    f"payment_date_raw={sample.get('payment_date_raw')} "
                    f"resolved_source={sample.get('month_source')} "
                    f"month={sample.get('month_key')}"
                )

        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_csv(
                debug_dir / "02_rows_after_strict_finished_filter.csv",
                _debug_rows_projection(rows),
                DEBUG_CSV_FIELDS,
            )
            write_debug_json(debug_dir / "month_counts_after_strict_filter.json", months_after_map)

        if len(rows) == 0 and len(rows_before_finished_filter) > 0:
            print("  [WARN] Strict mode produced zero finished months; applying tab-month fallback recovery.")
            rebound_rows, rebound_counters = recompute_month_fields(rows_before_finished_filter, mode="tab_forced")
            fallback_before_map = month_distribution(rebound_rows)
            if fallback_before_map:
                print("  Fallback month distribution before finished filter:")
                for m, c in fallback_before_map.items():
                    print(f"    - {m}: {c}")
            print(
                "  Fallback month source counters: "
                f"payment_date={rebound_counters.get('payment_date', 0)}, "
                f"tab_name={rebound_counters.get('tab_name', 0)}, "
                f"none={rebound_counters.get('none', 0)}, "
                f"mismatch={rebound_counters.get('mismatch', 0)}"
            )

            fallback_rows, fallback_dropped_current, fallback_included_months, fallback_excluded_months = (
                filter_finished_month_rows(rebound_rows)
            )
            for r in fallback_rows:
                r["finished_filter_pass"] = "fallback"
            stats["fallback_finished_rows"] = len(fallback_rows)
            fallback_after_map = month_distribution(fallback_rows)
            if fallback_after_map:
                print("  Fallback month distribution after finished filter:")
                for m, c in fallback_after_map.items():
                    print(f"    - {m}: {c}")
            if fallback_included_months:
                print(f"  Fallback included months (newest->oldest): {fallback_included_months}")
            if fallback_excluded_months:
                print(f"  Fallback excluded months (current/future): {fallback_excluded_months}")

            print("  Fallback sample rows (up to 5):")
            for sample in rebound_rows[:5]:
                print(
                    f"    - tab={sample.get('tab_name')} "
                    f"payment_date_raw={sample.get('payment_date_raw')} "
                    f"strict_month={sample.get('strict_month_key')} "
                    f"fallback_month={sample.get('fallback_month_key')} "
                    f"final_source={sample.get('month_source')}"
                )

            if len(fallback_rows) > 0:
                rows = fallback_rows
                stats["used_tab_month_zero_finished_fallback"] = True
                stats["rows_recovered_by_tab_month_fallback"] = len(fallback_rows)
                stats["months_after_fallback"] = len(fallback_included_months)
                stats["months_after_finished_filter"] = len(fallback_included_months)
                stats["rows_dropped_current_month"] = fallback_dropped_current
            else:
                stats["months_after_fallback"] = 0

            if WRITE_DEBUG_ARTIFACTS and debug_dir:
                write_debug_csv(
                    debug_dir / "03_rows_after_tab_fallback_filter.csv",
                    _debug_rows_projection(fallback_rows),
                    DEBUG_CSV_FIELDS,
                )
                write_debug_json(debug_dir / "month_counts_after_fallback_filter.json", fallback_after_map)

        if len(rows) == 0 and len(rows_before_finished_filter) > 0:
            tab_counter = Counter(r.get("tab_name", "") for r in rows_before_finished_filter)
            source_counter = Counter(r.get("month_source", "") for r in rows_before_finished_filter)
            print("\n" + "!" * 72)
            print("  WARNING: NO FINISHED MONTHS AFTER STRICT + FALLBACK")
            print(f"  Cutoff month: {stats['finished_filter_cutoff_month']}")
            print(f"  Top tabs seen: {tab_counter.most_common(5)}")
            print(f"  Month source counts: {dict(source_counter)}")
            print("  Sample rows (up to 5):")
            for sample in rows_before_finished_filter[:5]:
                print(
                    f"    - tab={sample.get('tab_name')} date={sample.get('payment_date_raw')} "
                    f"strict={sample.get('strict_month_key')} "
                    f"fallback={sample.get('fallback_month_key')} "
                    f"source={sample.get('month_source')}"
                )
            print("!" * 72)

        print("\n" + "-" * 72)
        print("  STEP 3.5: Apply conservative drag-error merge")
        print("-" * 72)
        rows, alias_map, merge_stats = apply_conservative_drag_merge(rows)
        stats["drag_merge_groups"] = merge_stats.get("drag_merge_groups", 0)
        stats["drag_merge_aliases_applied"] = merge_stats.get("drag_merge_aliases_applied", 0)
        print(f"  Drag merge groups: {stats['drag_merge_groups']}")
        print(f"  Drag merge aliases applied: {stats['drag_merge_aliases_applied']}")
        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_json(debug_dir / "drag_merge_alias_map.json", alias_map)
            write_debug_json(debug_dir / "profile_platform_counts.json", stats.get("platform_counts", {}))

        print("\n" + "-" * 72)
        print("  STEP 4: Build first-paid dictionaries")
        print("-" * 72)
        first_paid_date_by_profile, first_paid_month_by_profile, first_paid_owner_by_profile = (
            build_first_paid_dictionaries(rows)
        )
        stats["unique_profiles"] = len(first_paid_date_by_profile)
        print(f"  Unique canonical profiles: {stats['unique_profiles']}")
        print(f"  First-paid owner mappings: {len(first_paid_owner_by_profile)}")

        print("\n" + "-" * 72)
        print("  STEP 5: Aggregate by month/member")
        print("-" * 72)
        aggregated_rows, months_sorted = aggregate_monthly_metrics(rows, first_paid_month_by_profile)
        stats["month_blocks_written"] = len(months_sorted)
        stats["member_rows_written"] = len(aggregated_rows)
        print(f"  Month blocks: {stats['month_blocks_written']}")
        if months_sorted:
            print(f"  Block order (newest->oldest): {months_sorted}")
        print(f"  Member rows: {stats['member_rows_written']}")

        print("\n" + "-" * 72)
        print("  STEP 6: Build output table values")
        print("-" * 72)
        values, header_rows, total_rows, block_ranges = build_output_values(aggregated_rows, months_sorted)
        print(f"  Output rows (including headers/totals/separators): {len(values)}")
        print(f"  Header rows: {len(header_rows)} | Total rows: {len(total_rows)} | Table blocks: {len(block_ranges)}")

        print("\n" + "-" * 72)
        print("  STEP 7: Rebuild destination Monthly tab")
        print("-" * 72)
        rebuild_monthly_tab(
            service=service,
            spreadsheet_id=DEST_SPREADSHEET_ID,
            values=values,
            header_rows_1based=header_rows,
            total_rows_1based=total_rows,
            block_ranges_1based=block_ranges,
            stats=stats,
        )
        print("  Destination tab rotation completed successfully.")

        duration = time.perf_counter() - run_start
        stats["duration_seconds"] = round(duration, 2)
        stats["status"] = "success"
        stats["end_time"] = datetime.now().isoformat(timespec="seconds")

        print("\n" + "=" * 72)
        print("  RUN SUMMARY")
        print("=" * 72)
        for k in [
            "status",
            "tabs_scanned",
            "rows_read",
            "rows_kept",
            "months_detected_before_filter",
            "months_after_finished_filter",
            "rows_dropped_current_month",
            "rows_with_valid_payment_date",
            "rows_with_tab_month_fallback",
            "rows_with_date_tab_mismatch",
            "rows_dropped_no_usable_month",
            "finished_filter_cutoff_month",
            "rows_with_url_profile",
            "rows_with_text_profile",
            "platform_counts",
            "drag_merge_groups",
            "drag_merge_aliases_applied",
            "used_tab_month_zero_finished_fallback",
            "rows_recovered_by_tab_month_fallback",
            "months_after_fallback",
            "strict_finished_rows",
            "fallback_finished_rows",
            "debug_run_id",
            "debug_output_dir",
            "unique_profiles",
            "month_blocks_written",
            "member_rows_written",
            "api_failures",
            "duration_seconds",
        ]:
            print(f"  {k}: {stats.get(k)}")
        print("=" * 72)
        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_json(debug_dir / "run_summary_debug.json", stats)
        return stats

    except Exception as exc:
        duration = time.perf_counter() - run_start
        stats["duration_seconds"] = round(duration, 2)
        stats["status"] = "failed"
        stats["end_time"] = datetime.now().isoformat(timespec="seconds")
        stats["error"] = str(exc)

        print("\n" + "=" * 72)
        print("  RUN FAILED")
        print("=" * 72)
        print(f"  Error: {exc}")
        traceback.print_exc()
        print("=" * 72)
        if WRITE_DEBUG_ARTIFACTS and debug_dir:
            write_debug_json(debug_dir / "run_summary_debug.json", stats)
        return stats


if __name__ == "__main__":
    main()
