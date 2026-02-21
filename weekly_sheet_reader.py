# !pip install gspread google-api-python-client google-auth gspread-formatting pandas

"""
Weekly sheet reader for Deepnote.
Reads the leftmost tab from the payments sheet, filters PAID rows,
groups by Requested by and TikTok Profile, and aggregates.
Combined DB "Sheets Inputs" tab provides the list of names (Requested by categories).
"""

import os
import re
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


def normalize_profile_link(link: str) -> str:
    """Normalize LINK cell for stable keying without changing navigation URL semantics."""
    if not link:
        return ""
    s = str(link)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", s)
    s = s.replace("\u00A0", " ")
    return s.strip().strip("\"'").strip()


def normalize_username(raw_username: str) -> Optional[str]:
    """Normalize username to lowercase without leading @."""
    if raw_username is None:
        return None
    uname = str(raw_username).strip().strip("\"'").strip().lower().lstrip("@")
    return uname or None


def normalize_profile_url(raw: str) -> Optional[str]:
    """
    Canonicalize TikTok profile URL to https://www.tiktok.com/@username.
    Returns None when input is not a valid profile URL.
    """
    raw_norm = normalize_profile_link(raw)
    if not raw_norm:
        return None
    candidate = raw_norm
    if "://" not in candidate:
        candidate = f"https://{candidate.lstrip('/')}"
    try:
        parsed = urlparse(candidate)
    except Exception:
        return None
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        bare_host = host[4:]
    else:
        bare_host = host
    if not (bare_host == "tiktok.com" or bare_host.endswith(".tiktok.com")):
        return None
    if bare_host == "vt.tiktok.com":
        return None
    path = (parsed.path or "").strip()
    if not path:
        return None
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None
    first = segments[0]
    if not first.startswith("@"):
        return None
    username = first[1:]
    if not username:
        return None
    if not re.fullmatch(r"[\w\.-]+", username):
        return None
    if len(segments) == 1:
        pass
    elif len(segments) == 2 and segments[1] == "likes":
        pass
    elif len(segments) >= 3 and segments[1] in {"video", "photo"}:
        pass
    else:
        return None
    return f"https://www.tiktok.com/@{username}"


def normalize_link_key(link: str) -> str:
    """Case-insensitive key for link-based row mapping."""
    canonical = normalize_profile_url(link)
    if canonical:
        return canonical.lower()
    return normalize_profile_link(link).lower()


def canonical_tiktok_identity(
    username_cell: Any, profile_cell: Any
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Return (username_norm, link_key, canonical_profile_url) for a profile cell.
    - username_norm: normalized username (lower, no @) from profile or username_cell
    - link_key: stable key for matching (from normalize_link_key)
    - canonical_profile_url: https://www.tiktok.com/@user if valid else None
    """
    profile_str = str(profile_cell).strip() if profile_cell is not None else ""
    canonical_profile_url = normalize_profile_url(profile_str)
    link_key = normalize_link_key(profile_str)
    username_str = str(username_cell).strip() if username_cell is not None else ""
    username_norm = normalize_username(username_str)
    if not username_norm and canonical_profile_url:
        path = (urlparse(canonical_profile_url).path or "").strip("/").lstrip("@")
        username_norm = normalize_username(path)
    if not username_norm and profile_str:
        username_norm = normalize_username(profile_str)
    return (username_norm, link_key, canonical_profile_url)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Payments sheet (leftmost tab)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1sKoJT1tyH3OFOG0ZQQpzOpJbPK7owxk075-ZtX0Qbrg/edit?usp=sharing"
SPREADSHEET_ID = "1sKoJT1tyH3OFOG0ZQQpzOpJbPK7owxk075-ZtX0Qbrg"

# Combined Database sheet – first tab "Sheets Inputs"
COMBINED_DB_URL = "https://docs.google.com/spreadsheets/d/1VIll8-H7_j3BwknGKwiPRFW0LMIeKcIMV6bZrx2UZ14/edit?usp=sharing"
COMBINED_DB_SPREADSHEET_ID = "1VIll8-H7_j3BwknGKwiPRFW0LMIeKcIMV6bZrx2UZ14"
SHEETS_INPUTS_TAB = "Sheets Inputs"
SHEETS_INPUT_RANGE = "Sheets Inputs!A2:B"  # Name (col A), Sheet URL (col B)

# Combined DB – all tab names (from tiktok.py)
COMBINED_DB_TAB_NAMES = {
    "SHEETS_INPUTS": "Sheets Inputs",
    "GOOD_COMBINED": "Good Accounts Combined",
    "RELIABLE_COMBINED": "Reliable Accounts Combined",
}

# Combined DB "Sheets Inputs" tab – column mapping (0-based)
COMBINED_DB_SHEETS_INPUT_COLUMNS = {
    "Name": 0,        # Column A – Campaign Manager Name (Requested by category)
    "Sheet URL": 1,   # Column B – URL to the individual sheet for this name
}

# Combined DB "Good Accounts Combined" & "Reliable Accounts Combined" tabs
# Campaign Manager Name is column A; columns B–V = source sheet columns (shifted +1)
# Same structure for both tabs
COMBINED_DB_AGGREGATE_COLUMNS = {
    "CAMPAIGN_MANAGER_NAME": 0,  # Column A
    "USERNAME": 1,               # Column B – TikTok USERNAME
    "LINK": 2,                   # Column C – Link (TikTok URL)
    "NICHE": 3,                  # Column D – Niche
    "TIMES_BOOKED": 4,           # Column E – # of Times Booked
    "LAST_PAYMENT_DATE": 5,      # Column F – Last Payment Date
    "MANUAL_APPROVE": 6,         # Column G – Manual Approve (checkbox)
    "TIKTOK_PRICE": 7,           # Column H – TikTok Price
    "PERF_BASED_PRICE": 8,       # Column I – Perf Based Price ($$ per 100k)
    "COST_100K": 9,              # Column J – Cost for 100k Views (formula)
    "MEDIAN_VIEWS": 10,          # Column K – 15 Videos - Median Views
    "CHANGE_MEDIAN": 11,         # Column L – Change in Median Views
    "UNDER_10K": 12,             # Column M – < 10k
    "BETWEEN_10K_100K": 13,      # Column N – 10k - 100k
    "OVER_100K": 14,             # Column O – > 100k
    "FIFTEENTH_DATE": 15,        # Column P – 15 Videos Ago Date
    "LATEST_DATE": 16,           # Column Q – Latest Video Date
    "CONTACT_INFO": 17,          # Column R – 2nd Contact info
    "GENRE": 18,                 # Column S – Genre
    "COUNTRY": 19,               # Column T – Country
    "TYPE": 20,                  # Column U – Type (Editor / Faceless / Niche Leader)
    "PAYMENT_INFO": 21,          # Column V – Paypal Info / Crypto Wallet
}

COMBINED_DB_AGGREGATE_HEADER_ROW = 2
COMBINED_DB_AGGREGATE_DATA_START_ROW = 3

# Credentials: use auto_auth.json in project, or set GOOGLE_APPLICATION_CREDENTIALS
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "auto_auth.json")

# -----------------------------------------------------------------------------
# Individual sheets (from Sheet URL) – same format as tiktok.py
# Each linked sheet has these tabs and columns.
# -----------------------------------------------------------------------------

# Tab names in each individual (source) sheet
SOURCE_SHEET_TAB_NAMES = {
    "MAIN": "Initial Reachout",
    "BAD": "Bad Accounts",
    "GOOD": "Good Accounts",
    "RELIABLE": "Reliable Accounts",
}

# Column mapping for each individual sheet (headers on row 2, data from row 3)
SOURCE_SHEET_COLUMNS = {
    "USERNAME": 0,             # Column A – TikTok USERNAME
    "LINK": 1,                # Column B – Link (TikTok URL)
    "NICHE": 2,                # Column C – Niche
    "TIMES_BOOKED": 3,         # Column D – # of Times Booked
    "LAST_PAYMENT_DATE": 4,    # Column E – Last Payment Date
    "MANUAL_APPROVE": 5,       # Column F – Manual Approve (checkbox)
    "TIKTOK_PRICE": 6,         # Column G – TikTok Price
    "PERF_BASED_PRICE": 7,     # Column H – Perf Based Price ($$ per 100k)
    "COST_100K": 8,            # Column I – Cost for 100k Views
    "MEDIAN_VIEWS": 9,         # Column J – 15 Videos - Median Views
    "CHANGE_MEDIAN": 10,       # Column K – Change in Median Views
    "UNDER_10K": 11,           # Column L – < 10k
    "BETWEEN_10K_100K": 12,    # Column M – 10k - 100k
    "OVER_100K": 13,           # Column N – > 100k
    "FIFTEENTH_DATE": 14,      # Column O – 15 Videos Ago Date
    "LATEST_DATE": 15,         # Column P – Latest Video Date
    "CONTACT_INFO": 16,        # Column Q – 2nd Contact info
    "GENRE": 17,               # Column R – Genre
    "COUNTRY": 18,             # Column S – Country
    "TYPE": 19,                # Column T – Type (Editor / Faceless / Niche Leader)
    "PAYMENT_INFO": 20,        # Column U – Paypal Info / Crypto Wallet
}

SOURCE_SHEET_HEADER_ROW = 2   # 1-based
SOURCE_SHEET_DATA_START_ROW = 3

# Cost for 100k Views formula: =(100000*G{row})/J{row} (G=TikTok Price, J=Median Views)
COST_100K_FORMULA = "=(100000*G{row})/J{row}"

# -----------------------------------------------------------------------------
# Payments sheet (leftmost tab) – column mapping
# -----------------------------------------------------------------------------

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

# Columns to ignore for processing (we still load them; they are dropped before grouping)
COLUMNS_TO_IGNORE = {"Song Title", "Type", "Paypal/Crypto", "Total amount (USD)"}

# Literal value required for Payment Status to keep a row
PAYMENT_STATUS_PAID = "PAID"

# Type of Compensation categories – controls where Price per edit goes
TYPE_COMP_FLAT_FEE = "flat fee"
TYPE_COMP_DISCORD_SERVER = "discord server"
TYPE_COMP_PERFORMANCE_BASED = "performance based"

HEADER_ROW = 1   # 1-based: row 1 = headers
DATA_START_ROW = 2   # 1-based: data starts here


def _normalize_tiktok_link(url_or_profile: str) -> Optional[str]:
    """
    Normalize TikTok URL or @username to lowercase username (no @) for comparison.
    Returns None if invalid.
    """
    canon = normalize_profile_url(url_or_profile)
    if canon:
        path = (urlparse(canon).path or "").strip("/").lstrip("@")
        return normalize_username(path)
    return normalize_username(url_or_profile)


def _extract_tiktok_username(value: Any) -> str:
    """Extract normalized TikTok username (no @, no URL). Returns \"\" if empty/invalid."""
    n = _normalize_tiktok_link(str(value).strip() if value is not None else "")
    return n or ""


# Regexes for extracting multiple TikTok profiles from a single cell
_TIKTOK_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?tiktok\.com/@[\w._-]+",
    re.IGNORECASE,
)
_AT_USER_PATTERN = re.compile(r"@[\w._-]+")


def normalize_tiktok_profile_cell_preserve_format(cell: Any) -> str:
    """
    Normalize TikTok profile tokens inside a cell without changing separators/formatting.
    Replaces each tiktok.com/@... URL with canonical form only. Does not convert plain @username.
    Does not change non-TikTok text, newlines, spaces, or commas.
    """
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


def pick_primary_tiktok_profile_from_cell(profile_cell: Any) -> Optional[str]:
    """Extract the first/primary TikTok profile from a cell. Returns None if none found."""
    profiles = _split_tiktok_profiles(profile_cell)
    return profiles[0] if profiles else None


def _split_tiktok_profiles(cell: Any) -> List[str]:
    """
    Split a cell that may contain multiple TikTok profile links into a list of
    single profile strings (URLs or @username). Used to explode Payments rows
    where one cell has newline/space/comma-separated or quoted multiple links.

    - Prefer regex extraction so we don't split on spaces inside other text.
    - Strip surrounding quotes and whitespace; deduplicate while preserving order.
    - Fallback: split by lines and whitespace, return non-empty tokens only.
    """
    if cell is None:
        return []
    raw = str(cell).strip().strip('"\'')
    if not raw:
        return []

    # 1) Find all TikTok URLs (tiktok.com/@...)
    urls = _TIKTOK_URL_PATTERN.findall(raw)
    if urls:
        seen: Set[str] = set()
        out = []
        for u in urls:
            s = u.strip().strip('"\'')
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # 2) No URLs: find all @username tokens
    at_users = _AT_USER_PATTERN.findall(raw)
    if at_users:
        seen = set()  # type: Set[str]
        out = []
        for a in at_users:
            s = a.strip().strip('"\'')
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # 3) Fallback: split by newlines then whitespace; non-empty only
    tokens = []
    for line in raw.splitlines():
        for part in line.replace(",", " ").split():
            s = part.strip().strip('"\'')
            if s:
                tokens.append(s)
    # Deduplicate preserving order
    seen: Set[str] = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_spreadsheet_id_from_url(url: str) -> Optional[str]:
    """Extract spreadsheet ID from a Google Sheets URL."""
    if not url:
        return None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None


# =============================================================================
# RATE LIMITING AND SHEET API WRAPPERS (from tiktok.py pattern)
# =============================================================================

class RateLimiter:
    """Manages rate limiting for Google Sheets API to avoid quota errors."""

    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []

    def _clean_old_requests(self) -> None:
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]

    def _wait_if_needed(self) -> None:
        self._clean_old_requests()
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest = min(self.request_times)
            wait_time = 60 - (time.time() - oldest) + 1
            if wait_time > 0:
                print(f"    [Rate limit] Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self._clean_old_requests()

    def _record_request(self) -> None:
        self.request_times.append(time.time())
        time.sleep(0.1)

    def execute_with_retry(
        self,
        request_func,
        max_retries: int = 3,
        operation_name: str = "API request",
    ):
        self._wait_if_needed()
        for attempt in range(max_retries):
            try:
                result = request_func()
                self._record_request()
                return result
            except (TimeoutError, OSError) as e:
                err = str(e)
                if attempt < max_retries - 1:
                    wait = max(5, (2 ** attempt) + 3)
                    print(f"    [Retry] {operation_name}: {err[:60]}... waiting {wait}s")
                    time.sleep(wait)
                    self.request_times = []
                else:
                    print(f"    [WARNING] {operation_name} failed after {max_retries} retries (timeout)")
                    return None
            except Exception as e:
                err = str(e)
                if "429" in err or "RATE_LIMIT" in err or "Quota exceeded" in err:
                    if attempt < max_retries - 1:
                        wait = max(2, (2 ** attempt) + 1)
                        print(f"    [Retry] {operation_name}: {err[:60]}... waiting {wait}s")
                        time.sleep(wait)
                        self.request_times = []
                    else:
                        print(f"    [WARNING] {operation_name} failed after {max_retries} retries")
                        return None
                else:
                    raise
        return None


_read_limiter = RateLimiter(max_requests_per_minute=50)
_write_limiter = RateLimiter(max_requests_per_minute=50)
_batch_limiter = RateLimiter(max_requests_per_minute=50)


def _sheets_get_values(
    service,
    spreadsheet_id: str,
    range_name: str,
    value_render_option: str = "FORMATTED_VALUE",
) -> Optional[Dict[str, Any]]:
    """Rate-limited get. Returns None on failure."""

    def _req():
        return service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption=value_render_option,
        ).execute()

    return _read_limiter.execute_with_retry(
        _req, max_retries=3, operation_name=f"get {range_name}"
    )


def _sheets_update_values_safe(
    service,
    spreadsheet_id: str,
    range_name: str,
    values: List[List[Any]],
    value_input_option: str = "USER_ENTERED",
) -> bool:
    """Rate-limited update. Returns True on success."""

    def _req():
        return service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body={"values": values},
        ).execute()

    result = _write_limiter.execute_with_retry(
        _req, max_retries=3, operation_name=f"update {range_name}"
    )
    return result is not None


def _sheets_clear_values_safe(
    service,
    spreadsheet_id: str,
    range_name: str,
) -> bool:
    """Rate-limited clear. Returns True on success."""

    def _req():
        return service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_name,
        ).execute()

    result = _write_limiter.execute_with_retry(
        _req, max_retries=3, operation_name=f"clear {range_name}"
    )
    return result is not None


def _get_sheet_id_by_tab_name(
    service,
    spreadsheet_id: str,
    tab_name: str,
) -> Optional[int]:
    """Return the grid sheetId (numeric) for the tab with the given name, or None."""
    meta = _sheets_get_metadata(service, spreadsheet_id)
    if not meta:
        return None
    for s in meta.get("sheets", []):
        if s.get("properties", {}).get("title") == tab_name:
            return s["properties"].get("sheetId")
    return None


def _sheets_delete_row_safe(
    service,
    spreadsheet_id: str,
    tab_name: str,
    row_num_1based: int,
) -> bool:
    """Delete one row in the given tab. row_num_1based is user row number (1-based). Returns True on success."""
    grid_sheet_id = _get_sheet_id_by_tab_name(service, spreadsheet_id, tab_name)
    if grid_sheet_id is None:
        return False
    # API uses 0-based indices; endIndex is exclusive
    start = row_num_1based - 1
    end = row_num_1based
    req = {
        "deleteDimension": {
            "range": {
                "sheetId": grid_sheet_id,
                "dimension": "ROWS",
                "startIndex": start,
                "endIndex": end,
            }
        }
    }
    return _sheets_batch_update_safe(service, spreadsheet_id, [req])


def _sheets_batch_update_safe(
    service,
    spreadsheet_id: str,
    requests: List[Dict[str, Any]],
) -> bool:
    """Rate-limited batchUpdate. Returns True on success."""

    def _req():
        return service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        ).execute()

    result = _batch_limiter.execute_with_retry(
        _req, max_retries=3, operation_name=f"batch update ({len(requests)} reqs)"
    )
    return result is not None


def _sheets_get_metadata(service, spreadsheet_id: str) -> Optional[Dict[str, Any]]:
    """Rate-limited get spreadsheet metadata."""

    def _req():
        return service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

    return _read_limiter.execute_with_retry(
        _req, max_retries=3, operation_name=f"get metadata {spreadsheet_id[:15]}..."
    )


# =============================================================================

def setup_sheets_service():
    """Build Google Sheets API service. Uses service account from file or env."""
    try:
        path = CREDENTIALS_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Credentials file not found: {path}. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or add auto_auth.json"
            )
        creds = Credentials.from_service_account_file(
            path,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        return build("sheets", "v4", credentials=creds)
    except Exception as e:
        print(f"ERROR setting up Google Sheets: {e}")
        return None


def get_leftmost_sheet_name(service, spreadsheet_id: str) -> Optional[str]:
    """Return the name of the first (leftmost) tab."""
    if not service:
        return None
    meta = _sheets_get_metadata(service, spreadsheet_id)
    if not meta:
        print("ERROR: Could not get sheet metadata")
        return None
    sheets = meta.get("sheets", [])
    if not sheets:
        return None
    return sheets[0]["properties"]["title"]


def row_has_data(row: List[Any]) -> bool:
    """True if the row has at least one non-empty cell."""
    if not row:
        return False
    for cell in row:
        if cell is not None and str(cell).strip():
            return True
    return False


def load_sheet_data(
    spreadsheet_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load all data from the leftmost tab for rows that have data.

    Returns a list of dicts, one per data row, with keys from COLUMNS.
    Empty rows are skipped.
    """
    sid = spreadsheet_id or SPREADSHEET_ID
    service = setup_sheets_service()
    if not service:
        return []

    tab_name = get_leftmost_sheet_name(service, sid)
    if not tab_name:
        print("No sheets found.")
        return []

    range_name = f"'{tab_name}'!A:M"
    result = _sheets_get_values(service, sid, range_name)
    if not result:
        print(f"ERROR: Could not read payments range {range_name}")
        return []

    values = result.get("values", [])
    if len(values) < DATA_START_ROW:
        return []

    # Pass 1: build raw records and collect all usernames seen in the sheet (Name + TikTok Profile).
    raw_records: List[Dict[str, Any]] = []
    seen_usernames: Set[str] = set()
    for row in values[HEADER_ROW:]:
        if not row_has_data(row):
            continue
        record = {}
        for col_name, idx in COLUMNS.items():
            value = row[idx] if idx < len(row) else ""
            record[col_name] = value if value is not None else ""
        record["TikTok Profile"] = normalize_tiktok_profile_cell_preserve_format(
            record.get("TikTok Profile", "")
        )
        raw_records.append(record)
        u = _extract_tiktok_username(record.get("Name", ""))
        if u:
            seen_usernames.add(u)
        for profile in _split_tiktok_profiles(record.get("TikTok Profile", "")):
            u = _extract_tiktok_username(profile)
            if u:
                seen_usernames.add(u)

    return raw_records


def load_combined_db_sheets(
    spreadsheet_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Load Name and Sheet URL from the Combined DB "Sheets Inputs" tab.
    Returns list of dicts with keys "Name", "Sheet URL".
    Uses SHEETS_INPUT_RANGE (A2:B) – row 1 = header, data from row 2.
    """
    sid = spreadsheet_id or COMBINED_DB_SPREADSHEET_ID
    service = setup_sheets_service()
    if not service:
        return []
    result = _sheets_get_values(service, sid, SHEETS_INPUT_RANGE)
    if not result:
        print("ERROR: Could not read Combined DB Sheets Inputs")
        return []
    values = result.get("values", [])
    out = []
    for row in values:
        name = (row[COMBINED_DB_SHEETS_INPUT_COLUMNS["Name"]] or "").strip() if len(row) > 0 else ""
        url = (row[COMBINED_DB_SHEETS_INPUT_COLUMNS["Sheet URL"]] or "").strip() if len(row) > 1 else ""
        if name or url:
            out.append({"Name": name, "Sheet URL": url})
    return out


def load_combined_db_names(
    spreadsheet_id: Optional[str] = None,
) -> List[str]:
    """
    Load the list of Names (column A) from the Combined DB "Sheets Inputs" tab.
    These correspond to the "Requested by" categories.
    """
    sheets = load_combined_db_sheets(spreadsheet_id)
    return [s["Name"] for s in sheets if s.get("Name")]


def _parse_number(value: Any) -> Optional[float]:
    """Parse a cell value to float; return None if empty or invalid."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# Date format: D/M/YYYY (e.g. 1/2/2026 — day and month without leading zeros)
PAYMENT_DATE_OUTPUT_FORMAT = "{d}/{m}/{y}"  # D/M/YYYY


def _parse_payment_date(value: Any) -> Optional[datetime]:
    """
    Parse Payment date: D/M/YYYY (e.g. 1/2/2026), or Excel serial number.
    Returns datetime or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Excel serial (number of days since 1899-12-30)
    num = _parse_number(value)
    if num is not None and num > 0:
        try:
            excel_epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
            return excel_epoch + timedelta(days=float(num))
        except (ValueError, TypeError, OverflowError):
            pass
    # D/M/YYYY (day and month can be 1 or 2 digits)
    parts = s.split("/")
    if len(parts) == 3:
        try:
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 100:
                y += 2000 if y < 50 else 1900
            return datetime(y, m, d)
        except (ValueError, TypeError):
            pass
    # Other string formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s[:10], fmt)
        except (ValueError, TypeError):
            continue
    return None


def _format_payment_date(dt: datetime) -> str:
    """Format datetime as D/M/YYYY (e.g. 1/2/2026)."""
    return PAYMENT_DATE_OUTPUT_FORMAT.format(d=dt.day, m=dt.month, y=dt.year)


def _mode(values: List[Any]) -> Any:
    """Return the most frequent non-empty value; if all empty, return empty string."""
    non_empty = [v for v in values if v is not None and str(v).strip()]
    if not non_empty:
        return ""
    return Counter(non_empty).most_common(1)[0][0]


def process_payment_data(
    data: List[Dict[str, Any]],
    only_categories_in_combined_db: bool = False,
    combined_db_names: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter by Payment Status == PAID, group by Requested by then TikTok Profile,
    and aggregate per group.

    - Rows with Payment Status != "PAID" (or empty) are ignored.
    - Columns Song Title, Type, Paypal/Crypto, Total amount (USD) are ignored.
    - Per (Requested by, TikTok Profile): sum(# edits / # views), latest(Payment date),
      and for other kept fields (including Price per edit) take the most repeating value.

    Returns: dict keyed by "Requested by" category, each value = list of aggregated
    rows (one per TikTok Profile in that category).
    """
    if combined_db_names is None and only_categories_in_combined_db:
        combined_db_names = load_combined_db_names()
    if only_categories_in_combined_db and combined_db_names:
        combined_set = {n.strip().lower() for n in combined_db_names}
    else:
        combined_set = None

    # Build list of keys we keep (all except ignored)
    keys_keep = [k for k in COLUMNS if k not in COLUMNS_TO_IGNORE]
    # Keys we aggregate specially (sum or latest only; rest use mode)
    key_sum = "# edits / # views"
    key_latest = "Payment date"
    keys_mode = [
        k for k in keys_keep
        if k not in (key_sum, key_latest, "TikTok Profile", "Requested by")
    ]

    paid_rows = []
    for row in data:
        status = (row.get("Payment Status") or "").strip().upper()
        if status != PAYMENT_STATUS_PAID:
            continue
        if combined_set is not None:
            requested = (row.get("Requested by") or "").strip()
            if requested.lower() not in combined_set:
                continue
        candidates = _split_tiktok_profiles(row.get("TikTok Profile", ""))
        if not candidates:
            continue
        has_valid_tiktok = any(normalize_profile_url(c) is not None for c in candidates)
        if not has_valid_tiktok:
            continue
        row_out = {k: row.get(k, "") for k in keys_keep}
        row_out["_payer"] = row.get("Paypal/Crypto", "")
        paid_rows.append(row_out)

    def _canonical_urls_from_profile_cell(cell: Any) -> List[str]:
        candidates = _split_tiktok_profiles(cell)
        out: List[str] = []
        seen: Set[str] = set()
        for c in candidates:
            canon = normalize_profile_url(c)
            if canon and canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out

    def _payer_date_key(r: Dict[str, Any]) -> Tuple[str, str]:
        payer = str(r.get("_payer") or "").strip().lower()
        dt = _parse_payment_date(r.get("Payment date"))
        dt_key = _format_payment_date(dt) if dt else str(r.get("Payment date") or "").strip()
        return (payer, dt_key)

    def _collapse_increment_typos(
        usernames: List[str],
        counts: Counter,
        first_seen_index: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Returns mapping loser_username -> winner_username for usernames that differ only
        by last digit (+/-1). Winner = most frequent; tie -> smaller first_seen_index.
        Only applies when both end with digit, same prefix, abs(last_a - last_b) == 1.
        """
        prefix_groups: Dict[str, List[str]] = {}
        for u in usernames:
            if not u or len(u) < 2:
                continue
            if u[-1].isdigit():
                prefix = u[:-1]
                prefix_groups.setdefault(prefix, []).append(u)
        alias_map: Dict[str, str] = {}
        for prefix, group in prefix_groups.items():
            if len(group) < 2:
                continue
            group_set = set(group)
            typo_pairs: List[Tuple[str, str]] = []
            for a in group:
                for b in group:
                    if a >= b:
                        continue
                    last_a, last_b = a[-1], b[-1]
                    if last_a.isdigit() and last_b.isdigit():
                        if abs(int(last_a) - int(last_b)) == 1:
                            typo_pairs.append((a, b))
            if not typo_pairs:
                continue
            all_in_group = list(set(a for pair in typo_pairs for a in pair))
            winner = max(
                all_in_group,
                key=lambda x: (counts.get(x, 0), -first_seen_index.get(x, 999999)),
            )
            for u in all_in_group:
                if u != winner:
                    alias_map[u] = winner
        return alias_map

    rows_by_requested: Dict[str, List[Dict[str, Any]]] = {}
    for row in paid_rows:
        requested = (row.get("Requested by") or "").strip()
        rows_by_requested.setdefault(requested, []).append(row)

    by_requested: Dict[str, List[Dict[str, Any]]] = {}
    for requested_by, rows in rows_by_requested.items():
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        component_rows: Dict[str, List[Dict[str, Any]]] = {}
        anchor_by_payer_date: Dict[Tuple[str, str], str] = {}
        for row in rows:
            urls = _canonical_urls_from_profile_cell(row.get("TikTok Profile", ""))
            if not urls:
                continue
            row["_tiktok_urls"] = urls
            for u in urls:
                if u not in parent:
                    parent[u] = u
            for i in range(1, len(urls)):
                union(urls[0], urls[i])
            key = _payer_date_key(row)
            payer = key[0]
            if payer:
                u0 = urls[0]
                if key not in anchor_by_payer_date:
                    anchor_by_payer_date[key] = u0
                else:
                    union(anchor_by_payer_date[key], u0)

        for row in rows:
            urls = row.get("_tiktok_urls", [])
            if not urls:
                continue
            rep = find(urls[0])
            component_rows.setdefault(rep, []).append(row)

        for _rep, component_rows_list in component_rows.items():
            seen_urls: Set[str] = set()
            all_urls_in_order: List[str] = []
            username_counts: Counter = Counter()
            first_seen_index: Dict[str, int] = {}
            idx = 0
            for r in component_rows_list:
                for u in r["_tiktok_urls"]:
                    uname = _extract_tiktok_username(u)
                    if uname:
                        username_counts[uname] += 1
                        if uname not in first_seen_index:
                            first_seen_index[uname] = idx
                            idx += 1
                    if u not in seen_urls:
                        seen_urls.add(u)
                        all_urls_in_order.append(u)
            alias_map = _collapse_increment_typos(
                list(username_counts.keys()), username_counts, first_seen_index
            )
            should_collapse = (
                len(component_rows_list) > 1 and len(alias_map) > 0
            )
            if should_collapse:
                final_seen: Set[str] = set()
                final_urls: List[str] = []
                for u in all_urls_in_order:
                    uname = _extract_tiktok_username(u)
                    if uname in alias_map:
                        winner = alias_map[uname]
                        canon = f"https://www.tiktok.com/@{winner}"
                        if canon not in final_seen:
                            final_seen.add(canon)
                            final_urls.append(canon)
                    else:
                        if u not in final_seen:
                            final_seen.add(u)
                            final_urls.append(u)
                tiktok_profile_out = "\n".join(final_urls)
            else:
                tiktok_profile_out = "\n".join(all_urls_in_order)
            sum_edits = 0.0
            for r in component_rows_list:
                v = _parse_number(r.get(key_sum))
                if v is not None:
                    sum_edits += v
            latest_dt = None
            for r in component_rows_list:
                dt = _parse_payment_date(r.get(key_latest))
                if dt is not None and (latest_dt is None or dt > latest_dt):
                    latest_dt = dt
            latest_str = _format_payment_date(latest_dt) if latest_dt else ""
            agg = {
                "Requested by": requested_by,
                "TikTok Profile": tiktok_profile_out,
                key_sum: sum_edits,
                key_latest: latest_str,
            }
            for k in keys_mode:
                if k in agg:
                    continue
                agg[k] = _mode([r.get(k) for r in component_rows_list])
            by_requested.setdefault(requested_by, []).append(agg)

    for cat in by_requested:
        by_requested[cat] = sorted(by_requested[cat], key=lambda r: (r.get("TikTok Profile") or "").lower())

    return by_requested


def _load_manager_sheet_links(
    service, spreadsheet_id: str
) -> Dict[str, Tuple[str, int]]:
    """
    Load all LINK values from all tabs in the manager sheet.
    Returns: {normalized_link: (tab_name, row_number_1based), ...} or {} on failure.
    """
    if not service:
        return {}
    result = {}
    tab_names = [
        SOURCE_SHEET_TAB_NAMES["MAIN"],
        SOURCE_SHEET_TAB_NAMES["BAD"],
        SOURCE_SHEET_TAB_NAMES["GOOD"],
        SOURCE_SHEET_TAB_NAMES["RELIABLE"],
    ]
    link_col = SOURCE_SHEET_COLUMNS["LINK"]
    col_letter = chr(ord("A") + link_col)
    for tab_name in tab_names:
        resp = _sheets_get_values(
            service, spreadsheet_id, f"'{tab_name}'!{col_letter}:{col_letter}"
        )
        if not resp:
            print(f"  WARNING: Could not read tab '{tab_name}' (rate limit or error)")
            continue
        values = resp.get("values", [])
        for i, row in enumerate(values):
            row_num = i + 1
            if row_num < SOURCE_SHEET_DATA_START_ROW:
                continue
            link_cell = (row[0] or "").strip() if row else ""
            primary = pick_primary_tiktok_profile_from_cell(link_cell)
            if not primary:
                continue
            norm = _normalize_tiktok_link(primary)
            if norm:
                result[norm] = (tab_name, row_num)
    return result


def _get_first_empty_row_in_tab(
    service, spreadsheet_id: str, tab_name: str
) -> int:
    """Return 1-based row number of first empty row in the tab (after headers)."""
    if not service:
        return SOURCE_SHEET_DATA_START_ROW
    resp = _sheets_get_values(service, spreadsheet_id, f"'{tab_name}'!A:A")
    if not resp:
        print(f"  WARNING: Could not read '{tab_name}' for empty row, using fallback")
        return SOURCE_SHEET_DATA_START_ROW
    values = resp.get("values", [])
    for i in range(SOURCE_SHEET_DATA_START_ROW - 1, len(values) + 5):
        if i >= len(values):
            return i + 1
        row = values[i]
        if not row or not (row[0] if row else ""):
            return i + 1
    return len(values) + 1


def _read_row(service, spreadsheet_id: str, tab_name: str, row: int) -> List[Any]:
    """Read full row as list of 21 values (columns A-U)."""
    if not service:
        return [""] * 21
    resp = _sheets_get_values(
        service, spreadsheet_id, f"'{tab_name}'!A{row}:U{row}"
    )
    if not resp:
        print(f"  WARNING: Could not read row {row} from '{tab_name}', using empty row")
        return [""] * 21
    values = resp.get("values", [])
    if not values:
        return [""] * 21
    row_vals = values[0]
    while len(row_vals) < 21:
        row_vals.append("")
    return row_vals[:21]


def _build_row_for_reliable(
    payment_row: Dict[str, Any],
    existing_row: Optional[List[Any]] = None,
) -> List[Any]:
    """
    Build a 21-column row for the Reliable tab.
    If existing_row provided (from move), merge; else create new from payment data.
    Type of Compensation:
    - Flat Fee: Price per edit → TIKTOK_PRICE
    - Performance Based: Price per edit → PERF_BASED_PRICE
    - Discord Server: caller must skip (not used here)
    """
    row = [""] * 21
    if existing_row:
        row = list(existing_row)
    username = str(payment_row.get("Name") or "").strip()
    comp_type = str(payment_row.get("Type of Compensation") or "").strip().lower()
    price = payment_row.get("Price per edit", "")

    if not existing_row:
        link_cell = normalize_tiktok_profile_cell_preserve_format(payment_row.get("TikTok Profile"))
        row[SOURCE_SHEET_COLUMNS["LINK"]] = link_cell
    row[SOURCE_SHEET_COLUMNS["USERNAME"]] = username
    row[SOURCE_SHEET_COLUMNS["GENRE"]] = payment_row.get("Genre", "")
    row[SOURCE_SHEET_COLUMNS["TIMES_BOOKED"]] = payment_row.get("# edits / # views", "")
    row[SOURCE_SHEET_COLUMNS["LAST_PAYMENT_DATE"]] = payment_row.get("Payment date", "")
    row[SOURCE_SHEET_COLUMNS["MANUAL_APPROVE"]] = True

    if comp_type == TYPE_COMP_FLAT_FEE:
        row[SOURCE_SHEET_COLUMNS["TIKTOK_PRICE"]] = price
    elif comp_type == TYPE_COMP_PERFORMANCE_BASED:
        row[SOURCE_SHEET_COLUMNS["PERF_BASED_PRICE"]] = price
    else:
        row[SOURCE_SHEET_COLUMNS["TIKTOK_PRICE"]] = price
    return row


def sync_payments_to_manager_sheets(
    by_requested: Dict[str, List[Dict[str, Any]]],
    combined_sheets: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    For each Requested by category that matches a Combined DB Name (Campaign Manager):
    - Open ONLY that manager's sheet from Sheet URL (rows go to matching sheet only)
    - For each payment row: skip if Type of Compensation is "Discord Server"
    - Flat Fee: Price per edit → TIKTOK_PRICE; Performance Based: Price per edit → PERF_BASED_PRICE
    - If exists in Reliable: update with payment data
    - If exists in other tab: move to Reliable, update with payment data
    - If not exists: add new row to Reliable
    Returns stats dict for reporting.
    """
    name_to_url = {
        s["Name"].strip().lower(): s["Sheet URL"].strip()
        for s in combined_sheets
        if s.get("Name")
    }
    service = setup_sheets_service()
    if not service:
        print("ERROR: Could not setup Google Sheets service for sync")
        return {"error": True}

    reliable_tab = SOURCE_SHEET_TAB_NAMES["RELIABLE"]
    run_stats = {
        "categories_matched": 0,
        "categories_skipped_no_sheet": 0,
        "total_updated": 0,
        "total_moved": 0,
        "total_added": 0,
        "total_skipped_discord": 0,
        "total_skipped_invalid": 0,
        "total_api_failures": 0,
        "per_category": {},
    }

    for requested_by, payment_rows in by_requested.items():
        sheet_url = name_to_url.get(requested_by.strip().lower())
        if not sheet_url:
            run_stats["categories_skipped_no_sheet"] += 1
            print(f"\n  [SKIP] '{requested_by}': no matching sheet in Combined DB")
            continue

        sheet_id = extract_spreadsheet_id_from_url(sheet_url)
        if not sheet_id:
            print(f"\n  [SKIP] '{requested_by}': invalid Sheet URL")
            continue

        run_stats["categories_matched"] += 1
        cat_stats = {"updated": 0, "moved": 0, "added": 0, "skipped_discord": 0, "skipped_invalid": 0}
        run_stats["per_category"][requested_by] = cat_stats

        print(f"\n{'─'*60}")
        print(f"  MANAGER: {requested_by}")
        print(f"  Sheet ID: {sheet_id[:30]}...")
        print(f"  Payment rows to process: {len(payment_rows)}")

        link_map = _load_manager_sheet_links(service, sheet_id)
        print(f"  Existing links in manager sheet: {len(link_map)}")
        rows_to_delete: List[Tuple[str, int]] = []  # (tab_name, row_num) to delete after moves; delete bottom-up

        for pay_row in payment_rows:
            comp_type = str(pay_row.get("Type of Compensation") or "").strip().lower()
            if comp_type == TYPE_COMP_DISCORD_SERVER:
                cat_stats["skipped_discord"] += 1
                run_stats["total_skipped_discord"] += 1
                profile = str(pay_row.get("TikTok Profile") or "")[:40]
                print(f"    [SKIP Discord] @{profile}...")
                continue

            profile_cell = pay_row.get("TikTok Profile")
            primary = pick_primary_tiktok_profile_from_cell(profile_cell)
            if primary is None:
                cat_stats["skipped_invalid"] += 1
                run_stats["total_skipped_invalid"] += 1
                print(f"    [SKIP Invalid] Profile: {str(profile_cell)[:50]}")
                continue

            username_norm, _link_key, canonical_url = canonical_tiktok_identity(None, primary)
            if not username_norm and not canonical_url:
                cat_stats["skipped_invalid"] += 1
                run_stats["total_skipped_invalid"] += 1
                print(f"    [SKIP Invalid] Profile: {str(primary)[:50]}")
                continue

            norm = username_norm or _normalize_tiktok_link(canonical_url or primary)
            if not norm:
                cat_stats["skipped_invalid"] += 1
                run_stats["total_skipped_invalid"] += 1
                print(f"    [SKIP Invalid] Profile: {str(primary)[:50]}")
                continue

            info = link_map.get(norm)

            if info:
                tab_name, row_num = info
                if tab_name == reliable_tab:
                    existing = _read_row(service, sheet_id, reliable_tab, row_num)
                    new_row = _build_row_for_reliable(pay_row, existing_row=existing)
                    new_row[SOURCE_SHEET_COLUMNS["COST_100K"]] = COST_100K_FORMULA.format(row=row_num)
                    if _sheets_update_values_safe(
                        service,
                        sheet_id,
                        f"'{reliable_tab}'!A{row_num}:U{row_num}",
                        [new_row],
                    ):
                        cat_stats["updated"] += 1
                        run_stats["total_updated"] += 1
                        print(f"    [UPDATED] @{norm} in Reliable row {row_num}")
                    else:
                        run_stats["total_api_failures"] += 1
                        print(f"    [FAILED] Could not update @{norm} (rate limit or API error)")
                    continue
                first_empty = _get_first_empty_row_in_tab(service, sheet_id, reliable_tab)
                existing = _read_row(service, sheet_id, tab_name, row_num)
                new_row = _build_row_for_reliable(pay_row, existing_row=existing)
                new_row[SOURCE_SHEET_COLUMNS["COST_100K"]] = COST_100K_FORMULA.format(row=first_empty)
                if not _sheets_update_values_safe(
                    service,
                    sheet_id,
                    f"'{reliable_tab}'!A{first_empty}:U{first_empty}",
                    [new_row],
                ):
                    run_stats["total_api_failures"] += 1
                    print(f"    [FAILED] Could not move @{norm} to Reliable")
                    continue
                rows_to_delete.append((tab_name, row_num))  # delete source row later (bottom-up to avoid index shift)
                cat_stats["moved"] += 1
                run_stats["total_moved"] += 1
                print(f"    [MOVED] @{norm} from {tab_name} → Reliable row {first_empty}")
                link_map[norm] = (reliable_tab, first_empty)
            else:
                first_empty = _get_first_empty_row_in_tab(service, sheet_id, reliable_tab)
                new_row = _build_row_for_reliable(pay_row, existing_row=None)
                new_row[SOURCE_SHEET_COLUMNS["COST_100K"]] = COST_100K_FORMULA.format(row=first_empty)
                if not _sheets_update_values_safe(
                    service,
                    sheet_id,
                    f"'{reliable_tab}'!A{first_empty}:U{first_empty}",
                    [new_row],
                ):
                    run_stats["total_api_failures"] += 1
                    print(f"    [FAILED] Could not add @{norm} to Reliable")
                    continue
                cat_stats["added"] += 1
                run_stats["total_added"] += 1
                print(f"    [ADDED] @{norm} to Reliable row {first_empty}")
                link_map[norm] = (reliable_tab, first_empty)

        # Delete source rows that were moved (bottom-up so row indices stay valid)
        for tab_name, row_num in sorted(rows_to_delete, key=lambda x: -x[1]):
            if _sheets_delete_row_safe(service, sheet_id, tab_name, row_num):
                print(f"    [DELETED] Row {row_num} from {tab_name}")
            else:
                run_stats["total_api_failures"] += 1
                print(f"    [WARNING] Could not delete row {row_num} from {tab_name}")

        print(f"  ─── {requested_by} stats: {cat_stats['updated']} updated, {cat_stats['moved']} moved, "
              f"{cat_stats['added']} added, {cat_stats['skipped_discord']} skipped (Discord), "
              f"{cat_stats['skipped_invalid']} skipped (invalid)")

    return run_stats


def aggregate_reliable_accounts_to_combined(
    combined_sheets: List[Dict[str, Any]],
    service,
) -> bool:
    """
    Clear the Reliable tab in the Combined DB and rebuild from each manager's
    Reliable tab. Applies Cost for 100k formula, copies median views, updates header.
    Uses combined_sheets format: [{"Name": ..., "Sheet URL": ...}]
    Returns True on success.
    """
    combined_tab = COMBINED_DB_TAB_NAMES["RELIABLE_COMBINED"]
    reliable_tab = SOURCE_SHEET_TAB_NAMES["RELIABLE"]
    username_col = SOURCE_SHEET_COLUMNS["USERNAME"]
    link_col = SOURCE_SHEET_COLUMNS["LINK"]
    median_col_source = SOURCE_SHEET_COLUMNS["MEDIAN_VIEWS"]

    print("\n" + "=" * 70)
    print("  AGGREGATING RELIABLE ACCOUNTS TO COMBINED SHEET")
    print("=" * 70)

    all_reliable_rows_info = []  # [(sheet_name, sheet_id, source_row_idx, row_data, median_value)]

    for sh in combined_sheets:
        name = (sh.get("Name") or "").strip()
        url = (sh.get("Sheet URL") or "").strip()
        if not name or not url:
            continue
        sheet_id = extract_spreadsheet_id_from_url(url)
        if not sheet_id:
            print(f"  [SKIP] Could not extract spreadsheet ID from {name}")
            continue

        print(f"  Reading Reliable from: {name}")
        result = _sheets_get_values(
            service,
            sheet_id,
            f"'{reliable_tab}'!A3:U",
            value_render_option="FORMATTED_VALUE",
        )
        if not result:
            print(f"    [FAIL] Could not read Reliable tab")
            continue

        rows = result.get("values", [])
        added = 0
        for row_idx, row in enumerate(rows, start=SOURCE_SHEET_DATA_START_ROW):
            if not any(cell for cell in row if cell):
                continue
            username = (row[username_col] or "").strip() if len(row) > username_col else ""
            link = (row[link_col] or "").strip() if len(row) > link_col else ""
            if not username or not link:
                continue
            median_val = row[median_col_source] if len(row) > median_col_source else ""
            all_reliable_rows_info.append((name, sheet_id, row_idx, row, median_val))
            added += 1
        print(f"    Added {added} accounts")

    total = len(all_reliable_rows_info)
    print(f"\n  Total reliable accounts: {total}")

    meta = _sheets_get_metadata(service, COMBINED_DB_SPREADSHEET_ID)
    if not meta:
        print("  [FAIL] Could not get Combined DB metadata")
        return False

    sheets_list = meta.get("sheets", [])
    combined_sheet_id = None
    current_row_count = 1000
    for s in sheets_list:
        if s["properties"]["title"] == combined_tab:
            combined_sheet_id = s["properties"]["sheetId"]
            current_row_count = s["properties"].get("gridProperties", {}).get("rowCount", 1000)
            break

    if combined_sheet_id is None:
        print(f"  [FAIL] Could not find '{combined_tab}' tab")
        return False

    # Clear existing data
    print(f"\n  Clearing existing data in '{combined_tab}'...")
    current_data = _sheets_get_values(
        service,
        COMBINED_DB_SPREADSHEET_ID,
        f"{combined_tab}!A:V",
    )
    current_values = current_data.get("values", []) if current_data else []
    last_data_row = 2
    for i, row in enumerate(current_values[2:], start=3):
        manager_name = (row[0] or "").strip() if len(row) > 0 else ""
        username = (row[1] or "").strip() if len(row) > 1 else ""
        if manager_name or username:
            last_data_row = i

    if last_data_row > 2:
        ok = _sheets_clear_values_safe(
            service,
            COMBINED_DB_SPREADSHEET_ID,
            f"{combined_tab}!A3:V{last_data_row}",
        )
        if ok:
            print(f"    Cleared {last_data_row - 2} rows")
        else:
            print("    [WARN] Clear may have failed")

    if not all_reliable_rows_info:
        print("\n  No reliable accounts to write")
        print("=" * 70)
        return True

    rows_needed = 2 + total
    if current_row_count < rows_needed:
        print(f"  Expanding sheet from {current_row_count} to {rows_needed} rows")
        expand_req = {
            "updateSheetProperties": {
                "properties": {"sheetId": combined_sheet_id, "gridProperties": {"rowCount": rows_needed}},
                "fields": "gridProperties.rowCount",
            }
        }
        if not _sheets_batch_update_safe(service, COMBINED_DB_SPREADSHEET_ID, [expand_req]):
            print("  [WARN] Sheet expand may have failed")

    # Write data: [Campaign Manager Name] + row_data
    combined_rows = []
    for sheet_name, _, _, row_data, _ in all_reliable_rows_info:
        combined_rows.append([sheet_name] + row_data)

    write_range = f"{combined_tab}!A3:V{2 + total}"
    if not _sheets_update_values_safe(service, COMBINED_DB_SPREADSHEET_ID, write_range, combined_rows):
        print("  [FAIL] Could not write aggregated rows")
        return False
    print(f"  Wrote {total} rows")

    # Apply Cost for 100k formula: =(100000*H{row})/K{row}
    formulas = [[f"=(100000*H{3 + i})/K{3 + i}"] for i in range(total)]
    formula_range = f"{combined_tab}!J3:J{2 + total}"
    if _sheets_update_values_safe(service, COMBINED_DB_SPREADSHEET_ID, formula_range, formulas):
        print(f"  Applied Cost for 100k formula to {total} rows")
    else:
        print("  [WARN] Could not apply formula")

    # Copy median views to column K
    median_values = [[info[4]] for info in all_reliable_rows_info]
    median_range = f"{combined_tab}!K3:K{2 + total}"
    if _sheets_update_values_safe(service, COMBINED_DB_SPREADSHEET_ID, median_range, median_values):
        print(f"  Updated median views for {total} rows")
    else:
        print("  [WARN] Could not update median views")

    # Update K2 header with current date
    current_date = datetime.now().strftime("%m/%d")
    new_header = f"15 Videos - Median Views ({current_date})"
    if _sheets_update_values_safe(
        service,
        COMBINED_DB_SPREADSHEET_ID,
        f"{combined_tab}!K{COMBINED_DB_AGGREGATE_HEADER_ROW}",
        [[new_header]],
        value_input_option="RAW",
    ):
        print(f"  Updated header: {new_header}")

    print("=" * 70)
    print("  AGGREGATION COMPLETE")
    print("=" * 70)
    return True


def get_unique_requested_by(data: List[Dict[str, Any]]) -> List[str]:
    """From payment data, return unique 'Requested by' values (after PAID filter not applied here)."""
    seen = set()
    out = []
    for row in data:
        v = (row.get("Requested by") or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def main(
    use_only_combined_db_categories: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Entry point for weekly run.
    1. Load payment data from leftmost tab.
    2. Get unique Requested by categories.
    3. Load Combined DB names from Sheets Inputs (column A).
    4. Filter PAID, group by Requested by and TikTok Profile, aggregate.
    5. Sync payments to manager sheets.

    Returns: (raw_data, by_requested_by_aggregated, unique_requested_by).
    """
    run_start = time.perf_counter()
    start_dt = datetime.now()

    print("=" * 70)
    print("  WEEKLY PAYMENT SYNC — RUN START")
    print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        return _main_impl(use_only_combined_db_categories, run_start, start_dt)
    except Exception as e:
        run_elapsed = time.perf_counter() - run_start
        end_dt = datetime.now()
        print("\n" + "=" * 70)
        print("  RUN FAILED — UNHANDLED ERROR")
        print("=" * 70)
        print(f"  Error: {e}")
        traceback.print_exc()
        print(f"\n  Duration: {run_elapsed:.2f}s | End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        raise


def _main_impl(
    use_only_combined_db_categories: bool,
    run_start: float,
    start_dt: datetime,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[str]]:
    """Internal main implementation (wrapped by main() for error handling)."""
    # Step 1: Load payment data
    print("\n" + "─" * 70)
    print("  STEP 1: Load payment data from leftmost tab")
    print("─" * 70)
    print(f"  Payments sheet ID: {SPREADSHEET_ID}")
    data = load_sheet_data()
    print(f"  → Loaded {len(data)} rows with data")

    # Step 2: Unique Requested by
    print("\n" + "─" * 70)
    print("  STEP 2: Unique 'Requested by' categories")
    print("─" * 70)
    unique_requested_by = get_unique_requested_by(data)
    print(f"  → {len(unique_requested_by)} categories: {unique_requested_by[:10]}{'...' if len(unique_requested_by) > 10 else ''}")

    # Step 3: Load Combined DB (Name, Sheet URL)
    print("\n" + "─" * 70)
    print("  STEP 3: Load Combined DB 'Sheets Inputs' (Name, Sheet URL)")
    print("─" * 70)
    combined_sheets = load_combined_db_sheets()
    combined_names = [s["Name"] for s in combined_sheets if s.get("Name")]
    print(f"  → {len(combined_sheets)} sheets (Name + URL)")

    # Step 4: Process payment data (filter PAID, aggregate)
    print("\n" + "─" * 70)
    print("  STEP 4: Process payment data (PAID only, aggregate by TikTok Profile)")
    print("─" * 70)
    by_requested = process_payment_data(
        data,
        only_categories_in_combined_db=use_only_combined_db_categories,
        combined_db_names=combined_names if use_only_combined_db_categories else None,
    )
    total_aggregated = sum(len(rows) for rows in by_requested.values())
    print(f"  → {len(by_requested)} categories, {total_aggregated} aggregated rows")
    for cat, rows in sorted(by_requested.items(), key=lambda x: (x[0].lower(), 0)):
        print(f"     • {cat}: {len(rows)} profile(s)")

    # Step 5: Sync to manager sheets
    print("\n" + "─" * 70)
    print("  STEP 5: Sync payments to manager sheets")
    print("─" * 70)
    sync_stats = sync_payments_to_manager_sheets(by_requested, combined_sheets)

    # Step 6: Aggregate Reliable to Combined DB
    print("\n" + "─" * 70)
    print("  STEP 6: Aggregate Reliable accounts to Combined DB")
    print("─" * 70)
    svc = setup_sheets_service()
    if svc:
        aggregate_reliable_accounts_to_combined(combined_sheets, svc)
    else:
        print("  ERROR: Could not setup Google Sheets service for aggregation")

    # Run summary
    run_elapsed = time.perf_counter() - run_start
    end_dt = datetime.now()

    print("\n" + "=" * 70)
    print("  RUN SUMMARY — STATISTICS")
    print("=" * 70)
    if sync_stats.get("error"):
        print("  ERROR: Sync failed (no service)")
    else:
        print(f"  Categories matched (processed):     {sync_stats.get('categories_matched', 0)}")
        print(f"  Categories skipped (no sheet):     {sync_stats.get('categories_skipped_no_sheet', 0)}")
        print(f"  Profiles updated (in Reliable):    {sync_stats.get('total_updated', 0)}")
        print(f"  Profiles moved (to Reliable):      {sync_stats.get('total_moved', 0)}")
        print(f"  Profiles added (new to Reliable):  {sync_stats.get('total_added', 0)}")
        print(f"  Profiles skipped (Discord Server): {sync_stats.get('total_skipped_discord', 0)}")
        print(f"  Profiles skipped (invalid):       {sync_stats.get('total_skipped_invalid', 0)}")
        print(f"  API failures (rate limit/error):  {sync_stats.get('total_api_failures', 0)}")
        total_actions = (
            sync_stats.get("total_updated", 0)
            + sync_stats.get("total_moved", 0)
            + sync_stats.get("total_added", 0)
        )
        print(f"  Total sheet actions:               {total_actions}")

    print("\n" + "=" * 70)
    print("  RUN COMPLETE — TIMING")
    print("=" * 70)
    print(f"  Start:      {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End:        {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:   {run_elapsed:.2f} seconds ({run_elapsed / 60:.2f} minutes)")
    print("=" * 70)

    return data, by_requested, unique_requested_by


if __name__ == "__main__":
    raw, aggregated_by_category, unique_categories = main(use_only_combined_db_categories=False)
