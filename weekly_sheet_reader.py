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


# =============================================================================
# CANONICAL TIKTOK NORMALIZATION (from tiktok.py - do NOT reinterpret)
# =============================================================================

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
    if len(segments) != 1:
        return None
    first = segments[0]
    if not first.startswith("@"):
        return None
    username = first[1:]
    if not username:
        return None
    if not re.fullmatch(r"[\w\.-]+", username):
        return None
    return f"https://www.tiktok.com/@{username}"


def _username_from_canonical_profile_url(url: str) -> Optional[str]:
    """Extract normalized username from canonical profile URL."""
    canon = normalize_profile_url(url)
    if not canon:
        return None
    parsed = urlparse(canon)
    segment = (parsed.path or "").strip("/").lstrip("@")
    return normalize_username(segment)


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
    Returns (username_norm, link_key, canonical_profile_url)
    - username_norm: normalized username (lower, no @) using normalize_username,
      OR fallback extracted from canonical_profile_url
    - canonical_profile_url: normalize_profile_url(profile_cell) if valid else None
    - link_key: normalize_link_key(profile_cell) ALWAYS (even if url invalid/empty)
      for stable matching
    """
    profile_str = str(profile_cell or "").strip() if profile_cell is not None else ""
    username_str = str(username_cell or "").strip() if username_cell is not None else ""

    canonical_profile_url = normalize_profile_url(profile_str)
    link_key = normalize_link_key(profile_str)

    username_norm = normalize_username(username_str)
    if not username_norm and canonical_profile_url:
        username_norm = _username_from_canonical_profile_url(canonical_profile_url)
    if not username_norm and profile_str:
        username_norm = _username_from_canonical_profile_url(profile_str)
    if not username_norm and profile_str:
        username_norm = normalize_username(profile_str)

    return (username_norm, link_key, canonical_profile_url)


def _split_trailing_number(u: str) -> Tuple[str, str]:
    """If u ends with digits, return (base_without_digits, digits_str); else return (u, \"\")."""
    if not u:
        return ("", "")
    m = re.match(r"^(.+?)(\d+)$", u)
    if m:
        return (m.group(1), m.group(2))
    return (u, "")


def _canonicalize_incremented_profile(
    name_cell: Any,
    profile_cell: Any,
    seen_usernames: Set[str],
) -> Optional[str]:
    """
    If profile looks like same base as Name but with different trailing digits (human typo),
    and the Name username appears elsewhere in the sheet, return the canonical username (Name).
    Otherwise return None (no change).
    """
    name_u = normalize_username(name_cell) or canonical_tiktok_identity(name_cell, name_cell)[0]
    prof_u = canonical_tiktok_identity(None, profile_cell)[0]
    if not name_u or not prof_u:
        return None
    if prof_u == name_u:
        return None
    name_base, name_digits = _split_trailing_number(name_u)
    prof_base, prof_digits = _split_trailing_number(prof_u)
    if prof_base != name_base or not name_digits or not prof_digits or prof_digits == name_digits:
        return None
    if name_u not in seen_usernames:
        return None
    return name_u


# Regexes for extracting multiple TikTok profiles from a single cell
_TIKTOK_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?tiktok\.com/@[\w._-]+",
    re.IGNORECASE,
)
_AT_USER_PATTERN = re.compile(r"@[\w._-]+")


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
        raw_records.append(record)
        u = normalize_username(record.get("Name", ""))
        if u:
            seen_usernames.add(u)
        for profile in _split_tiktok_profiles(record.get("TikTok Profile", "")):
            u = canonical_tiktok_identity(None, profile)[0]
            if u:
                seen_usernames.add(u)

    # Pass 2: explode by multi-link, then canonicalize incremented trailing-digit typo.
    rows_out = []
    for record in raw_records:
        profiles = _split_tiktok_profiles(record.get("TikTok Profile", ""))
        if not profiles:
            rows_out.append(record)
            continue
        for profile in profiles:
            new_record = dict(record)
            new_record["TikTok Profile"] = profile
            canonical_u = _canonicalize_incremented_profile(
                record.get("Name"), profile, seen_usernames
            )
            if canonical_u is not None:
                new_record["TikTok Profile"] = f"https://www.tiktok.com/@{canonical_u}"
            rows_out.append(new_record)

    return rows_out


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
        paid_rows.append({k: row.get(k, "") for k in keys_keep})

    # Group by (Requested by, TikTok Profile)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in paid_rows:
        requested = (row.get("Requested by") or "").strip()
        profile = (row.get("TikTok Profile") or "").strip()
        key = (requested, profile)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Aggregate each group
    by_requested: Dict[str, List[Dict[str, Any]]] = {}
    for (requested_by, tiktok_profile), rows in groups.items():
        if not requested_by and not tiktok_profile:
            continue
        # Sum # edits / # views
        sum_edits = 0.0
        for r in rows:
            v = _parse_number(r.get(key_sum))
            if v is not None:
                sum_edits += v
        # Latest Payment date (output format D/M/YYYY)
        latest_dt = None
        for r in rows:
            dt = _parse_payment_date(r.get(key_latest))
            if dt is not None and (latest_dt is None or dt > latest_dt):
                latest_dt = dt
        latest_str = _format_payment_date(latest_dt) if latest_dt else ""
        # Mode for text/numeric fields (Name, Type of Compensation, Price per edit, Genre, Payment Status)
        agg = {
            "Requested by": requested_by,
            "TikTok Profile": tiktok_profile,
            key_sum: sum_edits,
            key_latest: latest_str,
        }
        for k in keys_mode:
            if k in agg:
                continue
            agg[k] = _mode([r.get(k) for r in rows])
        by_requested.setdefault(requested_by, []).append(agg)

    # Sort each category's list by TikTok Profile for stable output
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
            link = (row[0] or "").strip() if row else ""
            canon = normalize_profile_url(link)
            if canon:
                result[normalize_link_key(link)] = (tab_name, row_num)
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
    link = str(payment_row.get("TikTok Profile") or "").strip()
    comp_type = str(payment_row.get("Type of Compensation") or "").strip().lower()
    price = payment_row.get("Price per edit", "")

    if not existing_row:
        row[SOURCE_SHEET_COLUMNS["LINK"]] = link
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

            profile = str(pay_row.get("TikTok Profile") or "").strip()
            username_norm, link_key, canonical_url = canonical_tiktok_identity(None, profile)
            if not username_norm and not canonical_url:
                cat_stats["skipped_invalid"] += 1
                run_stats["total_skipped_invalid"] += 1
                print(f"    [SKIP Invalid] Profile: {profile[:50]}")
                continue

            match_key = (
                normalize_link_key(canonical_url)
                if canonical_url
                else normalize_link_key("https://www.tiktok.com/@" + (username_norm or ""))
            )
            info = link_map.get(match_key)

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
                        print(f"    [UPDATED] @{username_norm} in Reliable row {row_num}")
                    else:
                        run_stats["total_api_failures"] += 1
                        print(f"    [FAILED] Could not update @{username_norm} (rate limit or API error)")
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
                    print(f"    [FAILED] Could not move @{username_norm} to Reliable")
                    continue
                rows_to_delete.append((tab_name, row_num))  # delete source row later (bottom-up to avoid index shift)
                cat_stats["moved"] += 1
                run_stats["total_moved"] += 1
                print(f"    [MOVED] @{username_norm} from {tab_name} → Reliable row {first_empty}")
                link_map[match_key] = (reliable_tab, first_empty)
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
                    print(f"    [FAILED] Could not add @{username_norm} to Reliable")
                    continue
                cat_stats["added"] += 1
                run_stats["total_added"] += 1
                print(f"    [ADDED] @{username_norm} to Reliable row {first_empty}")
                link_map[match_key] = (reliable_tab, first_empty)

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
    # --- Self-test: _split_tiktok_profiles (no external frameworks) ---
    def _test_split_tiktok_profiles() -> None:
        multiline = "https://www.tiktok.com/@user1\nhttps://tiktok.com/@user2\nwww.tiktok.com/@user3"
        got = _split_tiktok_profiles(multiline)
        assert len(got) == 3 and "user1" in got[0] and "user2" in got[1] and "user3" in got[2], got
        got2 = _split_tiktok_profiles("@user1 @user2")
        assert got2 == ["@user1", "@user2"], got2
        assert _split_tiktok_profiles("") == []
        assert _split_tiktok_profiles(None) == []
        print("  _split_tiktok_profiles self-test OK")

    _test_split_tiktok_profiles()

    # --- Self-test: canonical normalization (mirror tiktok.py expectations) ---
    def _test_canonical_normalization() -> None:
        assert normalize_profile_url("tiktok.com/@User") == "https://www.tiktok.com/@User"
        assert normalize_username("@User") == "user"
        assert normalize_link_key("https://www.tiktok.com/@User") == "https://www.tiktok.com/@user"
        assert normalize_profile_url("https://vt.tiktok.com/xyz") is None
        un, lk, cu = canonical_tiktok_identity("", "tiktok.com/@abc")
        assert un == "abc", un
        assert cu is not None, cu
        assert lk == cu.lower(), (lk, cu)
        print("  canonical normalization self-test OK")

    _test_canonical_normalization()

    # --- Self-test: _canonicalize_incremented_profile ---
    def _test_canonicalize_incremented() -> None:
        # name=klipolahraga1, profile=@klipolahraga2, seen has klipolahraga1 => rewrite to klipolahraga1
        r = _canonicalize_incremented_profile("klipolahraga1", "@klipolahraga2", {"klipolahraga1"})
        assert r == "klipolahraga1", r
        # name=brand1, profile=@brand2, seen does NOT contain brand1 => no rewrite
        assert _canonicalize_incremented_profile("brand1", "@brand2", set()) is None
        assert _canonicalize_incremented_profile("brand1", "@brand2", {"brand2"}) is None
        # name=abc (no trailing digits), profile=@abc2 => no rewrite
        assert _canonicalize_incremented_profile("abc", "@abc2", {"abc"}) is None
        # name=abc1, profile=@abd2 => no rewrite (different base)
        assert _canonicalize_incremented_profile("abc1", "@abd2", {"abc1"}) is None
        print("  _canonicalize_incremented_profile self-test OK")
    _test_canonicalize_incremented()
    # --- End self-test ---

    raw, aggregated_by_category, unique_categories = main(use_only_combined_db_categories=False)
    # use_only_combined_db_categories=True to restrict to names present in Combined DB "Sheets Inputs"

# --- Diff-style summary: Payments multi-profile support ---
# + _split_tiktok_profiles(cell): regex extract tiktok.com/@... and @user; fallback splitlines/whitespace; strip, dedupe.
# + load_sheet_data(): after building each record, split "TikTok Profile" with _split_tiktok_profiles; if >1 profile,
#   explode into N rows (shallow copy, one profile per row); if 1, set and append once; if 0, append as-is.
# + __main__: self-test for multiline URLs, "@user1 @user2", blank/None. No change to process_payment_data,
#   sync_payments_to_manager_sheets, or column mappings.
#
# --- Diff-style summary: Incremented trailing-digit canonicalization ---
# + canonical_tiktok_identity / normalize_username / normalize_link_key: canonical normalization from tiktok.py.
# + _split_trailing_number(u): (base, digits) if u ends with digits; else (u, "").
# + _canonicalize_incremented_profile(name_cell, profile_cell, seen_usernames): same base + different trailing
#   digits only; name_u must be in seen_usernames (canonical target exists in sheet). Returns canonical username or None.
# + load_sheet_data(): two-pass — Pass 1 build raw_records and seen_usernames from Name + TikTok Profile;
#   Pass 2 explode by multi-link then apply canonicalization during preprocessing (rewrite profile to tiktok.com/@{canonical_u}).
#   Applied after multi-link explode; process/sync logic untouched.
