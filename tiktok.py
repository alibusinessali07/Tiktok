import sys
import subprocess
import re
import time
import csv
import os
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz

# ============================================================================
# ALL TOGGLES AND CONFIGURATION - Configure everything here
# ============================================================================

# ============================================================================
# MODE AND FEATURE TOGGLES
# ============================================================================
# Mode Selection:
# - MANUAL_MODE = True: Use individual toggle values below (you control each toggle manually)
# - MANUAL_MODE = False: Use FULL_RUN toggle to control all features at once
MANUAL_MODE = False  # Set to True to manually control each toggle individually

# Full Run Toggle - Only used when MANUAL_MODE = False
# True = all features enabled, False = only basic features enabled
FULL_RUN = True  # Set to False for limited run (only Initial Reachout tab, skips rows with invalid TikTok Price or empty Median Views)

# Individual Feature Toggles - Used when MANUAL_MODE = True, or as defaults when MANUAL_MODE = False
RUN_PROFILE_SCRAPING = True  # Main profile scraping
RUN_NA_PROFILES = True       # Write NA values for profiles with empty/NA URLs
RUN_UPDATE_HEADERS = True    # Update sheet headers after successful run
RUN_ACCOUNT_CLASSIFICATION = True  # Classify and move accounts to appropriate tabs
SKIP_TIKTOK_LOGIN = True    # True = skip login_to_tiktok() and continue with existing browser profile session
DEBUG_PROFILE_SCRAPE = False  # Debug profile grid/meta counts and first-row meta slots
RUN_BUFFER_WARMUP = False  # True = run 10s warm-up profile; False = skip warm-up
FAST_STARTUP = True  # Skip expensive Playwright browser install when marker exists
RATE_LIMIT_MIN_INTERVAL_SEC = 0.1  # Keep safe request spacing by default
SHEETS_WRITE_FLUSH_EVERY_PROFILES = 10
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# Tab processing toggles - Enable/disable which tabs to process during each run
PROCESS_INITIAL_REACHOUT = True   # Process Initial Reachout tab
PROCESS_BAD_ACCOUNTS = True       # Process Bad Accounts tab
PROCESS_GOOD_ACCOUNTS = True      # Process Good Accounts tab
PROCESS_RELIABLE_ACCOUNTS = True  # Process Reliable Accounts tab

# Apply FULL_RUN logic when MANUAL_MODE is False
if not MANUAL_MODE:
    if FULL_RUN:
        # Full run: All features enabled
        RUN_PROFILE_SCRAPING = True
        RUN_NA_PROFILES = True
        RUN_UPDATE_HEADERS = True
        RUN_ACCOUNT_CLASSIFICATION = True
        PROCESS_INITIAL_REACHOUT = True
        PROCESS_BAD_ACCOUNTS = True
        PROCESS_GOOD_ACCOUNTS = True
        PROCESS_RELIABLE_ACCOUNTS = True
    else:
        # Limited run: Only basic features enabled
        RUN_PROFILE_SCRAPING = True
        RUN_NA_PROFILES = True
        RUN_UPDATE_HEADERS = True
        RUN_ACCOUNT_CLASSIFICATION = True
        PROCESS_INITIAL_REACHOUT = True
        PROCESS_BAD_ACCOUNTS = False
        PROCESS_GOOD_ACCOUNTS = False
        PROCESS_RELIABLE_ACCOUNTS = False
# ============================================================================

# ============================================================================
# SPREADSHEET CONFIGURATION
# ============================================================================
TEST_MODE = False  # Set to True for test mode, False for production

# Spreadsheet IDs
SPREADSHEET_ID = "1L7r3rWl0JLVJVypSPML7Y0cvkulYi5LKBOTP7bER9AI"
COMBINED_DB_SPREADSHEET_ID = "1VIll8-H7_j3BwknGKwiPRFW0LMIeKcIMV6bZrx2UZ14"

if TEST_MODE:
    SPREADSHEET_ID = "1L7r3rWl0JLVJVypSPML7Y0cvkulYi5LKBOTP7bER9AI"
    COMBINED_DB_SPREADSHEET_ID = "1t21VCweRezB0MkgzXimqBKLh7mI6Ri4tEsljFG9dmcA"
PROFILE_DIR = Path(os.getenv("CHROME_EXTENSION_PROFILE", "./chrome_profile_tiktok_sorter")).resolve()
BUFFER_PROFILE_URL = os.getenv("BUFFER_PROFILE_URL", "https://www.tiktok.com/@tiktok")

# Multi-sheet orchestration
PROCESS_MULTIPLE_SHEETS = True  # Set to True to process multiple sheets from a combined database
SHEETS_INPUT_RANGE = "Sheets Inputs!A2:B"  # The tab and range in the combined DB that lists the sheets (Expected columns: A = Name, B = Sheet URL)
# ============================================================================


# ============================================================================
# COLUMN MAPPING - Headers on row 2
# ============================================================================
COLUMNS = {
    'USERNAME': 0,             # Column A - TikTok USERNAME
    'LINK': 1,                 # Column B - Link (TikTok URL)
    'NICHE': 2,                # Column C - Niche
    'TIMES_BOOKED': 3,         # Column D - # of Times Booked
    'LAST_PAYMENT_DATE': 4,    # Column E - Last Payment Date
    'MANUAL_APPROVE': 5,       # Column F - Manual Approve (checkbox)
    'TIKTOK_PRICE': 6,        # Column G - TikTok Price
    'PERF_BASED_PRICE': 7,    # Column H - Perf Based Price ($$ per 100k) (display only, not used in logic)
    'COST_100K': 8,           # Column I - Cost for 100k Views
    'MEDIAN_VIEWS': 9,        # Column J - 15 Videos - Median Views
    'CHANGE_MEDIAN': 10,      # Column K - Change in Median Views
    'UNDER_10K': 11,          # Column L - < 10k
    'BETWEEN_10K_100K': 12,   # Column M - 10k - 100k
    'OVER_100K': 13,          # Column N - > 100k
    'FIFTEENTH_DATE': 14,     # Column O - 15 Videos Ago Date
    'LATEST_DATE': 15,        # Column P - Latest Video Date
    'CONTACT_INFO': 16,       # Column Q - 2nd Contact info
    'GENRE': 17,              # Column R - Genre
    'COUNTRY': 18,            # Column S - Country
    'TYPE': 19,               # Column T - Type (Editor / Faceless / Niche Leader)
    'PAYMENT_INFO': 20        # Column U - Paypal Info / Crypto Wallet
}

# Header row on row 2 (1-indexed), data starts on row 3
HEADER_ROW = 2
DATA_START_ROW = 3

# Tab names
TAB_NAMES = {
    'MAIN': 'Initial Reachout',
    'BAD': 'Bad Accounts',
    'GOOD': 'Good Accounts',
    'RELIABLE': 'Reliable Accounts'
}
# ============================================================================


def ensure_dependencies() -> None:
    """
    Ensure required packages and browser are available. Installs on first run.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright>=1.45.0"], stdout=subprocess.DEVNULL)
    
    try:
        import google.oauth2.service_account  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-api-python-client", "google-auth"], stdout=subprocess.DEVNULL)
    
    try:
        import pytz  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"], stdout=subprocess.DEVNULL)
    
    try:
        import tqdm  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"], stdout=subprocess.DEVNULL)
    
    # Install chromium browser only when needed (fast/idempotent startup)
    try:
        marker_path = PROFILE_DIR / ".playwright_chromium_installed"
        force_install = str(os.getenv("FORCE_PLAYWRIGHT_INSTALL", "")).strip().lower() in ("1", "true", "yes", "on")
        should_install = (not FAST_STARTUP) or force_install or (not marker_path.exists())
        if should_install:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # Marker is written after install attempt (best-effort).
            marker_path.write_text(f"{datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    except Exception:
        pass


def get_cairo_time() -> datetime:
    """Get current time in Cairo timezone."""
    cairo_tz = pytz.timezone('Africa/Cairo')
    return datetime.now(cairo_tz)


def setup_logging() -> logging.Logger:
    """
    Setup logging system that logs to a persistent file with Cairo timezone.
    Each day gets its own log file.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with date (includes day for daily logs)
    cairo_time = get_cairo_time()
    log_filename = f"tiktok_scraper_{cairo_time.strftime('%Y_%m_%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Create custom formatter with Cairo timezone
    class CairoFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            cairo_time = get_cairo_time()
            if datefmt:
                return cairo_time.strftime(datefmt)
            return cairo_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Setup logger
    logger = logging.getLogger('TikTokScraper')
    configured_level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(configured_level)
    
    # Prevent propagation to root logger (prevents console output)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - append mode to keep history
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(configured_level)
    
    # Formatter for clean, readable logs
    formatter = CairoFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
_logger = None
_sheets_service = None
_active_playwright = None
_active_context = None
_active_page = None
_current_sheet_data: Dict[str, Any] = {}
_pending_value_updates: Dict[Any, List[List[Any]]] = {}
_sheet_grid_metadata_cache: Dict[str, Dict[str, Dict[str, int]]] = {}
_captcha_beeped_profiles: Set[str] = set()


def launch_browser_with_profile():
    """Launch Chrome persistent context (PROFILE_DIR), extensions enabled. Return (playwright, context, page)."""
    from playwright.sync_api import sync_playwright

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

    global _active_playwright, _active_context, _active_page
    _active_playwright = playwright
    _active_context = context
    _active_page = page
    _logger.info("Browser launched (profile: %s)", PROFILE_DIR)
    return playwright, context, page


def close_browser_resources(playwright=None, context=None, page=None, reason: str = "manual") -> None:
    """Safely close browser resources and stop Playwright."""
    global _active_playwright, _active_context, _active_page
    p = playwright if playwright is not None else _active_playwright
    c = context if context is not None else _active_context
    pg = page if page is not None else _active_page

    closed_any = False
    try:
        if pg is not None:
            pg.close()
            closed_any = True
    except Exception:
        pass
    try:
        if c is not None:
            c.close()
            closed_any = True
    except Exception:
        pass
    try:
        if p is not None:
            p.stop()
            closed_any = True
    except Exception:
        pass

    _active_playwright = None
    _active_context = None
    _active_page = None
    if closed_any and _logger:
        _logger.info("Browser closed and Playwright stopped. reason=%s", reason)


def ensure_live_page(page):
    """Return a live page, relaunching browser context only when necessary."""
    global _active_playwright, _active_context, _active_page
    try:
        passed_page_closed = (page is None) or bool(page.is_closed())
    except Exception:
        passed_page_closed = True

    if not passed_page_closed:
        if _logger:
            _logger.info("[browser][debug] using passed live page")
        return page

    # If caller page is closed, prefer already-live global page.
    if page is not None and _active_playwright is not None and _active_context is not None and _active_page is not None:
        try:
            if not _active_page.is_closed():
                if _logger:
                    _logger.info("[browser][debug] using existing live global page")
                return _active_page
        except Exception:
            pass

    if page is not None and _logger:
        _logger.info("[browser] Page was closed — relaunching browser context...")
    elif _logger:
        _logger.info("[browser][debug] page is None — launching browser context")

    _, _, live_page = launch_browser_with_profile()
    return live_page


def warmup_buffer_profile(page) -> None:
    """One-time warm-up profile visit to stabilize TikTok + extension injection."""
    try:
        _logger.info("Warm-up: opening buffer profile %s", BUFFER_PROFILE_URL)
        page.goto(BUFFER_PROFILE_URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(10_000)  # exact 10 seconds as requested
        _logger.info("Warm-up complete")
    except Exception as e:
        _logger.warning("Warm-up failed (continuing): %s", e)


def log_and_print(message: str, level: str = "INFO") -> None:
    """
    Print to console and log to file simultaneously.
    
    Args:
        message: The message to print and log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    global _logger
    
    # Print to console respecting configured log threshold
    level_upper = level.upper()
    threshold = getattr(logging, LOG_LEVEL, logging.INFO)
    msg_level = getattr(logging, level_upper, logging.INFO)
    if msg_level >= threshold:
        print(message)
    
    # Log to file if logger is initialized
    if _logger:
        if level_upper == "INFO":
            _logger.info(message)
        elif level_upper == "WARNING":
            _logger.warning(message)
        elif level_upper == "ERROR":
            _logger.error(message)
        elif level_upper == "DEBUG":
            _logger.debug(message)
        else:
            _logger.info(message)


def now_utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def isoformat_utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def within_last_days(ts: int, days: int) -> bool:
    cutoff = now_utc_ts() - days * 86400
    return ts >= cutoff


def extract_username_from_profile_url(url: str) -> Optional[str]:
    # Avoid character class ranges; keep '-' at the end
    m = re.search(r"tiktok\.com/(@[\w._-]+)", url)
    if m:
        return m.group(1).lstrip("@")
    return None


def sanitize_filename_component(text: str) -> str:
    """Return a safe component for filenames (alnum, dash, underscore)."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return safe.strip("_") or "tiktok"


# ============================================================================
# RATE LIMITING AND QUOTA MANAGEMENT
# ============================================================================

class RateLimiter:
    """Manages rate limiting for Google Sheets API requests to avoid quota errors."""
    
    def __init__(self, max_requests_per_minute: int = 50):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute (default 50, below the 60 limit)
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []
        self.lock = False  # Simple lock to prevent concurrent modifications
    
    def _clean_old_requests(self):
        """Remove request times older than 60 seconds."""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
    
    def _wait_if_needed(self):
        """Wait if we're approaching the rate limit."""
        self._clean_old_requests()
        
        if len(self.request_times) >= self.max_requests_per_minute:
            # Calculate wait time until oldest request is 60 seconds old
            oldest_request = min(self.request_times)
            current_time = time.time()
            wait_time = 60 - (current_time - oldest_request) + 1
            
            if wait_time > 0:
                log_and_print(f"    Rate limit approaching, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self._clean_old_requests()
    
    def _record_request(self):
        """Record a successful request."""
        self.request_times.append(time.time())
        # Small delay between requests to avoid hitting limit
        time.sleep(RATE_LIMIT_MIN_INTERVAL_SEC)
    
    def execute_with_retry(self, request_func, max_retries: int = 3, operation_name: str = "API request"):
        """
        Execute an API request with rate limiting and retry logic.
        
        Args:
            request_func: Function that makes the API request (should return the result)
            max_retries: Maximum number of retries for rate limit errors
            operation_name: Name of the operation for logging
        
        Returns:
            Result from request_func, or None if all retries failed
        """
        self._wait_if_needed()
        
        for attempt in range(max_retries):
            try:
                result = request_func()
                self._record_request()
                return result
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RATE_LIMIT_EXCEEDED" in error_str or "Quota exceeded" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait 2^attempt seconds, minimum 2 seconds
                        wait_time = max(2, (2 ** attempt) + 1)
                        log_and_print(f"    Rate limit hit for {operation_name}, waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        # Reset request times after waiting to allow recovery
                        self.request_times = []
                    else:
                        log_and_print(f"    WARNING: Rate limit error for {operation_name} after {max_retries} attempts", "WARNING")
                        return None
                else:
                    # Other error - don't retry, re-raise
                    raise
        
        return None


# Global rate limiters for different operation types
_read_rate_limiter = RateLimiter(max_requests_per_minute=50)
_write_rate_limiter = RateLimiter(max_requests_per_minute=50)
_batch_rate_limiter = RateLimiter(max_requests_per_minute=50)


def sheets_get_values(service, spreadsheet_id: str, range_name: str, value_render_option: str = 'FORMATTED_VALUE'):
    """
    Rate-limited wrapper for spreadsheets().values().get()
    
    Args:
        service: Google Sheets API service object
        spreadsheet_id: The spreadsheet ID
        range_name: The range to read (e.g., "Sheet1!A1:B10")
        value_render_option: How to render values (FORMATTED_VALUE, UNFORMATTED_VALUE, FORMULA)
    
    Returns:
        Result from the API call, or None if rate limited after retries
    """
    def make_request():
        return service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption=value_render_option
        ).execute()
    
    return _read_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"get values from {range_name}"
    )


def sheets_update_values(service, spreadsheet_id: str, range_name: str, values: List[List[Any]], 
                         value_input_option: str = 'USER_ENTERED'):
    """
    Rate-limited wrapper for spreadsheets().values().update()
    
    Args:
        service: Google Sheets API service object
        spreadsheet_id: The spreadsheet ID
        range_name: The range to update (e.g., "Sheet1!A1:B10")
        values: 2D array of values to write
        value_input_option: How to interpret values (USER_ENTERED, RAW)
    
    Returns:
        Result from the API call, or None if rate limited after retries
    """
    def make_request():
        return service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body={'values': values}
        ).execute()
    
    return _write_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"update values in {range_name}"
    )


def sheets_batch_get_values(service, spreadsheet_id: str, ranges: List[str], value_render_option: str = 'FORMATTED_VALUE'):
    """Rate-limited wrapper for spreadsheets().values().batchGet()."""
    def make_request():
        return service.spreadsheets().values().batchGet(
            spreadsheetId=spreadsheet_id,
            ranges=ranges,
            valueRenderOption=value_render_option
        ).execute()

    return _read_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"batch get values ({len(ranges)} ranges)"
    )


def sheets_values_batch_update(service, spreadsheet_id: str, data: List[Dict[str, Any]], value_input_option: str = 'USER_ENTERED'):
    """Rate-limited wrapper for spreadsheets().values().batchUpdate()."""
    def make_request():
        return service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                'valueInputOption': value_input_option,
                'data': data,
            }
        ).execute()

    return _write_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"batch update values ({len(data)} ranges)"
    )


def sheets_clear_values(service, spreadsheet_id: str, range_name: str):
    """
    Rate-limited wrapper for spreadsheets().values().clear()
    
    Args:
        service: Google Sheets API service object
        spreadsheet_id: The spreadsheet ID
        range_name: The range to clear (e.g., "Sheet1!A1:B10")
    
    Returns:
        Result from the API call, or None if rate limited after retries
    """
    def make_request():
        return service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
    
    return _write_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"clear values in {range_name}"
    )


def sheets_get_metadata(service, spreadsheet_id: str):
    """
    Rate-limited wrapper for spreadsheets().get()
    
    Args:
        service: Google Sheets API service object
        spreadsheet_id: The spreadsheet ID
    
    Returns:
        Result from the API call, or None if rate limited after retries
    """
    def make_request():
        return service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    
    return _read_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"get metadata for spreadsheet {spreadsheet_id[:20]}..."
    )


def sheets_batch_update(service, spreadsheet_id: str, requests: List[Dict[str, Any]]):
    """
    Rate-limited wrapper for spreadsheets().batchUpdate()
    
    Args:
        service: Google Sheets API service object
        spreadsheet_id: The spreadsheet ID
        requests: List of request objects for batch operations
    
    Returns:
        Result from the API call, or None if rate limited after retries
    """
    def make_request():
        return service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
    
    return _batch_rate_limiter.execute_with_retry(
        make_request,
        max_retries=3,
        operation_name=f"batch update with {len(requests)} requests"
    )


def setup_google_sheets_service() -> Any:
    """Setup Google Sheets API service using service account credentials."""
    global _sheets_service
    if _sheets_service is not None:
        return _sheets_service
    try:
        # Load credentials from auto_auth.json
        credentials = Credentials.from_service_account_file(
            'auto_auth.json',
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        # Build the service
        _sheets_service = build('sheets', 'v4', credentials=credentials)
        return _sheets_service
    except Exception as e:
        if _logger:
            _logger.info(f"ERROR setting up Google Sheets service: {e}")
        return None


def queue_sheet_value_update(spreadsheet_id: str, range_name: str, values: List[List[Any]], value_input_option: str = 'USER_ENTERED') -> None:
    """Queue a value update; latest write wins per (sheet, option, range)."""
    key = (spreadsheet_id, value_input_option, range_name)
    _pending_value_updates[key] = values


def _col_letters_to_index(col_letters: str) -> Optional[int]:
    """Convert A1 column letters to 1-based index (A=1)."""
    if not col_letters:
        return None
    value = 0
    for ch in col_letters.upper():
        if not ('A' <= ch <= 'Z'):
            return None
        value = (value * 26) + (ord(ch) - ord('A') + 1)
    return value if value > 0 else None


def _parse_a1_endpoint(endpoint: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse endpoint like J458 or J into (col_idx, row_idx)."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)?", (endpoint or "").strip())
    if not m:
        return None, None
    col_idx = _col_letters_to_index(m.group(1))
    row_idx = int(m.group(2)) if m.group(2) else None
    return col_idx, row_idx


def _parse_a1_range_requirements(range_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse A1 range and return sheet/max row/max col requirements.
    Supports:
      - 'Bad Accounts'!J458:P458
      - Bad Accounts!J458:P458
      - 'Reliable Accounts'!J:P (row unknown)
    """
    if "!" not in (range_name or ""):
        return None
    sheet_part, a1_part = range_name.split("!", 1)
    sheet_name = sheet_part.strip()
    if sheet_name.startswith("'") and sheet_name.endswith("'") and len(sheet_name) >= 2:
        sheet_name = sheet_name[1:-1].replace("''", "'")
    a1_part = a1_part.strip()
    if not a1_part:
        return None
    if ":" in a1_part:
        start_ref, end_ref = a1_part.split(":", 1)
    else:
        start_ref, end_ref = a1_part, a1_part
    start_col, start_row = _parse_a1_endpoint(start_ref)
    end_col, end_row = _parse_a1_endpoint(end_ref)
    required_col = max([c for c in (start_col, end_col) if c is not None], default=None)
    if start_row is None or end_row is None:
        required_row = None
    else:
        required_row = max(start_row, end_row)
    return {
        "sheet_name": sheet_name,
        "required_row": required_row,
        "required_col": required_col,
    }


def _get_sheet_grid_metadata(service, spreadsheet_id: str, force_refresh: bool = False) -> Dict[str, Dict[str, int]]:
    """Return cached sheet metadata keyed by sheet title."""
    if not force_refresh and spreadsheet_id in _sheet_grid_metadata_cache:
        return _sheet_grid_metadata_cache[spreadsheet_id]
    meta = sheets_get_metadata(service, spreadsheet_id) or {}
    sheet_map: Dict[str, Dict[str, int]] = {}
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {}) or {}
        title = str(props.get("title") or "")
        grid = props.get("gridProperties", {}) or {}
        if not title:
            continue
        sheet_map[title] = {
            "sheetId": int(props.get("sheetId", -1)),
            "rowCount": int(grid.get("rowCount", 0)),
            "columnCount": int(grid.get("columnCount", 0)),
        }
    _sheet_grid_metadata_cache[spreadsheet_id] = sheet_map
    return sheet_map


def ensure_sheet_grid_capacity(spreadsheet_id: str, sheet_name: str, required_rows: Optional[int], required_cols: Optional[int]) -> bool:
    """Ensure sheet grid can fit required rows/cols by expanding if needed."""
    service = setup_google_sheets_service()
    if not service:
        return False
    metadata = _get_sheet_grid_metadata(service, spreadsheet_id)
    sheet_info = metadata.get(sheet_name)
    if not sheet_info:
        if _logger:
            _logger.error("Sheets grid check: sheet '%s' not found in spreadsheet %s", sheet_name, spreadsheet_id)
        return False
    cur_rows = int(sheet_info.get("rowCount", 0))
    cur_cols = int(sheet_info.get("columnCount", 0))
    new_rows = cur_rows
    new_cols = cur_cols
    if required_rows is not None and required_rows > cur_rows:
        new_rows = required_rows + 50
    if required_cols is not None and required_cols > cur_cols:
        new_cols = required_cols + 5
    if new_rows == cur_rows and new_cols == cur_cols:
        return True
    req: Dict[str, Any] = {
        "updateSheetProperties": {
            "properties": {
                "sheetId": int(sheet_info.get("sheetId", -1)),
                "gridProperties": {},
            },
            "fields": "",
        }
    }
    fields: List[str] = []
    if new_rows != cur_rows:
        req["updateSheetProperties"]["properties"]["gridProperties"]["rowCount"] = int(new_rows)
        fields.append("gridProperties.rowCount")
    if new_cols != cur_cols:
        req["updateSheetProperties"]["properties"]["gridProperties"]["columnCount"] = int(new_cols)
        fields.append("gridProperties.columnCount")
    req["updateSheetProperties"]["fields"] = ",".join(fields)
    result = sheets_batch_update(service, spreadsheet_id, [req])
    if result is None:
        return False
    _get_sheet_grid_metadata(service, spreadsheet_id, force_refresh=True)
    return True


def _range_fits_grid(range_name: str, sheet_map: Dict[str, Dict[str, int]]) -> bool:
    """Best-effort local validation against sheet row/col limits."""
    parsed = _parse_a1_range_requirements(range_name)
    if not parsed:
        return True
    info = sheet_map.get(str(parsed.get("sheet_name") or ""))
    if not info:
        return False
    required_row = parsed.get("required_row")
    required_col = parsed.get("required_col")
    if required_row is not None and int(required_row) > int(info.get("rowCount", 0)):
        return False
    if required_col is not None and int(required_col) > int(info.get("columnCount", 0)):
        return False
    return True


def flush_pending_sheet_writes(reason: str = "manual") -> bool:
    """Flush queued value updates via values.batchUpdate (best effort)."""
    if not _pending_value_updates:
        return True
    queued_count = len(_pending_value_updates)
    service = setup_google_sheets_service()
    if not service:
        return False

    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for (spreadsheet_id, value_input_option, range_name), values in list(_pending_value_updates.items()):
        gkey = (spreadsheet_id, value_input_option)
        grouped.setdefault(gkey, []).append({"range": range_name, "values": values})

    all_ok = True
    for (spreadsheet_id, value_input_option), data in grouped.items():
        # Keep requests comfortably sized.
        batch_size = 100
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            try:
                # Preflight range requirements by sheet and auto-expand grid when needed.
                required_by_sheet: Dict[str, Dict[str, Optional[int]]] = {}
                for entry in chunk:
                    parsed = _parse_a1_range_requirements(str(entry.get("range") or ""))
                    if not parsed:
                        continue
                    sheet_name = str(parsed.get("sheet_name") or "")
                    if not sheet_name:
                        continue
                    req = required_by_sheet.setdefault(sheet_name, {"row": None, "col": None})
                    req_row = parsed.get("required_row")
                    req_col = parsed.get("required_col")
                    if req_row is not None:
                        req["row"] = max(int(req["row"] or 0), int(req_row))
                    if req_col is not None:
                        req["col"] = max(int(req["col"] or 0), int(req_col))

                for sheet_name, req in required_by_sheet.items():
                    try:
                        ensured = ensure_sheet_grid_capacity(
                            spreadsheet_id,
                            sheet_name,
                            int(req["row"]) if req.get("row") is not None else None,
                            int(req["col"]) if req.get("col") is not None else None,
                        )
                        if not ensured:
                            all_ok = False
                    except Exception as cap_e:
                        all_ok = False
                        if _logger:
                            _logger.error(
                                "Failed ensuring grid capacity for '%s' in %s: %s",
                                sheet_name,
                                spreadsheet_id,
                                cap_e,
                            )

                result = sheets_values_batch_update(
                    service,
                    spreadsheet_id,
                    chunk,
                    value_input_option=value_input_option,
                )
                if result is None:
                    all_ok = False
            except HttpError as http_e:
                err_text = str(http_e)
                if ("400" in err_text) and ("exceeds grid limits" in err_text.lower()):
                    all_ok = False
                    current_map = _get_sheet_grid_metadata(service, spreadsheet_id, force_refresh=True)
                    valid_entries: List[Dict[str, Any]] = []
                    invalid_ranges: List[str] = []
                    for entry in chunk:
                        r = str(entry.get("range") or "")
                        if _range_fits_grid(r, current_map):
                            valid_entries.append(entry)
                        else:
                            invalid_ranges.append(r)
                    if invalid_ranges and _logger:
                        _logger.error(
                            "Dropping out-of-grid updates after 400 exceeds-grid error: %s",
                            invalid_ranges,
                        )
                    if valid_entries:
                        try:
                            retry_result = sheets_values_batch_update(
                                service,
                                spreadsheet_id,
                                valid_entries,
                                value_input_option=value_input_option,
                            )
                            if retry_result is None:
                                all_ok = False
                        except Exception as retry_e:
                            all_ok = False
                            if _logger:
                                _logger.error("Retry after dropping invalid ranges failed: %s", retry_e)
                else:
                    all_ok = False
                    if _logger:
                        _logger.error("Sheets batch update HttpError (continuing): %s", http_e)
            except Exception as e:
                all_ok = False
                if _logger:
                    _logger.error("Sheets batch update error (continuing): %s", e)

    # Clear queue even on partial failure to avoid infinite retries on bad ranges.
    _pending_value_updates.clear()
    if _logger:
        _logger.info(
            "Flushed Sheets queue (%s): queued=%s remaining=%s",
            reason,
            queued_count,
            len(_pending_value_updates),
        )
    return all_ok


def normalize_profile_link(link: str) -> str:
    """Normalize LINK cell for stable keying without changing navigation URL semantics."""
    if not link:
        return ""
    return str(link).strip().strip("\"'").strip()


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


def canonical_username_from_row(row: list) -> Optional[str]:
    """Canonical username from row using col A first, then col B link fallback."""
    if row is None:
        return None
    username_cell = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] else ""
    username = normalize_username(username_cell)
    if username:
        return username
    link_cell = row[COLUMNS['LINK']] if len(row) > COLUMNS['LINK'] else ""
    canonical_url = normalize_profile_url(link_cell)
    if not canonical_url:
        return None
    return _username_from_canonical_profile_url(canonical_url)


def is_valid_profile_url(url: str) -> bool:
    """
    Returns True only for real TikTok profile URLs.
    Rejects video links, vt short links, mobile video links, etc.
    """
    return normalize_profile_url(url) is not None


def normalize_link_key(link: str) -> str:
    """Case-insensitive key for link-based row mapping."""
    canonical = normalize_profile_url(link)
    if canonical:
        return canonical.lower()
    return normalize_profile_link(link).lower()


def _resolve_profile_location(username_normalized: str, profile_url: str = "") -> Any:
    """Resolve row using canonical (username, link) pair mapping."""
    pair_map = _current_sheet_data.get("username_link_to_row", {}) if _current_sheet_data else {}
    uname = (username_normalized or "").lower().lstrip("@")
    lkey = normalize_link_key(profile_url)
    return pair_map.get((uname, lkey))


def extract_spreadsheet_id_from_url(url: str) -> Optional[str]:
    """
    Extract the Google Sheets ID from a URL like:
    https://docs.google.com/spreadsheets/d/<ID>/edit#...
    """
    if not url:
        return None
    
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return None
    return match.group(1)


def get_source_median_values(service, sheet_id: str, tab_name: str, row_numbers: List[int]) -> Dict[int, Any]:
    """Fetch source median values (column J) for many rows in one read."""
    if not row_numbers:
        return {}
    min_row = min(row_numbers)
    max_row = max(row_numbers)
    result = sheets_get_values(
        service,
        sheet_id,
        f"{tab_name}!J{min_row}:J{max_row}",
        value_render_option='FORMATTED_VALUE',
    ) or {}
    rows = result.get("values", [])
    out: Dict[int, Any] = {}
    for rn in row_numbers:
        idx = rn - min_row
        if 0 <= idx < len(rows) and rows[idx]:
            out[rn] = rows[idx][0]
        else:
            out[rn] = ""
    return out


def aggregate_reliable_accounts_to_combined(sheets_processed: List[Dict[str, str]]) -> bool:
    """
    Aggregate all Reliable Accounts from processed sheets into the 
    'Reliable Accounts Combined' tab in the combined database.
    Adds Campaign Manager Name as the first column.
    Skips rows with empty TikTok USERNAME or Link columns.
    """
    service = setup_google_sheets_service()
    if not service:
        log_and_print("ERROR: Could not setup Google Sheets service for aggregation", "ERROR")
        return False
    
    try:
        log_and_print("\n" + "=" * 80)
        log_and_print("AGGREGATING RELIABLE ACCOUNTS TO COMBINED SHEET")
        log_and_print("=" * 80)
        
        combined_tab_name = "Reliable Accounts Combined"
        
        # First, get the header from the combined sheet
        log_and_print(f"Reading header from '{combined_tab_name}' tab...")
        header_result = sheets_get_values(
            service,
            COMBINED_DB_SPREADSHEET_ID,
            f"{combined_tab_name}!A1:Z2"
        )
        
        header_rows = header_result.get('values', [])
        if not header_rows or len(header_rows) < 2:
            log_and_print(f"ERROR: Could not read header from '{combined_tab_name}' tab", "ERROR")
            return False
        
        # Prepare aggregated data - now tracking source rows for copyPaste
        all_reliable_rows_info = []  # List of tuples: (sheet_name, source_sheet_id, source_row_index)
        total_accounts = 0
        
        # For each processed sheet, read its Reliable Accounts tab
        for sheet_info in sheets_processed:
            sheet_name = sheet_info['name']
            sheet_id = sheet_info['spreadsheet_id']
            
            log_and_print(f"\nReading Reliable Accounts from: {sheet_name}")
            
            try:
                # Get sheet metadata to get sheet IDs
                source_sheet_metadata = sheets_get_metadata(service, sheet_id)
                source_sheets = source_sheet_metadata.get('sheets', [])
                source_reliable_sheet_id = None
                for sheet in source_sheets:
                    if sheet['properties']['title'] == TAB_NAMES['RELIABLE']:
                        source_reliable_sheet_id = sheet['properties']['sheetId']
                        break
                
                if source_reliable_sheet_id is None:
                    log_and_print(f"  WARNING: Could not find '{TAB_NAMES['RELIABLE']}' tab in {sheet_name}", "WARNING")
                    continue
                
                # Read all data from Reliable Accounts tab as VALUES (skip header rows 1-2, get data from row 3+)
                # We'll copy formulas from the combined sheet template row instead
                reliable_range = f"{TAB_NAMES['RELIABLE']}!A3:W"
                result = sheets_get_values(
                    service,
                    sheet_id,
                    reliable_range,
                    value_render_option='FORMATTED_VALUE'
                )
                
                rows = result.get('values', [])
                if not rows:
                    log_and_print(f"  No reliable accounts found in {sheet_name}")
                    continue
                
                # Track rows to copy (now with actual data/formulas)
                added_count = 0
                for row_idx, row in enumerate(rows, start=3):  # Start from row 3
                    # Skip completely empty rows
                    if not any(cell for cell in row if cell):
                        continue
                    
                    # Skip rows where TikTok USERNAME (Column A, index 0) or Link (Column B, index 1) is empty
                    username = row[0].strip() if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                    link = row[1].strip() if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                    
                    if not username or not link:
                        continue
                    
                    # Store the row data with formulas and sheet info
                    all_reliable_rows_info.append((sheet_name, sheet_id, source_reliable_sheet_id, row_idx, row))
                    added_count += 1
                
                log_and_print(f"  Added {added_count} accounts from {sheet_name}")
                total_accounts += added_count
                
            except Exception as e:
                log_and_print(f"  WARNING: Could not read Reliable Accounts from {sheet_name}: {e}", "WARNING")
                continue
        
        log_and_print(f"\nTotal reliable accounts collected: {total_accounts}")
        
        # Clear existing data in combined sheet (keep header)
        log_and_print(f"\nClearing existing data in '{combined_tab_name}' tab...")
        
        # First, get the current dimensions to know how much to clear
        sheet_metadata = sheets_get_metadata(service, COMBINED_DB_SPREADSHEET_ID)
        sheets = sheet_metadata.get('sheets', [])
        combined_sheet_id = None
        current_row_count = 1000  # Default
        for sheet in sheets:
            if sheet['properties']['title'] == combined_tab_name:
                combined_sheet_id = sheet['properties']['sheetId']
                current_row_count = sheet['properties'].get('gridProperties', {}).get('rowCount', 1000)
                break
        
        if combined_sheet_id is not None:
            # Get current data to see how many rows have content
            current_data = sheets_get_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                f"{combined_tab_name}!A:V"
            )
            current_values = current_data.get('values', [])
            
            # Find last row with actual data (where Campaign Manager Name and TikTok USERNAME are not empty)
            last_data_row = 2  # Start from header
            for i, row in enumerate(current_values[2:], start=3):  # Skip rows 1-2 (headers)
                # Check if Campaign Manager Name (Column A) and TikTok USERNAME (Column B after shift) are both empty
                manager_name = row[0].strip() if len(row) > 0 and row[0] else ""
                username = row[1].strip() if len(row) > 1 and row[1] else ""
                
                if manager_name or username:
                    last_data_row = i
            
            if last_data_row > 2:
                # Clear content of rows 3 onwards
                clear_range = f"{combined_tab_name}!A3:V{last_data_row}"
                sheets_clear_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    clear_range
                )
                log_and_print(f"  Cleared {last_data_row - 2} existing data rows")
        
        # Write aggregated data
        if all_reliable_rows_info:
            log_and_print(f"\nWriting {len(all_reliable_rows_info)} rows to '{combined_tab_name}' tab...")
            
            # Check if we need to expand the sheet
            rows_needed = 2 + len(all_reliable_rows_info)  # Header (2 rows) + data rows
            if current_row_count < rows_needed:
                log_and_print(f"  WARNING: Need to expand sheet from {current_row_count} to {rows_needed} rows")
                expand_request = {
                    'updateSheetProperties': {
                        'properties': {
                            'sheetId': combined_sheet_id,
                            'gridProperties': {
                                'rowCount': rows_needed
                            }
                        },
                        'fields': 'gridProperties.rowCount'
                    }
                }
                sheets_batch_update(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    [expand_request]
                )
            else:
                log_and_print(f"  Sheet has {current_row_count} rows, will use existing empty rows (need {rows_needed} rows)")
            
            # Prepare data with Campaign Manager Name prepended
            combined_rows = []
            for sheet_name, _, _, _, row_data in all_reliable_rows_info:
                # Prepend Campaign Manager Name to the row
                combined_row = [sheet_name] + row_data
                combined_rows.append(combined_row)
            
            # Write all data at once
            write_range = f"{combined_tab_name}!A3:V{2 + len(combined_rows)}"
            sheets_update_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                write_range,
                combined_rows,
                value_input_option='USER_ENTERED'
            )
            
            log_and_print(f"  Successfully wrote {len(combined_rows)} rows")
            
            # Apply Cost for 100k Views formula to combined sheet
            # In combined sheet: Campaign Manager Name is column A, so columns are shifted by 1
            # TikTok Price is column H, Median Views is column K, Cost for 100k Views is column J
            log_and_print(f"\n  Applying Cost for 100k Views formula to combined sheet...")
            formulas = []
            for i in range(len(combined_rows)):
                row_num = 3 + i  # Starting from row 3
                # Formula: =(100000*H{row})/K{row}
                formula = f"=(100000*H{row_num})/K{row_num}"
                formulas.append([formula])
            
            if formulas:
                formula_range = f"{combined_tab_name}!J3:J{2 + len(formulas)}"
                sheets_update_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    formula_range,
                    formulas,
                    value_input_option='USER_ENTERED'
                )
                log_and_print(f"  Applied formula to {len(formulas)} rows")
            
            # Update 15 Videos - Median Views column (Column K in combined sheet, Column J in source sheets)
            log_and_print(f"\n  Updating 15 Videos - Median Views column...")
            median_views_values = []
            # Group row lookups by source sheet to avoid one API call per row.
            rows_by_sheet: Dict[str, List[int]] = {}
            for _, sheet_id, _, source_row_idx, _ in all_reliable_rows_info:
                rows_by_sheet.setdefault(sheet_id, []).append(source_row_idx)
            median_cache_by_sheet: Dict[str, Dict[int, Any]] = {}
            for sheet_id, row_numbers in rows_by_sheet.items():
                median_cache_by_sheet[sheet_id] = get_source_median_values(service, sheet_id, TAB_NAMES['RELIABLE'], row_numbers)

            for sheet_name, sheet_id, _, source_row_idx, _ in all_reliable_rows_info:
                try:
                    median_value = median_cache_by_sheet.get(sheet_id, {}).get(source_row_idx, "")
                    median_views_values.append([median_value])
                except Exception as e:
                    log_and_print(f"    WARNING: Could not read median views from {sheet_name} row {source_row_idx}: {e}", "WARNING")
                    median_views_values.append([""])
            
            if median_views_values:
                median_range = f"{combined_tab_name}!K3:K{2 + len(median_views_values)}"
                sheets_update_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    median_range,
                    median_views_values,
                    value_input_option='USER_ENTERED'
                )
                log_and_print(f"  Updated median views for {len(median_views_values)} rows")
            
            # Update the header for the median views column (Column K, row 2) with current date
            current_date = datetime.now().strftime('%m/%d')
            new_header = f"15 Videos - Median Views ({current_date})"
            header_range = f"{combined_tab_name}!K{HEADER_ROW}"
            sheets_update_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                header_range,
                [[new_header]],
                value_input_option='RAW'
            )
            log_and_print(f"  Updated header: {new_header}")
        else:
            log_and_print("\n  No reliable accounts to write")
        
        log_and_print("=" * 80)
        log_and_print("AGGREGATION COMPLETE")
        log_and_print("=" * 80)
        return True
        
    except Exception as e:
        log_and_print(f"ERROR during aggregation: {e}", "ERROR")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False


def aggregate_good_accounts_to_combined(sheets_processed: List[Dict[str, str]]) -> bool:
    """
    Aggregate all Good Accounts from processed sheets into the 
    'Good Accounts Combined' tab in the combined database.
    Adds Campaign Manager Name as the first column.
    Skips rows with empty TikTok USERNAME or Link columns.
    """
    service = setup_google_sheets_service()
    if not service:
        log_and_print("ERROR: Could not setup Google Sheets service for aggregation", "ERROR")
        return False
    
    try:
        log_and_print("\n" + "=" * 80)
        log_and_print("AGGREGATING GOOD ACCOUNTS TO COMBINED SHEET")
        log_and_print("=" * 80)
        
        combined_tab_name = "Good Accounts Combined"
        
        # First, get the header from the combined sheet
        log_and_print(f"Reading header from '{combined_tab_name}' tab...")
        header_result = sheets_get_values(
            service,
            COMBINED_DB_SPREADSHEET_ID,
            f"{combined_tab_name}!A1:Z2"
        )
        
        header_rows = header_result.get('values', [])
        if not header_rows or len(header_rows) < 2:
            log_and_print(f"ERROR: Could not read header from '{combined_tab_name}' tab", "ERROR")
            return False
        
        # Prepare aggregated data - now tracking source rows for copyPaste
        all_good_rows_info = []  # List of tuples: (sheet_name, source_sheet_id, source_row_index)
        total_accounts = 0
        
        # For each processed sheet, read its Good Accounts tab
        for sheet_info in sheets_processed:
            sheet_name = sheet_info['name']
            sheet_id = sheet_info['spreadsheet_id']
            
            log_and_print(f"\nReading Good Accounts from: {sheet_name}")
            
            try:
                # Get sheet metadata to get sheet IDs
                source_sheet_metadata = sheets_get_metadata(service, sheet_id)
                source_sheets = source_sheet_metadata.get('sheets', [])
                source_good_sheet_id = None
                for sheet in source_sheets:
                    if sheet['properties']['title'] == TAB_NAMES['GOOD']:
                        source_good_sheet_id = sheet['properties']['sheetId']
                        break
                
                if source_good_sheet_id is None:
                    log_and_print(f"  WARNING: Could not find '{TAB_NAMES['GOOD']}' tab in {sheet_name}", "WARNING")
                    continue
                
                # Read all data from Good Accounts tab as VALUES (skip header rows 1-2, get data from row 3+)
                # We'll copy formulas from the combined sheet template row instead
                good_range = f"{TAB_NAMES['GOOD']}!A3:W"
                result = sheets_get_values(
                    service,
                    sheet_id,
                    good_range,
                    value_render_option='FORMATTED_VALUE'
                )
                
                if not result:
                    log_and_print(f"  WARNING: Could not read Good Accounts from {sheet_name} (rate limited)", "WARNING")
                    continue
                
                rows = result.get('values', [])
                if not rows:
                    log_and_print(f"  No good accounts found in {sheet_name}")
                    continue
                
                # Track rows to copy (now with actual data/formulas)
                added_count = 0
                for row_idx, row in enumerate(rows, start=3):  # Start from row 3
                    # Skip completely empty rows
                    if not any(cell for cell in row if cell):
                        continue
                    
                    # Skip rows where TikTok USERNAME (Column A, index 0) or Link (Column B, index 1) is empty
                    username = row[0].strip() if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                    link = row[1].strip() if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                    
                    if not username or not link:
                        continue
                    
                    # Store the row data with formulas and sheet info
                    all_good_rows_info.append((sheet_name, sheet_id, source_good_sheet_id, row_idx, row))
                    added_count += 1
                
                log_and_print(f"  Added {added_count} accounts from {sheet_name}")
                total_accounts += added_count
                
            except Exception as e:
                log_and_print(f"  WARNING: Could not read Good Accounts from {sheet_name}: {e}", "WARNING")
                continue
        
        log_and_print(f"\nTotal good accounts collected: {total_accounts}")
        
        # Clear existing data in combined sheet (keep header)
        log_and_print(f"\nClearing existing data in '{combined_tab_name}' tab...")
        
        # First, get the current dimensions to know how much to clear
        sheet_metadata = sheets_get_metadata(service, COMBINED_DB_SPREADSHEET_ID)
        sheets = sheet_metadata.get('sheets', [])
        combined_sheet_id = None
        current_row_count = 1000  # Default
        for sheet in sheets:
            if sheet['properties']['title'] == combined_tab_name:
                combined_sheet_id = sheet['properties']['sheetId']
                current_row_count = sheet['properties'].get('gridProperties', {}).get('rowCount', 1000)
                break
        
        if combined_sheet_id is not None:
            # Get current data to see how many rows have content
            current_data = sheets_get_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                f"{combined_tab_name}!A:V"
            )
            current_values = current_data.get('values', [])
            
            # Find last row with actual data (where Campaign Manager Name and TikTok USERNAME are not empty)
            last_data_row = 2  # Start from header
            for i, row in enumerate(current_values[2:], start=3):  # Skip rows 1-2 (headers)
                # Check if Campaign Manager Name (Column A) and TikTok USERNAME (Column B after shift) are both empty
                manager_name = row[0].strip() if len(row) > 0 and row[0] else ""
                username = row[1].strip() if len(row) > 1 and row[1] else ""
                
                if manager_name or username:
                    last_data_row = i
            
            if last_data_row > 2:
                # Clear content of rows 3 onwards
                clear_range = f"{combined_tab_name}!A3:V{last_data_row}"
                sheets_clear_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    clear_range
                )
                log_and_print(f"  Cleared {last_data_row - 2} existing data rows")
        
        # Write aggregated data
        if all_good_rows_info:
            log_and_print(f"\nWriting {len(all_good_rows_info)} rows to '{combined_tab_name}' tab...")
            
            # Check if we need to expand the sheet
            rows_needed = 2 + len(all_good_rows_info)  # Header (2 rows) + data rows
            if current_row_count < rows_needed:
                log_and_print(f"  WARNING: Need to expand sheet from {current_row_count} to {rows_needed} rows")
                expand_request = {
                    'updateSheetProperties': {
                        'properties': {
                            'sheetId': combined_sheet_id,
                            'gridProperties': {
                                'rowCount': rows_needed
                            }
                        },
                        'fields': 'gridProperties.rowCount'
                    }
                }
                sheets_batch_update(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    [expand_request]
                )
            else:
                log_and_print(f"  Sheet has {current_row_count} rows, will use existing empty rows (need {rows_needed} rows)")
            
            # Prepare data with Campaign Manager Name prepended
            combined_rows = []
            for sheet_name, _, _, _, row_data in all_good_rows_info:
                # Prepend Campaign Manager Name to the row
                combined_row = [sheet_name] + row_data
                combined_rows.append(combined_row)
            
            # Write all data at once
            write_range = f"{combined_tab_name}!A3:V{2 + len(combined_rows)}"
            sheets_update_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                write_range,
                combined_rows,
                value_input_option='USER_ENTERED'
            )
            
            log_and_print(f"  Successfully wrote {len(combined_rows)} rows")
            
            # Apply Cost for 100k Views formula to combined sheet
            # In combined sheet: Campaign Manager Name is column A, so columns are shifted by 1
            # TikTok Price is column H, Median Views is column K, Cost for 100k Views is column J
            log_and_print(f"\n  Applying Cost for 100k Views formula to combined sheet...")
            formulas = []
            for i in range(len(combined_rows)):
                row_num = 3 + i  # Starting from row 3
                # Formula: =(100000*H{row})/K{row}
                formula = f"=(100000*H{row_num})/K{row_num}"
                formulas.append([formula])
            
            if formulas:
                formula_range = f"{combined_tab_name}!J3:J{2 + len(formulas)}"
                sheets_update_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    formula_range,
                    formulas,
                    value_input_option='USER_ENTERED'
                )
                log_and_print(f"  Applied formula to {len(formulas)} rows")
            
            # Update 15 Videos - Median Views column (Column K in combined sheet, Column J in source sheets)
            log_and_print(f"\n  Updating 15 Videos - Median Views column...")
            median_views_values = []
            # Group row lookups by source sheet to avoid one API call per row.
            rows_by_sheet: Dict[str, List[int]] = {}
            for _, sheet_id, _, source_row_idx, _ in all_good_rows_info:
                rows_by_sheet.setdefault(sheet_id, []).append(source_row_idx)
            median_cache_by_sheet: Dict[str, Dict[int, Any]] = {}
            for sheet_id, row_numbers in rows_by_sheet.items():
                median_cache_by_sheet[sheet_id] = get_source_median_values(service, sheet_id, TAB_NAMES['GOOD'], row_numbers)

            for sheet_name, sheet_id, _, source_row_idx, _ in all_good_rows_info:
                try:
                    median_value = median_cache_by_sheet.get(sheet_id, {}).get(source_row_idx, "")
                    median_views_values.append([median_value])
                except Exception as e:
                    log_and_print(f"    WARNING: Could not read median views from {sheet_name} row {source_row_idx}: {e}", "WARNING")
                    median_views_values.append([""])
            
            if median_views_values:
                median_range = f"{combined_tab_name}!K3:K{2 + len(median_views_values)}"
                sheets_update_values(
                    service,
                    COMBINED_DB_SPREADSHEET_ID,
                    median_range,
                    median_views_values,
                    value_input_option='USER_ENTERED'
                )
                log_and_print(f"  Updated median views for {len(median_views_values)} rows")
            
            # Update the header for the median views column (Column K, row 2) with current date
            current_date = datetime.now().strftime('%m/%d')
            new_header = f"15 Videos - Median Views ({current_date})"
            header_range = f"{combined_tab_name}!K{HEADER_ROW}"
            sheets_update_values(
                service,
                COMBINED_DB_SPREADSHEET_ID,
                header_range,
                [[new_header]],
                value_input_option='RAW'
            )
            log_and_print(f"  Updated header: {new_header}")
        else:
            log_and_print("\n  No good accounts to write")
        
        log_and_print("=" * 80)
        log_and_print("AGGREGATION COMPLETE")
        log_and_print("=" * 80)
        return True
        
    except Exception as e:
        log_and_print(f"ERROR during aggregation: {e}", "ERROR")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False


def read_sheets_to_process() -> List[Dict[str, str]]:
    """
    Read the list of sheets to process from the combined database.
    Returns a list of dicts with keys: 'name', 'url', 'spreadsheet_id'
    """
    service = setup_google_sheets_service()
    if not service:
        log_and_print("ERROR: Could not setup Google Sheets service to read sheets list", "ERROR")
        return []
    
    try:
        log_and_print(f"Reading sheets list from: {COMBINED_DB_SPREADSHEET_ID}")
        log_and_print(f"Range: {SHEETS_INPUT_RANGE}")
        
        result = sheets_get_values(
            service,
            COMBINED_DB_SPREADSHEET_ID,
            SHEETS_INPUT_RANGE
        )
        
        rows = result.get('values', [])
        if not rows:
            log_and_print("No sheets found in the input range", "WARNING")
            return []
        
        sheets_list = []
        for idx, row in enumerate(rows, start=2):  # Start from 2 (row 2 in sheet)
            if len(row) < 2:
                log_and_print(f"Row {idx}: Not enough columns (need Name and URL). Skipping.", "WARNING")
                continue
            
            name = row[0].strip() if row[0] else f"Sheet{idx}"
            sheet_url = row[1].strip() if row[1] else ""
            
            if not sheet_url:
                log_and_print(f"Row {idx} ({name}): No URL provided. Skipping.", "WARNING")
                continue
            
            spreadsheet_id = extract_spreadsheet_id_from_url(sheet_url)
            if not spreadsheet_id:
                log_and_print(f"Row {idx} ({name}): Could not extract spreadsheet ID from URL: {sheet_url}. Skipping.", "WARNING")
                continue
            
            sheets_list.append({
                'name': name,
                'url': sheet_url,
                'spreadsheet_id': spreadsheet_id
            })
        
        log_and_print(f"Found {len(sheets_list)} sheet(s) to process")
        for sheet_info in sheets_list:
            log_and_print(f"  - {sheet_info['name']}: {sheet_info['spreadsheet_id']}")
        
        return sheets_list
        
    except Exception as e:
        log_and_print(f"ERROR reading sheets list: {e}", "ERROR")
        return []


def _get_enabled_manager_tabs() -> List[str]:
    """Return enabled manager tabs in fixed processing order."""
    tabs: List[str] = []
    if PROCESS_INITIAL_REACHOUT:
        tabs.append(TAB_NAMES['MAIN'])
    if PROCESS_BAD_ACCOUNTS:
        tabs.append(TAB_NAMES['BAD'])
    if PROCESS_GOOD_ACCOUNTS:
        tabs.append(TAB_NAMES['GOOD'])
    if PROCESS_RELIABLE_ACCOUNTS:
        tabs.append(TAB_NAMES['RELIABLE'])
    return tabs


def _pad_row_to_width(row: List[Any], width: int) -> List[Any]:
    row_norm = list(row or [])
    if len(row_norm) < width:
        row_norm.extend([""] * (width - len(row_norm)))
    elif len(row_norm) > width:
        row_norm = row_norm[:width]
    return row_norm


def _row_has_any_content(row: List[Any]) -> bool:
    for cell in row or []:
        if cell is None:
            continue
        if str(cell).strip():
            return True
    return False


def _batch_write_rows_to_tab(service, spreadsheet_id: str, tab_name: str, start_row: int, rows: List[List[Any]], chunk_size: int = 400) -> None:
    """Write rows to tab in contiguous chunks via values.batchUpdate."""
    if not rows:
        return
    for offset in range(0, len(rows), chunk_size):
        chunk = rows[offset:offset + chunk_size]
        chunk_start = start_row + offset
        chunk_end = chunk_start + len(chunk) - 1
        range_name = f"{tab_name}!A{chunk_start}:U{chunk_end}"
        sheets_values_batch_update(
            service,
            spreadsheet_id,
            [{"range": range_name, "values": chunk}],
            value_input_option='USER_ENTERED',
        )


def _batch_apply_cost_formula(service, spreadsheet_id: str, tab_name: str, rows_written: int, chunk_size: int = 500) -> None:
    """Apply Cost for 100k Views formulas for row 3..(2+rows_written)."""
    if rows_written <= 0:
        return
    start_row = DATA_START_ROW
    end_row = DATA_START_ROW + rows_written - 1
    for block_start in range(start_row, end_row + 1, chunk_size):
        block_end = min(end_row, block_start + chunk_size - 1)
        formulas = [[f"=(100000*G{r})/J{r}"] for r in range(block_start, block_end + 1)]
        sheets_values_batch_update(
            service,
            spreadsheet_id,
            [{"range": f"{tab_name}!I{block_start}:I{block_end}", "values": formulas}],
            value_input_option='USER_ENTERED',
        )


def rebuild_manager_sheet_tabs(service, spreadsheet_id: str, manager_label: str = "") -> dict:
    """
    Canonical pass:
    read all enabled tabs -> normalize links -> dedupe by username -> clear row 3+ -> rewrite -> apply formulas.
    """
    tabs_to_process = _get_enabled_manager_tabs()
    if not tabs_to_process:
        return {"row_map": {}, "username_link_row_map": {}, "counts": {}}

    # Keep fixed priority among enabled tabs.
    priority_rank = {
        TAB_NAMES['RELIABLE']: 0,
        TAB_NAMES['GOOD']: 1,
        TAB_NAMES['BAD']: 2,
        TAB_NAMES['MAIN']: 3,
    }
    width = COLUMNS['PAYMENT_INFO'] + 1  # A..U
    all_rows_by_tab: Dict[str, List[List[Any]]] = {tab: [] for tab in tabs_to_process}

    for tab_name in tabs_to_process:
        result = sheets_get_values(service, spreadsheet_id, f"{tab_name}!A:U") or {}
        values = result.get("values", [])
        data_rows = values[HEADER_ROW:] if len(values) >= HEADER_ROW else []
        for row in data_rows:
            padded = _pad_row_to_width(row, width)
            if not _row_has_any_content(padded):
                continue
            canonical_link = normalize_profile_url(padded[COLUMNS['LINK']] if len(padded) > COLUMNS['LINK'] else "")
            if canonical_link:
                padded[COLUMNS['LINK']] = canonical_link
            canonical_username = canonical_username_from_row(padded)
            if canonical_username and not normalize_username(padded[COLUMNS['USERNAME']] if len(padded) > COLUMNS['USERNAME'] else ""):
                padded[COLUMNS['USERNAME']] = f"@{canonical_username}"
            all_rows_by_tab[tab_name].append(padded)

    manager_text = manager_label or spreadsheet_id
    read_counts = {
        TAB_NAMES['MAIN']: len(all_rows_by_tab.get(TAB_NAMES['MAIN'], [])),
        TAB_NAMES['BAD']: len(all_rows_by_tab.get(TAB_NAMES['BAD'], [])),
        TAB_NAMES['GOOD']: len(all_rows_by_tab.get(TAB_NAMES['GOOD'], [])),
        TAB_NAMES['RELIABLE']: len(all_rows_by_tab.get(TAB_NAMES['RELIABLE'], [])),
    }
    _logger.info(
        "[sheet-rebuild] %s read counts: initial=%s bad=%s good=%s reliable=%s",
        manager_text,
        read_counts.get(TAB_NAMES['MAIN'], 0),
        read_counts.get(TAB_NAMES['BAD'], 0),
        read_counts.get(TAB_NAMES['GOOD'], 0),
        read_counts.get(TAB_NAMES['RELIABLE'], 0),
    )

    best_row_by_username: Dict[str, Tuple[int, str, List[Any], int]] = {}
    duplicate_drops = 0
    for tab_name in tabs_to_process:
        for row_idx, row in enumerate(all_rows_by_tab.get(tab_name, [])):
            uname = canonical_username_from_row(row)
            if not uname:
                continue
            candidate = (priority_rank.get(tab_name, 99), tab_name, row, row_idx)
            existing = best_row_by_username.get(uname)
            if existing is None or candidate[0] < existing[0]:
                if existing is not None:
                    duplicate_drops += 1
                best_row_by_username[uname] = candidate
            elif existing is not None:
                duplicate_drops += 1

    rebuilt_rows_by_tab: Dict[str, List[List[Any]]] = {tab: [] for tab in tabs_to_process}
    for uname, (_rank, tab_name, row, _orig_idx) in best_row_by_username.items():
        rebuilt_rows_by_tab.setdefault(tab_name, []).append(row)

    _logger.info(
        "[sheet-rebuild] deduped unique usernames: %s (dropped duplicates: %s)",
        len(best_row_by_username),
        duplicate_drops,
    )

    # Clear rows 3+ across tabs using current grid size; keep headers intact.
    sheet_map = _get_sheet_grid_metadata(service, spreadsheet_id, force_refresh=True)
    for tab_name in tabs_to_process:
        info = sheet_map.get(tab_name, {})
        row_count = int(info.get("rowCount", 0))
        if row_count >= DATA_START_ROW:
            sheets_clear_values(service, spreadsheet_id, f"{tab_name}!A{DATA_START_ROW}:U{row_count}")

    # Write rebuilt rows, then re-apply Cost formula.
    row_map: Dict[str, Tuple[str, int]] = {}
    username_link_row_map: Dict[Tuple[str, str], Tuple[str, int]] = {}
    write_counts: Dict[str, int] = {}
    for tab_name in tabs_to_process:
        rows = rebuilt_rows_by_tab.get(tab_name, [])
        _batch_write_rows_to_tab(service, spreadsheet_id, tab_name, DATA_START_ROW, rows)
        _batch_apply_cost_formula(service, spreadsheet_id, tab_name, len(rows))
        write_counts[tab_name] = len(rows)
        for idx, row in enumerate(rows):
            sheet_row = DATA_START_ROW + idx
            uname = canonical_username_from_row(row)
            if not uname:
                continue
            row_map[uname] = (tab_name, sheet_row)
            link_key = normalize_link_key(row[COLUMNS['LINK']] if len(row) > COLUMNS['LINK'] else "")
            if link_key:
                username_link_row_map[(uname, link_key)] = (tab_name, sheet_row)

    _logger.info(
        "[sheet-rebuild] wrote: initial=%s bad=%s good=%s reliable=%s",
        write_counts.get(TAB_NAMES['MAIN'], 0),
        write_counts.get(TAB_NAMES['BAD'], 0),
        write_counts.get(TAB_NAMES['GOOD'], 0),
        write_counts.get(TAB_NAMES['RELIABLE'], 0),
    )
    _logger.info(
        "[sheet-rebuild] applied cost formula rows: %s=%s, %s=%s, %s=%s, %s=%s",
        TAB_NAMES['MAIN'], write_counts.get(TAB_NAMES['MAIN'], 0),
        TAB_NAMES['BAD'], write_counts.get(TAB_NAMES['BAD'], 0),
        TAB_NAMES['GOOD'], write_counts.get(TAB_NAMES['GOOD'], 0),
        TAB_NAMES['RELIABLE'], write_counts.get(TAB_NAMES['RELIABLE'], 0),
    )

    return {
        "row_map": row_map,
        "username_link_row_map": username_link_row_map,
        "counts": write_counts,
    }


def read_entire_sheet() -> Dict[str, Any]:
    """Read required sheet columns and build row mappings.
    LINK (Column B) is the canonical scrape input; USERNAME (Column A) is optional."""
    service = setup_google_sheets_service()
    if not service:
        return {}
    
    # Google Sheets ID from the URL
    spreadsheet_id = SPREADSHEET_ID
    
    # Determine which tabs to process based on toggles
    tabs_to_process = _get_enabled_manager_tabs()
    
    if not tabs_to_process:
        log_and_print("No tabs enabled for processing. Enable at least one PROCESS_* toggle.", "WARNING")
        return {}
    
    # Read only required columns for run orchestration/mapping/filtering (A..J):
    # A username, B link, G TikTok Price, J Median Views
    sheet_data = {
        'username_to_row': {},  # Maps username -> (tab_name, row_number)
        'link_to_row': {},  # Maps canonical link key -> (tab_name, row_number)
        'username_link_to_row': {},  # Maps (username_normalized, link_key) -> (tab_name, row_number)
        'username_to_prev_median': {},  # Maps username -> previous median views
        'link_to_prev_median': {},  # Maps canonical link key -> previous median views
        'profiles_to_scrape': [],  # List of dicts: {profile_url, username, username_normalized}
        'profiles_for_na': [],  # List of dicts for rows with empty/NA LINK
    }
    
    try:
        log_and_print(f"Reading from {len(tabs_to_process)} tab(s): {', '.join(tabs_to_process)}")
        
        for tab_name in tabs_to_process:
            range_name = f"{tab_name}!A:J"
            
            log_and_print(f"Reading from tab: {tab_name}")
            result = sheets_get_values(
                service,
                spreadsheet_id,
                range_name
            )
            
            values = result.get('values', [])
            if not values or len(values) < HEADER_ROW:
                log_and_print(f"No data found in tab '{tab_name}'", "WARNING")
                continue
            
            # Build mapping from rows
            # Headers are on row 2, data starts on row 3
            # Column A (index 0) = TikTok USERNAME
            # Column B (index 1) = Link (TikTok URL)
            # Column J (index 9) = Previous median views
            
            for i, row in enumerate(values[HEADER_ROW:], start=DATA_START_ROW):  # Skip rows 1-2 (header on row 2), start counting from row 3
                if not row or len(row) == 0:
                    continue
                
                username_raw = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                link_raw = row[COLUMNS['LINK']] if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                raw_link = normalize_profile_link(str(link_raw or ""))
                canon_link = normalize_profile_url(raw_link)
                if canon_link is not None:
                    log_and_print(f"[url][canon] raw={raw_link} -> canon={canon_link}")
                    link = canon_link
                else:
                    log_and_print(f"[url][invalid] raw={raw_link}")
                    link = raw_link
                username_normalized = canonical_username_from_row(row)
                if not username_normalized:
                    continue
                if not link or str(link).upper() in ['NA', 'N/A', 'NULL', 'EMPTY'] or str(link).lower() == 'tiktok':
                    continue

                username_display = normalize_username(username_raw) or username_normalized
                link_key = normalize_link_key(link)

                # Store row mappings for strict username+link rows only.
                sheet_data['username_to_row'][username_normalized] = (tab_name, i)
                sheet_data['link_to_row'][link_key] = (tab_name, i)
                sheet_data['username_link_to_row'][(username_normalized, link_key)] = (tab_name, i)
                
                # Get previous median views from column J (index 9) if available
                if len(row) > COLUMNS['MEDIAN_VIEWS'] and row[COLUMNS['MEDIAN_VIEWS']]:
                    prev_median = row[COLUMNS['MEDIAN_VIEWS']]
                    if prev_median and str(prev_median).upper() != 'NA':
                        try:
                            # Remove commas if present and convert to int
                            prev_value_clean = str(prev_median).replace(',', '')
                            prev_val = int(float(prev_value_clean))
                            sheet_data['username_to_prev_median'][username_normalized] = prev_val
                            sheet_data['link_to_prev_median'][link_key] = prev_val
                        except (ValueError, TypeError):
                            pass
                
                # When FULL_RUN is False, only process rows where:
                # - TikTok Price (Column G) is a number and not empty
                # - OR 15 Videos - Median Views (Column J) is empty
                # (Perf Based Price column is display-only, not used in logic)
                # When FULL_RUN is True, skip this filtering and process all rows
                if not FULL_RUN:
                    # Check TikTok Price (Column G, index 6) - is it a valid number and not empty?
                    tiktok_price_is_valid_number = False
                    if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                        tiktok_price_str = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                        if tiktok_price_str:  # Not empty
                            # Check if it's a valid number (not text/error)
                            if not tiktok_price_str.startswith('#'):  # Not an Excel error like #DIV/0!, #VALUE!, etc.
                                try:
                                    # Try to parse as number
                                    if isinstance(row[COLUMNS['TIKTOK_PRICE']], (int, float)):
                                        float(row[COLUMNS['TIKTOK_PRICE']])
                                        tiktok_price_is_valid_number = True
                                    else:
                                        price_str_cleaned = tiktok_price_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').strip()
                                        float(price_str_cleaned)
                                        tiktok_price_is_valid_number = True
                                except (ValueError, TypeError):
                                    # Not a valid number - contains text/error
                                    tiktok_price_is_valid_number = False
                    # # Perf Based Price no longer used in logic (column is display-only)
                    # perf_based_price_is_valid_number = False
                    # if len(row) > COLUMNS['PERF_BASED_PRICE'] and row[COLUMNS['PERF_BASED_PRICE']] is not None:
                    #     ...
                    
                    # Check 15 Videos - Median Views (Column J, index 9) - is it empty?
                    median_views_empty = True
                    if len(row) > COLUMNS['MEDIAN_VIEWS'] and row[COLUMNS['MEDIAN_VIEWS']] is not None:
                        median_str = str(row[COLUMNS['MEDIAN_VIEWS']]).strip()
                        if median_str and median_str.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                            median_views_empty = False
                    
                    # Only process if: (TikTok Price is a number and not empty) OR (Median Views is empty)
                    if not (tiktok_price_is_valid_number or median_views_empty):
                        _logger.debug(f"   Row {i} (@{username_normalized}): Skipping scraping (FULL_RUN=False): TikTok Price is not a valid number and Median Views is not empty")
                        continue
                
                # Keep sheet LINK as canonical navigation URL.
                if canon_link is not None:
                    sheet_data['profiles_to_scrape'].append({
                        "profile_url": canon_link,
                        "username": username_display,
                        "username_normalized": username_normalized,
                        "link_key": normalize_link_key(canon_link),
                    })
                else:
                    # Invalid/non-profile URL - route to NA handling and skip scraping.
                    log_and_print(f"   Profile @{username_normalized} has invalid TikTok profile URL - writing NA and skipping")
                    sheet_data['profiles_for_na'].append({
                        "profile_url": raw_link,
                        "username": username_display,
                        "username_normalized": username_normalized,
                        "link_key": normalize_link_key(raw_link),
                    })
                    continue
        
        log_and_print(f"Found {len(sheet_data['profiles_to_scrape'])} TikTok profiles to scrape across {len(tabs_to_process)} tab(s)")
        if 'profiles_for_na' in sheet_data:
            log_and_print(f"Found {len(sheet_data['profiles_for_na'])} profiles with invalid/empty/NA URLs (will write NA values)")
        log_and_print(f"Loaded {len(sheet_data['username_to_row'])} username mappings")
        log_and_print(f"Found {len(sheet_data['username_to_prev_median'])} profiles with previous median views")
        
        return sheet_data
        
    except Exception as e:
        log_and_print(f"ERROR reading from Google Sheets: {e}", "ERROR")
        return {}


def write_na_to_sheet(username: str, profile_url: str = "") -> bool:
    """Queue NA values for failed/empty profiles using cached username lookup."""
    try:
        spreadsheet_id = SPREADSHEET_ID
        username_normalized = username.lower().lstrip('@')
        profile_url = normalize_profile_url(profile_url) or normalize_profile_link(profile_url)
        if not username_normalized or not profile_url:
            _logger.warning(f"Skipping NA write: missing username/link ({username}, {profile_url})")
            return False
        location = _resolve_profile_location(username_normalized, profile_url)
        if not location:
            _logger.error(f"ERROR: Username {username} not found in any enabled tab - skipping")
            return False
        found_tab, row_number = location
        
        # Write NA values to the analytics columns
        # Column J: Median Views, Column K: Change, Column L: < 10k, Column M: 10k-100k, Column N: > 100k, Column O: 15 Videos Ago Date, Column P: Latest Video Date
        range_to_update = f"{found_tab}!J{row_number}:P{row_number}"
        values_to_write = [["NA", "NA", "NA", "NA", "NA", "NA", "NA"]]
        queue_sheet_value_update(spreadsheet_id, range_to_update, values_to_write, value_input_option='USER_ENTERED')
        
        _logger.info(f"Successfully updated {username} with NA values in tab '{found_tab}' (row {row_number})")
        return True
        
    except Exception as e:
        _logger.error(f"ERROR writing NA to Google Sheets for {username}: {e}")
        return False


def update_sheet_headers() -> bool:
    """Update the Google Sheets headers with current date in all enabled tabs.
    Headers are on row 2."""
    service = setup_google_sheets_service()
    if not service:
        return False
    
    # Google Sheets ID from the URL
    spreadsheet_id = SPREADSHEET_ID
    
    try:
        # Get current date in MM/DD format
        current_date = datetime.now().strftime('%m/%d')
        
        # Determine which tabs to update based on toggles
        tabs_to_update = []
        if PROCESS_INITIAL_REACHOUT:
            tabs_to_update.append(TAB_NAMES['MAIN'])
        if PROCESS_BAD_ACCOUNTS:
            tabs_to_update.append(TAB_NAMES['BAD'])
        if PROCESS_GOOD_ACCOUNTS:
            tabs_to_update.append(TAB_NAMES['GOOD'])
        if PROCESS_RELIABLE_ACCOUNTS:
            tabs_to_update.append(TAB_NAMES['RELIABLE'])
        
        # Update the header for the median views column (Column J, row 2)
        new_header = f"15 Videos - Median Views ({current_date})"

        for tab_name in tabs_to_update:
            range_name = f"{tab_name}!J{HEADER_ROW}"
            
            sheets_update_values(
                service,
                spreadsheet_id,
                range_name,
                [[new_header]],
                value_input_option='RAW'
            )
            
            log_and_print(f"Updated header in '{tab_name}': {new_header}")
        
        return True
        
    except Exception as e:
        log_and_print(f"ERROR updating sheet headers: {e}", "ERROR")
        return False


def write_analytics_to_sheet(analytics_data: Dict[str, Any]) -> bool:
    """Queue analytics write using cached username mapping and previous median cache."""
    try:
        spreadsheet_id = SPREADSHEET_ID
        username = analytics_data.get('username', '').lstrip('@')
        username_normalized = username.lower().lstrip('@')
        profile_url = normalize_profile_url(str(analytics_data.get("profile_url") or "")) or normalize_profile_link(str(analytics_data.get("profile_url") or ""))
        if not username_normalized or not profile_url:
            _logger.warning(f"   Skipping analytics write: missing username/link (@{username}, {profile_url})")
            return False
        location = _resolve_profile_location(username_normalized, profile_url)
        if not location:
            _logger.error(f"   ERROR: Username @{username} not found in any enabled tab - skipping")
            return False
        found_tab, row_number = location
        previous_median_views = _current_sheet_data.get('username_to_prev_median', {}).get(username_normalized)
        if previous_median_views is None:
            previous_median_views = _current_sheet_data.get('link_to_prev_median', {}).get(normalize_link_key(profile_url))
        
        # Prepare the data to write
        # Column J: 15 Videos - Median Views
        # Column K: Change in Median Views (vs last run)
        # Column L: < 10k
        # Column M: 10k - 100k
        # Column N: > 100k
        # Column O: 15 Videos Ago Date
        # Column P: Latest Video Date
        
        median_views = analytics_data.get('median_views', 0)
        under_10k = analytics_data.get('videos_under_10k', 0)
        between_10k_100k = analytics_data.get('videos_10k_to_100k', 0)
        over_100k = analytics_data.get('videos_100k_plus', 0)
        fifteenth_date = analytics_data.get('fifteenth_video_date', '')
        latest_date = analytics_data.get('latest_video_date', '')
        
        # Calculate change in median views
        if previous_median_views is not None:
            change_in_median = median_views - previous_median_views
        else:
            change_in_median = "NA"
        
        # Update the specific cells for this row (columns J through P)
        range_to_update = f"{found_tab}!J{row_number}:P{row_number}"
        
        # Ensure dates are properly formatted without any prefix
        formatted_fifteenth_date = fifteenth_date if fifteenth_date else ""
        formatted_latest_date = latest_date if latest_date else ""
        
        values_to_write = [[median_views, change_in_median, under_10k, between_10k_100k, over_100k, formatted_fifteenth_date, formatted_latest_date]]
        
        queue_sheet_value_update(spreadsheet_id, range_to_update, values_to_write, value_input_option='USER_ENTERED')
        if _current_sheet_data is not None:
            _current_sheet_data.setdefault('username_to_prev_median', {})[username_normalized] = int(median_views)
            if profile_url:
                _current_sheet_data.setdefault('link_to_prev_median', {})[normalize_link_key(profile_url)] = int(median_views)
        
        _logger.info(f"Successfully updated analytics for @{username} in tab '{found_tab}' (row {row_number})")
        if LOG_LEVEL == "DEBUG":
            _logger.info(f"   Median views: {median_views:,}")
            if change_in_median != "NA":
                _logger.info(f"   Change in median views: {change_in_median:+,}")
            else:
                _logger.info(f"   Change in median views: NA (no previous data)")
            _logger.info(f"   Under 10k: {under_10k}")
            _logger.info(f"   10k-100k: {between_10k_100k}")
            _logger.info(f"   Over 100k: {over_100k}")
            _logger.info(f"   15th video date: {fifteenth_date}")
            _logger.info(f"   Latest video date: {latest_date}")
        
        return True
        
    except Exception as e:
        _logger.error(f"ERROR writing to Google Sheets: {e}")
        return False


# Profile grid + extension meta selectors (fast path)
PROFILE_GRID_ITEMS = '[data-e2e="user-post-item"]'
PROFILE_ANCHORS = '[data-e2e="user-post-item"] a[href*="/video/"], [data-e2e="user-post-item"] a[href*="/photo/"]'
PROFILE_METAS = '[data-e2e="user-post-item"] a[href*="/video/"] ov-ext-meta, [data-e2e="user-post-item"] a[href*="/photo/"] ov-ext-meta'

_PROFILE_WAIT_ALL_READY_JS = """(args) => {
    const appOk = !!document.querySelector(args.appSel);
    const items = document.querySelectorAll(args.gridItemsSel).length;
    const anchors = document.querySelectorAll(args.anchorsSel).length;
    return appOk && items >= args.minItems && anchors >= args.minItems;
}"""

_PROFILE_WAIT_METAS_JS = "([sel, n]) => document.querySelectorAll(sel).length >= n"
_PROFILE_DEBUG_COUNTS_JS = """(args) => ({
    gridItems: document.querySelectorAll(args.gridItemsSel).length,
    anchors: document.querySelectorAll(args.anchorsSel).length,
    metas: document.querySelectorAll(args.metasSel).length
})"""
_PROFILE_READY_OR_EMPTY_JS = """(args) => {
    const items = document.querySelectorAll(args.gridItemsSel).length;
    const anchors = document.querySelectorAll(args.anchorsSel).length;
    const root = document.querySelector(args.rootSel) || document;
    const txt = (root.innerText || root.textContent || "").toLowerCase();
    const emptyPhrases = [
        "no videos yet",
        "no content",
        "this account is private",
        "private account",
        "no posts",
    ];
    const emptyOrPrivate = emptyPhrases.some(p => txt.includes(p));
    return { hasItems: (items >= args.minItems || anchors >= args.minItems), emptyOrPrivate: emptyOrPrivate };
}"""
_PROFILE_DEBUG_SAMPLE_JS = """(args) => {
    const nodes = Array.from(document.querySelectorAll(args.gridItemsSel));
    const anchorSel = 'a[href*="/video/"], a[href*="/photo/"]';
    const anchorNodes = nodes.filter(n => (n.tagName || "").toUpperCase() === "A" || (n.matches && n.matches(anchorSel)));
    const sample = nodes.slice(0, 5).map((n) => ({
        tagName: (n.tagName || "").toUpperCase(),
        id: n.id || "",
        selfAnchorMatch: !!(n.matches && n.matches(anchorSel)),
        descendantAnchorFound: !!n.querySelector(anchorSel),
    }));
    return {
        totalNodes: nodes.length,
        anchorLikeNodes: anchorNodes.length,
        sample,
    };
}"""

_PROFILE_SCRAPE_ROWS_JS = """(args) => {
    const txt = (el) => (el && (el.innerText || el.textContent) ? (el.innerText || el.textContent).trim() : "");
    const out = [];
    const targetVideos = args.targetVideos || 15;
    const anchors = Array.from(document.querySelectorAll(args.anchorsSel || ""));

    for (const anchor of anchors) {
        if (out.length >= targetVideos) break;
        try {
            const container = (args.gridItemClosestSel ? anchor.closest(args.gridItemClosestSel) : null) || anchor;

            const pinnedSelectors = [
                '[data-e2e*="badge"]',
                '[data-e2e*="pin"]',
                '[aria-label*="pin"]',
                '[title*="pin"]',
            ];

            let isPinned = false;
            for (const selector of pinnedSelectors) {
                const pinNodes = container.querySelectorAll(selector);
                for (const pinEl of pinNodes) {
                    const txt = (pinEl.innerText || pinEl.textContent || "").toLowerCase();
                    if (txt.includes("pinned")) {
                        isPinned = true;
                        break;
                    }
                }
                if (isPinned) break;
            }
            if (isPinned) continue;

            let href = anchor.getAttribute("href") || anchor.href || "";
            if (!href || !anchor) continue;
            if (!href.includes("/video/") && !href.includes("/photo/")) continue;
            const absHref = href.startsWith("http") ? href : ("https://www.tiktok.com" + href);

            let contentType = "";
            let videoId = "";
            if (absHref.includes("/video/")) {
                contentType = "video";
                videoId = ((absHref.split("/video/")[1] || "").split(/[?#]/)[0] || "").trim();
            } else if (absHref.includes("/photo/")) {
                contentType = "photo";
                videoId = ((absHref.split("/photo/")[1] || "").split(/[?#]/)[0] || "").trim();
            }
            if (!videoId) continue;

            let meta = anchor.querySelector(args.metaSel);
            if (!meta) {
                meta = anchor.parentElement ? anchor.parentElement.querySelector(args.metaSel) : null;
            }
            if (!meta) {
                meta = container.querySelector(args.metaSel);
            }
            const root = meta && meta.shadowRoot ? meta.shadowRoot : null;

            let uploadDate = root ? txt(root.querySelector('.m[data-k="m"]:not([data-t]) .t')) : "";
            let uploadTime = root ? txt(root.querySelector('.m[data-k="m"][data-t="1"] .t')) : "";
            if (!uploadDate && root) {
                const rootTimeNode = root.querySelector('time[datetime], time, [datetime]');
                if (rootTimeNode) {
                    const dtAttr = rootTimeNode.getAttribute ? (rootTimeNode.getAttribute("datetime") || "") : "";
                    const dtText = txt(rootTimeNode);
                    const rawDt = (dtAttr || dtText || "").trim();
                    if (rawDt) {
                        if (rawDt.includes("T")) {
                            const parts = rawDt.split("T");
                            uploadDate = (parts[0] || "").trim();
                            if (!uploadTime && parts.length > 1) uploadTime = (parts[1] || "").trim();
                        } else {
                            uploadDate = rawDt;
                        }
                    }
                }
            }
            if (!uploadDate) {
                const containerTimeNode = container.querySelector('time[datetime], time, [data-e2e*="video-date"], [data-e2e*="browse-video-date"]');
                if (containerTimeNode) {
                    const dtAttr = containerTimeNode.getAttribute ? (containerTimeNode.getAttribute("datetime") || "") : "";
                    const dtText = txt(containerTimeNode);
                    const rawDt = (dtAttr || dtText || "").trim();
                    if (rawDt) {
                        if (rawDt.includes("T")) {
                            const parts = rawDt.split("T");
                            uploadDate = (parts[0] || "").trim();
                            if (!uploadTime && parts.length > 1) uploadTime = (parts[1] || "").trim();
                        } else {
                            uploadDate = rawDt;
                        }
                    }
                }
            }
            let viewsText = root ? txt(root.querySelector('.m[data-k="p"] .t')) : "";
            if (!viewsText) {
                viewsText = txt(anchor.querySelector('strong[data-e2e="video-views"], strong.video-count'));
            }
            const durationText = root ? txt(root.querySelector('.d[data-k="d"] .t')) : "";

            out.push({
                href: absHref,
                video_id: videoId,
                content_type: contentType,
                upload_date: uploadDate,
                upload_time: uploadTime,
                views_text: viewsText,
                duration: durationText,
                meta_found: !!meta,
                shadow_found: !!root,
            });
        } catch (e) {
            // continue scanning containers
        }
    }
    return out;
}"""

_PROFILE_SCRAPE_BASIC_JS = """(args) => {
    const txt = (el) => (el && (el.innerText || el.textContent) ? (el.innerText || el.textContent).trim() : "");
    const out = [];
    const targetVideos = args.targetVideos || 15;
    const anchors = Array.from(document.querySelectorAll(args.anchorsSel || ""));

    for (const anchor of anchors) {
        if (out.length >= targetVideos) break;
        try {
            const container = (args.gridItemClosestSel ? anchor.closest(args.gridItemClosestSel) : null) || anchor;
            const pinnedSelectors = [
                '[data-e2e*="badge"]',
                '[data-e2e*="pin"]',
                '[aria-label*="pin"]',
                '[title*="pin"]',
            ];
            let isPinned = false;
            for (const selector of pinnedSelectors) {
                const pinNodes = container.querySelectorAll(selector);
                for (const pinEl of pinNodes) {
                    const pinTxt = (pinEl.innerText || pinEl.textContent || "").toLowerCase();
                    if (pinTxt.includes("pinned")) {
                        isPinned = true;
                        break;
                    }
                }
                if (isPinned) break;
            }
            if (isPinned) continue;

            let href = anchor.getAttribute("href") || anchor.href || "";
            if (!href) continue;
            if (!href.includes("/video/") && !href.includes("/photo/")) continue;
            const absHref = href.startsWith("http") ? href : ("https://www.tiktok.com" + href);

            let contentType = "";
            let videoId = "";
            if (absHref.includes("/video/")) {
                contentType = "video";
                videoId = ((absHref.split("/video/")[1] || "").split(/[?#]/)[0] || "").trim();
            } else if (absHref.includes("/photo/")) {
                contentType = "photo";
                videoId = ((absHref.split("/photo/")[1] || "").split(/[?#]/)[0] || "").trim();
            }
            if (!videoId) continue;

            const viewsText = txt(anchor.querySelector('strong[data-e2e="video-views"], strong.video-count'));
            out.push({
                href: absHref,
                video_id: videoId,
                content_type: contentType,
                upload_date: "",
                upload_time: "",
                views_text: viewsText,
                meta_found: false,
                shadow_found: false,
            });
        } catch (e) {
            // continue
        }
    }
    return out;
}"""


def wait_for_profile_grid_and_extension(
    page,
    timeout_ms: int = 12000,
    min_items: int = 1,
    min_metas: int = 1,
) -> Dict[str, bool]:
    """
    Staged readiness pipeline:
    - basic page readiness (best-effort, no throw)
    - detect has-items OR empty/private
    - extension metas (best-effort if has-items)
    """
    status = {
        "has_items": False,
        "has_metas": False,
        "empty_or_private": False,
        "unknown": False,
    }

    # Stage 0: basic page readiness (best-effort)
    try:
        page.wait_for_function(
            "() => (!!document.querySelector('#app')) || (!!document.body && (document.body.innerText || '').length > 0)",
            timeout=2500,
        )
    except Exception:
        pass

    # Stage 1: items OR empty/private state
    stage = {"hasItems": False, "emptyOrPrivate": False}
    try:
        page.wait_for_function(
            """(args) => {
                const items = document.querySelectorAll(args.gridItemsSel).length;
                const anchors = document.querySelectorAll(args.anchorsSel).length;
                const root = document.querySelector(args.rootSel) || document;
                const txt = (root.innerText || root.textContent || "").toLowerCase();
                const emptyPhrases = ["no videos yet","no content","this account is private","private account","no posts"];
                return (items >= args.minItems || anchors >= args.minItems) || emptyPhrases.some(p => txt.includes(p));
            }""",
            arg={"gridItemsSel": PROFILE_GRID_ITEMS, "anchorsSel": PROFILE_ANCHORS, "rootSel": "#main-content-others_homepage", "minItems": min_items},
            timeout=min(timeout_ms, 8000),
        )
        stage = page.evaluate(
            _PROFILE_READY_OR_EMPTY_JS,
            {"gridItemsSel": PROFILE_GRID_ITEMS, "anchorsSel": PROFILE_ANCHORS, "rootSel": "#main-content-others_homepage", "minItems": min_items},
        ) or {"hasItems": False, "emptyOrPrivate": False}
    except Exception:
        stage = {"hasItems": False, "emptyOrPrivate": False}

    status["has_items"] = bool(stage.get("hasItems"))
    status["empty_or_private"] = bool(stage.get("emptyOrPrivate"))

    # Stage 2: metas are best-effort only
    if status["has_items"]:
        try:
            page.wait_for_function(_PROFILE_WAIT_METAS_JS, arg=[PROFILE_METAS, min_metas], timeout=min(timeout_ms, 15000))
            status["has_metas"] = True
        except Exception:
            status["has_metas"] = False
        # Final fast check: metas may already be attached even if wait timed out.
        if not status["has_metas"]:
            try:
                status["has_metas"] = page.locator(PROFILE_METAS).count() >= max(1, int(min_metas))
            except Exception:
                pass

    status["unknown"] = (not status["has_items"]) and (not status["empty_or_private"])
    return status


def get_profile_grid_debug_counts(page) -> Dict[str, int]:
    """Return profile grid/anchor/meta counts for diagnostics."""
    try:
        return page.evaluate(
            _PROFILE_DEBUG_COUNTS_JS,
            {"gridItemsSel": PROFILE_GRID_ITEMS, "anchorsSel": PROFILE_ANCHORS, "metasSel": PROFILE_METAS},
        ) or {"gridItems": 0, "anchors": 0, "metas": 0}
    except Exception:
        return {"gridItems": 0, "anchors": 0, "metas": 0}


def get_profile_grid_debug_sample(page) -> Dict[str, Any]:
    """Return focused debug sample to diagnose anchor-vs-container mismatches."""
    try:
        return page.evaluate(_PROFILE_DEBUG_SAMPLE_JS, {"gridItemsSel": PROFILE_GRID_ITEMS}) or {}
    except Exception:
        return {}


def _rows_to_profile_results(page, video_data: List[Dict[str, Any]], profile_url_before: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    _logger.info(f"   Collected {len(video_data)} videos with data (pinned videos already filtered out)")
    if video_data:
        meta_found_count = sum(1 for r in video_data if r.get("meta_found"))
        shadow_found_count = sum(1 for r in video_data if r.get("shadow_found"))
        views_nonempty_count = sum(1 for r in video_data if str(r.get("views_text") or "").strip())
        date_nonempty_count = sum(1 for r in video_data if str(r.get("upload_date") or "").strip())
        _logger.info(
            "   [profile-meta] metas=%s shadow=%s views=%s date=%s",
            meta_found_count,
            shadow_found_count,
            views_nonempty_count,
            date_nonempty_count,
        )
        if date_nonempty_count == 0:
            sample = video_data[0] if video_data else {}
            sample_compact = {
                "meta_found": bool(sample.get("meta_found")) if isinstance(sample, dict) else False,
                "shadow_found": bool(sample.get("shadow_found")) if isinstance(sample, dict) else False,
                "upload_date": str(sample.get("upload_date") or "") if isinstance(sample, dict) else "",
                "upload_time": str(sample.get("upload_time") or "") if isinstance(sample, dict) else "",
                "views_text": str(sample.get("views_text") or "") if isinstance(sample, dict) else "",
                "content_type": str(sample.get("content_type") or "") if isinstance(sample, dict) else "",
            }
            raw_snippet = str(sample_compact)
            if len(raw_snippet) > 220:
                raw_snippet = raw_snippet[:220] + "..."
            _logger.warning("[profile-meta][warn] upload_date missing for all items — likely selector drift")
            _logger.info("   [profile-meta][warn] sample=%s", raw_snippet)

    for i, item_info in enumerate(video_data, 1):
        href = str(item_info.get("href") or item_info.get("url") or "")
        content_id = str(item_info.get("video_id") or "")
        content_type = str(item_info.get("content_type") or "")
        if not href or not content_id or content_type not in ("video", "photo"):
            continue
        views_text = str(item_info.get("views_text") or "")
        views_val = parse_view_count(views_text) if views_text else -1
        upload_date = str(item_info.get("upload_date") or "")
        upload_time = str(item_info.get("upload_time") or "")
        create_ts = parse_upload_datetime_to_epoch(upload_date, upload_time)
        _logger.info(f"   [{i}/{len(video_data)}] {content_type.title()}: {content_id}")
        results.append({
            "video_id": content_id,
            "content_type": content_type,
            "views": int(views_val),
            "create_ts": int(create_ts),
            "url": href,
        })

    if page and page.url != profile_url_before:
        _logger.warning(f"   WARNING: Profile URL changed during collection: {profile_url_before} -> {page.url}")
    _logger.info(f"   Final results: {len(results)} items processed")
    return results


def parse_upload_datetime_to_epoch(upload_date: str, upload_time: str) -> int:
    """Parse extension upload_date/upload_time to epoch seconds (UTC)."""
    d = (upload_date or "").strip()
    t = (upload_time or "").strip()
    if not d:
        return 0

    dt_obj = None
    date_fmts = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d", "%m-%d"]
    for fmt in date_fmts:
        try:
            parsed = datetime.strptime(d, fmt)
            if "%Y" in fmt:
                dt_obj = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
            else:
                year = datetime.now(timezone.utc).year
                dt_obj = datetime(year, parsed.month, parsed.day, tzinfo=timezone.utc)
                if dt_obj > datetime.now(timezone.utc):
                    dt_obj = dt_obj.replace(year=year - 1)
            break
        except Exception:
            continue

    if dt_obj is None:
        return 0

    if not t:
        return int(dt_obj.timestamp())

    time_fmts = ["%H:%M", "%I:%M %p", "%I:%M%p", "%H:%M:%S"]
    for tfmt in time_fmts:
        try:
            tp = datetime.strptime(t, tfmt)
            dt_obj = dt_obj.replace(hour=tp.hour, minute=tp.minute, second=getattr(tp, "second", 0))
            return int(dt_obj.timestamp())
        except Exception:
            continue

    return int(dt_obj.timestamp())


def ui_fallback_collect_recent(page, days: int, expected_username: Optional[str]) -> List[Dict[str, Any]]:  # type: ignore
    """
    Collect first loaded profile-grid items directly from the profile page.
    Does not open item pages and does not scroll.
    The days parameter is kept for compatibility but not used for filtering.
    """
    results: List[Dict[str, Any]] = []
    profile_url_before = page.url

    try:
        video_data = page.evaluate(
            _PROFILE_SCRAPE_ROWS_JS,
            {
                "anchorsSel": PROFILE_ANCHORS,
                "metaSel": "ov-ext-meta",
                "targetVideos": 15,
                "gridItemClosestSel": '[data-e2e="user-post-item"]',
                "debug": DEBUG_PROFILE_SCRAPE,
            },
        ) or []
    except Exception as e:
        _logger.warning(f"   Error collecting profile grid data: {e}")
        video_data = []

    if DEBUG_PROFILE_SCRAPE or not video_data:
        counts = get_profile_grid_debug_counts(page)
        _logger.info(
            "   [profile-debug] grid_items=%s anchors=%s metas=%s rows=%s",
            counts.get("gridItems", 0),
            counts.get("anchors", 0),
            counts.get("metas", 0),
            len(video_data),
        )
        if not video_data:
            sample = get_profile_grid_debug_sample(page)
            if sample:
                _logger.info(
                    "   [profile-debug] anchor_like_nodes=%s/%s sample=%s",
                    sample.get("anchorLikeNodes", 0),
                    sample.get("totalNodes", 0),
                    sample.get("sample", []),
                )
        if DEBUG_PROFILE_SCRAPE and video_data:
            _logger.info("   [profile-debug] first meta slots: %s", video_data[0].get("meta_slots"))

    return _rows_to_profile_results(page, video_data, profile_url_before)


def ui_collect_basic(page, target_videos: int = 15) -> List[Dict[str, Any]]:
    """Basic non-extension scrape path when ov-ext-meta is unavailable."""
    profile_url_before = page.url
    try:
        rows = page.evaluate(
            _PROFILE_SCRAPE_BASIC_JS,
            {
                "anchorsSel": PROFILE_ANCHORS,
                "targetVideos": target_videos,
                "gridItemClosestSel": '[data-e2e="user-post-item"]',
            },
        ) or []
    except Exception as e:
        _logger.warning(f"   Error collecting basic profile grid data: {e}")
        rows = []
    return _rows_to_profile_results(page, rows, profile_url_before)


def parse_view_count(text_number: str) -> int:
    s = text_number.strip().lower().replace(",", "")
    try:
        if s.endswith("k"):
            return int(float(s[:-1]) * 1_000)
        if s.endswith("m"):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith("b"):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(float(s))
    except Exception:
        return -1


def login_to_tiktok(page) -> bool:
    """Login to TikTok using the provided credentials."""
    try:
        _logger.info("   Logging into TikTok...")
        
        # Go to login page
        login_url = "https://www.tiktok.com/login/phone-or-email/email"
        page.goto(login_url, wait_until="domcontentloaded")
        time.sleep(5)  # Wait longer for page to fully load
        
        # Fill in email - try multiple selectors
        email_selectors = [
            'input[placeholder="Email or username"]',
            'input[name="username"]',
            'input[type="email"]', 
            'input[placeholder*="email"]',
            'input[placeholder*="Email"]',
            'input[data-testid*="email"]',
            'input[aria-label*="email"]',
            'input[aria-label*="Email"]'
        ]
        
        email_input = None
        for selector in email_selectors:
            email_input = page.query_selector(selector)
            if email_input:
                break
        
        if email_input:
            email_input.fill("aliremoteali07@gmail.com")
            time.sleep(2)
        else:
            _logger.info("   ERROR: Could not find email input field")
            return False
        
        # Fill in password - try multiple selectors
        password_selectors = [
            'input[type="password"][placeholder="Password"]',
            'input[type="password"]',
            'input[name="password"]',
            'input[data-testid*="password"]',
            'input[aria-label*="password"]',
            'input[aria-label*="Password"]'
        ]
        
        password_input = None
        for selector in password_selectors:
            password_input = page.query_selector(selector)
            if password_input:
                break
        
        if password_input:
            password_input.fill("Aliessowika@7")
            time.sleep(2)
        else:
            _logger.info("   ERROR: Could not find password input field")
            return False
        
        # Click login button - try multiple selectors
        login_selectors = [
            'button[type="submit"][data-e2e="login-button"]',
            'button[data-e2e="login-button"]',
            'button[type="submit"]',
            'button:has-text("Log in")',
            'button:has-text("Login")',
            'button:has-text("Sign in")',
            '[data-testid*="login"]',
            '[data-testid*="submit"]'
        ]
        
        login_button = None
        for selector in login_selectors:
            login_button = page.query_selector(selector)
            if login_button:
                break
        
        if login_button:
            # Check if button is disabled
            is_disabled = login_button.get_attribute("disabled")
            if is_disabled:
                _logger.info("   WARNING: Login button is disabled, waiting for it to become enabled...")
                # Wait for button to become enabled
                for _ in range(10):  # Wait up to 10 seconds
                    time.sleep(1)
                    if not login_button.get_attribute("disabled"):
                        break
            
            login_button.click()
        else:
            _logger.info("   ERROR: Could not find login button")
            return False
        
        # Wait up to 2 minutes total for login to complete, checking every 2 seconds
        _logger.info("   Waiting for login to complete (checking every 2 seconds, up to 2 minutes)...")
        max_wait_time = 120  # 2 minutes in seconds
        check_interval = 2  # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            # Check if login was successful by looking for redirect or profile elements
            current_url = page.url
            
            # Check if we're redirected away from login page
            if "login" not in current_url.lower():
                _logger.info(f"   Login successful! (redirected away from login page after {elapsed_time} seconds)")
                return True
            
            # Also check for profile elements that might indicate successful login
            profile_indicators = [
                '[data-e2e="nav-profile"]',
                '[data-e2e="profile"]',
                '.profile',
                '[data-e2e="user-avatar"]',
                'button[data-e2e="nav-profile"]'
            ]
            
            for indicator in profile_indicators:
                if page.query_selector(indicator):
                    _logger.info(f"   Login successful! (found profile indicator: {indicator} after {elapsed_time} seconds)")
                    return True
        
        # If we get here, we've waited 2 minutes and login didn't complete
        _logger.info(f"   Login check timeout after {max_wait_time} seconds - still on login page")
        _logger.info(f"   Current URL: {page.url}")
        return False
            
    except Exception as e:
        _logger.info(f"   ERROR during login: {e}")
        return False


def is_account_not_found(page) -> bool:
    """Check if the page shows 'Couldn't find this account' error."""
    try:
        # Single robust DOM-text probe (no selector waits, no hashed classes)
        return bool(page.evaluate(
            """() => {
                const root = document.querySelector('#main-content-others_homepage') || document;
                const txt = (root.innerText || root.textContent || '').toLowerCase();
                return (
                    txt.includes("couldn't find this account") ||
                    txt.includes("couldn’t find this account") ||
                    txt.includes("couldnt find this account") ||
                    txt.includes("account not found")
                );
            }"""
        ))
    except Exception:
        return False


def is_captcha_present(page) -> bool:
    """Detect TikTok CAPTCHA using URL and known DOM markers."""
    try:
        current_url = (page.url or "").lower()
        if "captcha" in current_url or "verify" in current_url:
            _logger.info(
                "[captcha][debug] url=%s found=True markers={container:0, slide:0, switch:0, tiktok_verify:0}",
                page.url,
            )
            return True

        marker_flags = {
            "container": 1 if page.query_selector(".captcha-verify-container") else 0,
            "slide": 1 if page.query_selector("#captcha_slide_button") else 0,
            "switch": 1 if page.query_selector("#captcha_switch_button") else 0,
            "close": 1 if page.query_selector("#captcha_close_button") else 0,
            "drag_icon": 1 if page.query_selector(".secsdk-captcha-drag-icon") else 0,
            "tiktok_verify": 1 if page.query_selector("#tiktok-verify-ele") else 0,
        }
        marker_match = any(marker_flags.values())
        if marker_match:
            _logger.info(
                "[captcha][debug] url=%s found=True markers={container:%s, slide:%s, switch:%s, tiktok_verify:%s}",
                page.url,
                marker_flags["container"],
                marker_flags["slide"],
                marker_flags["switch"],
                marker_flags["tiktok_verify"],
            )
            return True

        _logger.info(
            "[captcha][debug] url=%s found=%s markers={container:%s, slide:%s, switch:%s, tiktok_verify:%s}",
            page.url,
            False,
            marker_flags["container"],
            marker_flags["slide"],
            marker_flags["switch"],
            marker_flags["tiktok_verify"],
        )
        return False
    except Exception:
        return False


class _CaptchaReloadRequested(Exception):
    """Internal control flow: restart current profile scrape after prolonged CAPTCHA wait."""


def _emit_soft_beep() -> None:
    """Emit a soft local alert sound (best effort)."""
    # Windows-native beep first (more reliable than terminal bell in IDE terminals).
    try:
        import winsound  # Windows stdlib
        try:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            winsound.Beep(750, 180)
        return
    except Exception:
        pass
    # Fallback: terminal bell
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def wait_while_captcha(page, check_interval_ms: int = 2000, profile_key: str = "") -> None:
    """Pause while CAPTCHA is present, then resume."""
    global _captcha_beeped_profiles
    page = ensure_live_page(page)
    captcha_wait_started_at = None
    while True:
        try:
            if not is_captcha_present(page):
                return

            if captcha_wait_started_at is None:
                captcha_wait_started_at = time.time()
                if profile_key and profile_key not in _captcha_beeped_profiles:
                    _emit_soft_beep()
                    _captcha_beeped_profiles.add(profile_key)
            _logger.info("[captcha] CAPTCHA detected — waiting for manual resolution...")
            while True:
                if not is_captcha_present(page):
                    _logger.info("[captcha] CAPTCHA cleared — resuming.")
                    return
                if captcha_wait_started_at is not None and (time.time() - captcha_wait_started_at) >= 60:
                    _logger.info("[captcha] CAPTCHA still present after 60s — restarting this profile from the beginning...")
                    raise _CaptchaReloadRequested("captcha_wait_cycle_timeout")
                try:
                    page.wait_for_timeout(check_interval_ms)
                except Exception:
                    time.sleep(max(0.1, check_interval_ms / 1000.0))
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=1000)
                except Exception:
                    pass
        except _CaptchaReloadRequested:
            raise
        except Exception:
            try:
                page.wait_for_timeout(500)
            except Exception:
                time.sleep(0.5)
            page = ensure_live_page(page)


def scrape_profile(profile_url: str, days: int = 10, page=None, username_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    profile_url = normalize_profile_url(profile_url) or normalize_profile_link(profile_url)
    username_from_url = extract_username_from_profile_url(profile_url) if profile_url else None
    username = (username_hint or username_from_url or "unknown").lstrip("@")
    if not profile_url:
        return results

    _logger.info(f"   Starting browser automation for @{username} | url={profile_url}")
    target_url = profile_url

    try:
        page = ensure_live_page(page)
        _logger.info(f"   Loading profile page: {target_url}")
        try:
            page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as goto_e:
            if "Target page, context or browser has been closed" in str(goto_e):
                _logger.info("[browser] goto failed due to closed page — relaunching and retrying once...")
                _, _, page = launch_browser_with_profile()
                page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            else:
                raise
        page.wait_for_function(
            "document && document.body && document.body.innerText && document.body.innerText.length > 0",
            timeout=1500,
        )

        # Best-effort close cookie consent if present (no sleeps)
        for sel in [
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
            'button:has-text("Accept")',
            'button[aria-label*="Accept"]',
        ]:
            try:
                btn = page.query_selector(sel)
                if btn:
                    btn.click()
                    break
            except Exception:
                pass

        # Pause for manual CAPTCHA resolution before profile-state checks.
        try:
            wait_while_captcha(page, profile_key=target_url)
        except _CaptchaReloadRequested:
            _logger.info("[captcha] Restarting profile scrape from beginning after 60s CAPTCHA wait cycle.")
            return scrape_profile(profile_url, days=days, page=page, username_hint=username_hint)

        if is_account_not_found(page):
            _logger.info("[invalid-profile] Couldn't find this account")
            return results

        # Defensive re-check in case CAPTCHA appears after consent handling/state probes.
        try:
            wait_while_captcha(page, profile_key=target_url)
        except _CaptchaReloadRequested:
            _logger.info("[captcha] Restarting profile scrape from beginning after 60s CAPTCHA wait cycle.")
            return scrape_profile(profile_url, days=days, page=page, username_hint=username_hint)

        status = wait_for_profile_grid_and_extension(page, timeout_ms=12000, min_items=1, min_metas=1)
        if status.get("unknown"):
            # Sticky CAPTCHA guard: CAPTCHA can render shortly after an initial negative check.
            if is_captcha_present(page):
                _logger.info("[captcha] CAPTCHA appeared during readiness checks — pausing before retrying profile state.")
                try:
                    wait_while_captcha(page, profile_key=target_url)
                except _CaptchaReloadRequested:
                    _logger.info("[captcha] Restarting profile scrape from beginning after 60s CAPTCHA wait cycle.")
                    return scrape_profile(profile_url, days=days, page=page, username_hint=username_hint)
                status = wait_for_profile_grid_and_extension(page, timeout_ms=12000, min_items=1, min_metas=1)
        _logger.info(
            "   [profile-ready] has_items=%s has_metas=%s empty_or_private=%s",
            status.get("has_items", False),
            status.get("has_metas", False),
            status.get("empty_or_private", False),
        )

        if status.get("empty_or_private"):
            _logger.info("   [profile] empty/private (0 videos)")
            return []

        if status.get("has_items"):
            if status.get("has_metas"):
                results = ui_fallback_collect_recent(page, days=days, expected_username=username_from_url)
            else:
                _logger.info("   [profile] metas missing; waiting for extension metas...")
                retry_status = wait_for_profile_grid_and_extension(page, timeout_ms=30000, min_items=1, min_metas=1)
                _logger.info(
                    "   [profile-ready][retry] has_items=%s has_metas=%s empty_or_private=%s",
                    retry_status.get("has_items", False),
                    retry_status.get("has_metas", False),
                    retry_status.get("empty_or_private", False),
                )
                if retry_status.get("empty_or_private"):
                    return []
                if retry_status.get("has_items") and retry_status.get("has_metas"):
                    results = ui_fallback_collect_recent(page, days=days, expected_username=username_from_url)
                else:
                    counts = get_profile_grid_debug_counts(page)
                    _logger.info("   [profile-ready][meta-wait][debug] counts=%s", counts)
                    _logger.info("   [profile] metas still missing after wait; using basic fallback scraper")
                    results = ui_collect_basic(page, target_videos=15)
            _logger.info(f"   Collected {len(results)} videos from @{username}")
        else:
            # Unknown readiness state: capture lightweight diagnostics and surface as error
            try:
                ts = int(time.time())
                diag_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(diag_dir, exist_ok=True)
                shot = os.path.join(diag_dir, f"profile_ready_unknown_{username}_{ts}.png")
                htmlp = os.path.join(diag_dir, f"profile_ready_unknown_{username}_{ts}.html")
                page.screenshot(path=shot, full_page=False)
                with open(htmlp, "w", encoding="utf-8") as f:
                    f.write(page.content())
                _logger.info("   [profile-debug] saved diagnostics: %s, %s", shot, htmlp)
            except Exception:
                pass
            raise Exception("Profile readiness unknown: no items and not empty/private")

    except Exception as e:
        _logger.warning(f"   WARNING: Error collecting videos from @{username}: {e}")
        counts = get_profile_grid_debug_counts(page)
        _logger.info(
            "   [profile-debug] on error grid_items=%s anchors=%s metas=%s",
            counts.get("gridItems", 0),
            counts.get("anchors", 0),
            counts.get("metas", 0),
        )
        results = []

    # Deduplicate by video_id
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in results:
        vid = r.get("video_id") or ""
        if vid and vid not in uniq:
            uniq[vid] = r
    return list(uniq.values())


def calculate_profile_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for a profile's videos."""
    if not rows:
        return {
            "profile": "",
            "fifteenth_video_date": "",
            "latest_video_date": "",
            "median_views": 0,
            "videos_under_10k": 0,
            "videos_10k_to_100k": 0,
            "videos_100k_plus": 0,
            "total_videos": 0
        }
    
    # Don't sort by create_ts since most videos have create_ts = 0
    # Keep original order to get the correct 15th video
    sorted_rows = rows
    
    # Get the date from the last video (15th if available, otherwise the last one)
    fifteenth_video_date = ""
    if sorted_rows:
        # If we have 15+ videos, use the 15th video date
        if len(sorted_rows) >= 15:
            target_video = sorted_rows[14]  # 0-indexed, so 14 is the 15th
        else:
            # If less than 15 videos, use the last video date
            target_video = sorted_rows[-1]  # Last video
        
        if target_video["create_ts"] > 0:  # Only use valid timestamps
            fifteenth_video_date = datetime.fromtimestamp(target_video["create_ts"], tz=timezone.utc).strftime('%Y-%m-%d')
    
    # Get the date from the most recent video (first video, skipping pinned ones)
    latest_video_date = ""
    if sorted_rows:
        # The first video in the list is the most recent (already skipping pinned in collection)
        first_video = sorted_rows[0]
        if first_video["create_ts"] > 0:  # Only use valid timestamps
            latest_video_date = datetime.fromtimestamp(first_video["create_ts"], tz=timezone.utc).strftime('%Y-%m-%d')
    
    # Get view counts (filter out -1 values which indicate unknown views)
    view_counts = [row["views"] for row in sorted_rows if row["views"] > 0]
    
    # Calculate median
    median_views = 0
    if view_counts:
        sorted_views = sorted(view_counts)
        n = len(sorted_views)
        if n % 2 == 0:
            median_views = (sorted_views[n//2 - 1] + sorted_views[n//2]) / 2
        else:
            median_views = sorted_views[n//2]
    
    # Count videos in different ranges
    videos_under_10k = sum(1 for views in view_counts if views < 10000)
    videos_10k_to_100k = sum(1 for views in view_counts if 10000 <= views < 100000)
    videos_100k_plus = sum(1 for views in view_counts if views >= 100000)
    
    return {
        "profile": rows[0].get("profile", "") if rows else "",
        "fifteenth_video_date": fifteenth_video_date,
        "latest_video_date": latest_video_date,
        "median_views": int(median_views),
        "videos_under_10k": videos_under_10k,
        "videos_10k_to_100k": videos_10k_to_100k,
        "videos_100k_plus": videos_100k_plus,
        "total_videos": len(rows)
    }


def write_csv(rows: List[Dict[str, Any]], out_dir: str, label: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tiktok_videos_last_10_days.csv")
    
    # Group rows by profile
    profiles = {}
    for row in rows:
        profile = row.get("profile", "unknown")
        if profile not in profiles:
            profiles[profile] = []
        profiles[profile].append(row)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profile", 
            "fifteenth_video_date", 
            "median_views", 
            "videos_under_10k", 
            "videos_10k_to_100k", 
            "videos_100k_plus",
            "total_videos"
        ])
        
        for profile, profile_rows in profiles.items():
            stats = calculate_profile_stats(profile_rows)
            writer.writerow([
                stats["profile"],
                stats["fifteenth_video_date"],
                stats["median_views"],
                stats["videos_under_10k"],
                stats["videos_10k_to_100k"],
                stats["videos_100k_plus"],
                stats["total_videos"]
            ])
    
    return out_path


def run_scraping_job(page, manager_label: str = ""):
    """
    Scraping job that uses an existing browser page.
    """
    # Start tracking full run time
    run_start_time = time.time()
    
    ensure_dependencies()
    
    # Initialize logging system
    global _logger
    _logger = setup_logging()
    
    # Get Cairo time for the run
    cairo_time = get_cairo_time()
    
    # Log run separator and metadata
    _logger.info("=" * 100)
    _logger.info("=" * 100)
    _logger.info(f"NEW RUN STARTED - {cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    _logger.info("=" * 100)

    service = setup_google_sheets_service()
    if not service:
        log_and_print("ERROR: Could not setup Google Sheets service. Exiting.", "ERROR")
        flush_pending_sheet_writes("setup_google_sheets_service_failed")
        return

    # Canonical rebuild pass (normalize + dedupe + rewrite + cost formula) before scraping.
    rebuild_result = rebuild_manager_sheet_tabs(
        service,
        SPREADSHEET_ID,
        manager_label=(manager_label or SPREADSHEET_ID),
    )

    # Read entire sheet from Google Sheets (prevents race conditions)
    global _current_sheet_data
    sheet_data = read_entire_sheet()
    if isinstance(sheet_data, dict) and isinstance(rebuild_result, dict):
        row_map = rebuild_result.get("row_map", {}) or {}
        link_row_map = rebuild_result.get("username_link_row_map", {}) or {}
        sheet_data.setdefault("username_to_row", {}).update(row_map)
        sheet_data.setdefault("username_link_to_row", {}).update(link_row_map)
    _current_sheet_data = sheet_data or {}
    if not sheet_data:
        log_and_print("ERROR: Could not read Google Sheets. Exiting.", "ERROR")
        flush_pending_sheet_writes("read_entire_sheet_failed")
        return
    
    # Allow empty profiles_to_scrape if we're not scraping
    if not sheet_data.get('profiles_to_scrape') and (RUN_PROFILE_SCRAPING or RUN_NA_PROFILES):
        log_and_print("ERROR: No TikTok URLs found in Google Sheets. Exiting.", "ERROR")
        flush_pending_sheet_writes("no_profiles_to_scrape")
        return
    
    # Extract profiles to scrape from LINK (canonical URL input)
    PROFILES_TO_SCRAPE = sheet_data.get('profiles_to_scrape', [])
    PROFILE_URLS = [p.get("profile_url", "") for p in PROFILES_TO_SCRAPE if isinstance(p, dict)]
    
    # Get profiles that need NA values (empty/NA URLs)
    PROFILES_FOR_NA = sheet_data.get('profiles_for_na', [])
    
    # Calculate total profiles (scrape + NA)
    total_profiles = len(PROFILE_URLS) + len(PROFILES_FOR_NA)

    # Initialize analytics tracking
    analytics = {
        'total_profiles': total_profiles,
        'successful_profiles': 0,
        'failed_profiles': 0,
        'invalid_or_not_found_profiles': 0,
        'empty_profiles': 0,
        'na_url_profiles': 0,  # New category for profiles with empty/NA URLs
        'profile_times': [],
        'video_times': [],
        'total_videos_processed': 0
    }

    DAYS = 10  # Kept for compatibility but not used for filtering
    current_time = datetime.now(timezone.utc)
    
    log_and_print("=" * 80)
    log_and_print("TIKTOK SCRAPER STARTED")
    log_and_print("=" * 80)
    log_and_print(f"Current Date (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"Current Date (Cairo): {cairo_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log_and_print(f"Processing: First 15 non-pinned items (videos and photos) per profile")
    log_and_print(f"Profiles to scrape: {len(PROFILE_URLS)}")
    log_and_print(f"Profiles with empty/NA URLs: {len(PROFILES_FOR_NA)}")
    log_and_print(f"Total profiles to process: {total_profiles}")
    log_and_print("=" * 80)
    
    # Initialize variables used throughout the function
    all_rows: List[Dict[str, Any]] = []
    consecutive_failures = 0
    max_consecutive_failures = 3
    max_retry_attempts = 2
    
    # Import tqdm for progress bars
    from tqdm import tqdm
    
    # Feature toggle: Profile scraping
    if not RUN_PROFILE_SCRAPING:
        log_and_print("Profile scraping is disabled (RUN_PROFILE_SCRAPING = False)", "WARNING")
    else:
        # Use tqdm progress bar for scraping profiles
        scraping_pbar = tqdm(
            PROFILES_TO_SCRAPE,
            desc="Scraping Profiles",
            unit="profile",
            ncols=100,
            position=0,
            leave=True
        )
        
        for i, profile_entry in enumerate(scraping_pbar, 1):
            if isinstance(profile_entry, dict):
                url = str(profile_entry.get("profile_url") or "")
                username_normalized = str(profile_entry.get("username_normalized") or "").lower().lstrip("@")
                username = str(profile_entry.get("username") or username_normalized or extract_username_from_profile_url(url) or "unknown")
            else:
                # Backward-compatible fallback
                url, username_normalized = profile_entry
                username = username_normalized

            # Update progress bar description with current profile
            scraping_pbar.set_description(f"Scraping @{username}")
            
            # Log only (no print to terminal to avoid interrupting tqdm)
            _logger.info(f"[{i}/{len(PROFILE_URLS)}] Scraping profile: @{username}")
            _logger.info(f"   URL: {url}")
            _logger.info("-" * 60)
            
            # Start timing for this profile
            profile_start_time = time.time()
            
            # Retry mechanism for profile scraping
            profile_success = False
            items = []
            is_not_found = False
            
            for attempt in range(max_retry_attempts):
                try:
                    if attempt > 0:
                        _logger.info(f"   Retry attempt {attempt + 1}/{max_retry_attempts} for @{username}")
                        time.sleep(0.5)
                    
                    page = ensure_live_page(page)
                    items = scrape_profile(url, days=DAYS, page=page, username_hint=username)
                    profile_success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # All accounts go through the full retry process
                    _logger.info(f"   Attempt {attempt + 1} failed for @{username}: {e}")
                    if attempt == max_retry_attempts - 1:
                        _logger.info(f"   All {max_retry_attempts} attempts failed for @{username}. Skipping profile.")
                        consecutive_failures += 1
                        
                        # Check for consecutive failures
                        if consecutive_failures >= max_consecutive_failures:
                            _logger.critical(f"   CRITICAL: {consecutive_failures} consecutive profile failures reached.")
                            _logger.critical(f"   Stopping execution to preserve data integrity.")
                            _logger.critical(f"   Successfully processed {len(all_rows)} profiles before stopping.")
                            break
                    else:
                        time.sleep(0.5)
            
            # If we hit the consecutive failure limit, break out of the main loop
            if consecutive_failures >= max_consecutive_failures:
                flush_pending_sheet_writes(f"consecutive_failure_limit@{username}")
                break
            
            # Reset consecutive failures counter on success
            if profile_success:
                consecutive_failures = 0
        
            # Track profile time (record it before handling outcomes)
            profile_end_time = time.time()
            profile_duration = profile_end_time - profile_start_time
            
            if profile_success and items:
                _logger.info(f"SUCCESS: Scraped @{username}")
                _logger.info(f"   Total items found: {len(items)}")
                
                # Validate that views were actually loaded - check if all views are unknown
                videos_with_views = sum(1 for item in items if item.get("views", 0) > 0)
                all_views_unknown = videos_with_views == 0 and len(items) > 0
                
                if all_views_unknown:
                    _logger.error(f"   ERROR: All videos found but no views were loaded - treating as error")
                    _logger.error(f"   Videos found: {len(items)}, Videos with views: {videos_with_views}")
                    _logger.error(f"   This will trigger a retry if profile exists in sheets with previous median views")
                    
                    # Check if profile exists in sheets and has previous median views
                    username_normalized = username.lower().lstrip('@')
                    previous_median_views = sheet_data.get('username_to_prev_median', {}).get(username_normalized)
                    if previous_median_views is None:
                        previous_median_views = sheet_data.get('link_to_prev_median', {}).get(normalize_link_key(url))
                    
                    # If profile exists and had previous median views, retry
                    if previous_median_views is not None and previous_median_views > 0:
                        _logger.warning(f"   Profile exists in sheets with previous median views: {previous_median_views:,}")
                        _logger.warning(f"   Retrying profile scrape due to missing view data...")
                        
                        # Retry the profile scrape
                        retry_success = False
                        for retry_attempt in range(1):  # single fast retry
                            try:
                                page = ensure_live_page(page)
                                items = scrape_profile(url, days=DAYS, page=page, username_hint=username)
                                videos_with_views_retry = sum(1 for item in items if item.get("views", 0) > 0)
                                
                                if videos_with_views_retry > 0:
                                    _logger.info(f"   Retry successful: {videos_with_views_retry} videos now have views")
                                    retry_success = True
                                    break
                                else:
                                    _logger.info(f"   Retry {retry_attempt + 1} still has no views")
                            except Exception as retry_e:
                                _logger.error(f"   Retry attempt {retry_attempt + 1} failed: {retry_e}")
                        
                        if not retry_success:
                            _logger.error(f"   ERROR: All retries failed - views still not loaded for @{username}")
                            _logger.warning(f"   Skipping this profile - data may be incomplete")
                            analytics['failed_profiles'] += 1
                            continue
                
                # Track successful profile
                analytics['successful_profiles'] += 1
                analytics['profile_times'].append(profile_duration)
                analytics['total_videos_processed'] += len(items)
                
                # Track individual video processing time (profile time / number of videos)
                if len(items) > 0:
                    avg_video_time = profile_duration / len(items)
                    analytics['video_times'].extend([avg_video_time] * len(items))
                
                for j, item in enumerate(items, 1):
                    upload_date = datetime.fromtimestamp(item["create_ts"], tz=timezone.utc) if item["create_ts"] > 0 else None
                    views = item["views"]
                    content_id = item["video_id"]  # Keep same field name for compatibility
                    content_type = item.get("content_type", "video")
                    
                    _logger.info(f"   {content_type.title()} {j}: {content_id}")
                    if upload_date:
                        _logger.info(f"      Uploaded: {upload_date.strftime('%Y-%m-%d')}")
                    else:
                        _logger.info(f"      Uploaded: Unknown")
                    _logger.info(f"      Views: {views:,}" if views > 0 else f"      Views: Unknown")
                    _logger.info(f"      URL: {item['url']}")
                    
                    it_with_profile = {
                        "profile": f"@{username}",
                        **item,
                    }
                    all_rows.append(it_with_profile)
                
                # Display timing for this profile
                _logger.info(f"   Added {len(items)} items to results")
                _logger.info(f"     Profile scraping time: {profile_duration:.2f} seconds ({profile_duration/60:.1f} minutes)")
                
                # Calculate analytics for this profile and write to Google Sheets
                try:
                    profile_stats = calculate_profile_stats(items)
                    username_normalized = username.lower().lstrip('@')
                    previous_median_views = sheet_data.get('username_to_prev_median', {}).get(username_normalized)
                    if previous_median_views is None:
                        previous_median_views = sheet_data.get('link_to_prev_median', {}).get(normalize_link_key(url))
                    
                    analytics_data = {
                        'username': f"@{username}",
                        'profile_url': url,
                        'median_views': profile_stats['median_views'],
                        'videos_under_10k': profile_stats['videos_under_10k'],
                        'videos_10k_to_100k': profile_stats['videos_10k_to_100k'],
                        'videos_100k_plus': profile_stats['videos_100k_plus'],
                        'fifteenth_video_date': profile_stats['fifteenth_video_date'],
                        'latest_video_date': profile_stats['latest_video_date']
                    }
                    
                    # Check if median views is 0 but previous wasn't - this suggests an error
                    if profile_stats['median_views'] == 0 and previous_median_views is not None and previous_median_views > 0:
                        _logger.warning(f"   WARNING: Median views is 0 but previous was {previous_median_views:,} - possible error")
                        _logger.warning(f"   Retrying profile scrape to verify data...")
                        
                        # Retry the profile scrape
                        retry_success = False
                        for retry_attempt in range(1):  # single fast retry
                            try:
                                page = ensure_live_page(page)
                                retry_items = scrape_profile(url, days=DAYS, page=page)
                                retry_stats = calculate_profile_stats(retry_items)
                                
                                if retry_stats['median_views'] > 0:
                                    _logger.info(f"   Retry successful: median views {retry_stats['median_views']:,}")
                                    items = retry_items
                                    profile_stats = retry_stats
                                    analytics_data = {
                                        'username': f"@{username}",
                                        'profile_url': url,
                                        'median_views': profile_stats['median_views'],
                                        'videos_under_10k': profile_stats['videos_under_10k'],
                                        'videos_10k_to_100k': profile_stats['videos_10k_to_100k'],
                                        'videos_100k_plus': profile_stats['videos_100k_plus'],
                                        'fifteenth_video_date': profile_stats['fifteenth_video_date'],
                                        'latest_video_date': profile_stats['latest_video_date']
                                    }
                                    # Update all_rows with the retried items (remove old items for this profile and add new ones)
                                    all_rows = [row for row in all_rows if row.get("profile", "").lower().lstrip('@') != username_normalized]
                                    for item in retry_items:
                                        it_with_profile = {
                                            "profile": f"@{username}",
                                            **item,
                                        }
                                        all_rows.append(it_with_profile)
                                    _logger.info(f"   Updated all_rows with retried data for @{username}")
                                    retry_success = True
                                    break
                                else:
                                    _logger.info(f"   Retry {retry_attempt + 1} still has median views = 0")
                            except Exception as retry_e:
                                _logger.error(f"   Retry attempt {retry_attempt + 1} failed: {retry_e}")
                        
                        if not retry_success:
                            _logger.warning(f"   WARNING: All retries failed - median views still 0 for @{username}")
                            _logger.warning(f"   Proceeding with current data (may be incomplete)")
                    
                    # Write analytics to Google Sheets with retry
                    sheets_success = False
                    for sheets_attempt in range(5):  # 5 attempts for Google Sheets write
                        try:
                            if write_analytics_to_sheet(analytics_data):
                                _logger.info(f"   Analytics written to Google Sheets for @{username}")
                                sheets_success = True
                                break
                            else:
                                if sheets_attempt < 4:
                                    _logger.warning(f"   Retry {sheets_attempt + 2}/5 for Google Sheets write for @{username}")
                                    time.sleep(2)
                        except Exception as sheets_e:
                            _logger.error(f"   Google Sheets write attempt {sheets_attempt + 1} failed: {sheets_e}")
                            if sheets_attempt < 4:
                                time.sleep(2)
                    
                    if not sheets_success:
                        _logger.error(f"   ERROR: Failed to write analytics to Google Sheets for @{username} after 5 attempts")
                        _logger.warning(f"   Data preserved in CSV output")
                        
                except Exception as analytics_e:
                    _logger.error(f"   ERROR calculating analytics for @{username}: {analytics_e}")
                    _logger.warning(f"   Profile data preserved, continuing...")
            
            elif not profile_success and not is_not_found:
                # Calculate timing for failed profiles (excluding not found profiles)
                _logger.error(f"   Profile failed after {max_retry_attempts} attempts")
                _logger.info(f"     Profile scraping time: {profile_duration:.2f} seconds ({profile_duration/60:.1f} minutes)")
                
                # Track failed profile
                analytics['failed_profiles'] += 1
                
                # Write NA values to Google Sheets for failed profiles with retry
                na_success = False
                for na_attempt in range(5):  # 5 attempts for NA values write
                    try:
                        if write_na_to_sheet(f"@{username}", profile_url=url):
                            _logger.info(f"   NA values written to Google Sheets for @{username}")
                            na_success = True
                            break
                        else:
                            if na_attempt < 4:
                                _logger.warning(f"   Retry {na_attempt + 2}/5 for NA values write for @{username}")
                                time.sleep(2)
                    except Exception as na_e:
                        _logger.error(f"   NA values write attempt {na_attempt + 1} failed: {na_e}")
                        if na_attempt < 4:
                            time.sleep(2)
                
                if not na_success:
                    _logger.error(f"   ERROR: Failed to write NA values to Google Sheets for @{username} after 5 attempts")
                
                _logger.info(f"   Continuing with next profile...")

            elif profile_success and not items:
                # Profile loaded but no videos found
                _logger.warning(f"   WARNING: No videos found for @{username}")
                _logger.info(f"     Profile scraping time: {profile_duration:.2f} seconds ({profile_duration/60:.1f} minutes)")
                
                # Track empty profile
                analytics['empty_profiles'] += 1
                
                # Write NA values to Google Sheets for empty profiles with retry
                na_success = False
                for na_attempt in range(5):  # 5 attempts for NA values write
                    try:
                        if write_na_to_sheet(f"@{username}", profile_url=url):
                            _logger.info(f"   NA values written to Google Sheets for @{username}")
                            na_success = True
                            break
                        else:
                            if na_attempt < 4:
                                _logger.warning(f"   Retry {na_attempt + 2}/5 for NA values write for @{username}")
                                time.sleep(2)
                    except Exception as na_e:
                        _logger.error(f"   NA values write attempt {na_attempt + 1} failed: {na_e}")
                        if na_attempt < 4:
                            time.sleep(2)
                
                if not na_success:
                    _logger.error(f"   ERROR: Failed to write NA values to Google Sheets for @{username} after 5 attempts")
                
                _logger.info(f"   Continuing with next profile...")

            # Periodic batched flush to reduce API round-trips and bound in-memory queue.
            if i % max(1, SHEETS_WRITE_FLUSH_EVERY_PROFILES) == 0:
                flush_pending_sheet_writes(f"profile_batch_{i}")
    
    # Don't close browser - keep it open for next run
    
    # Feature toggle: NA profiles
    if not RUN_NA_PROFILES:
        log_and_print("\nNA profiles processing is disabled (RUN_NA_PROFILES = False)", "WARNING")
    else:
        # Process profiles with empty/NA URLs (write NA values)
        if PROFILES_FOR_NA:
            # Use tqdm progress bar for NA profiles
            na_pbar = tqdm(
                PROFILES_FOR_NA,
                desc="Writing NA Values",
                unit="profile",
                ncols=100,
                position=0,
                leave=True
            )
            
            for na_entry in na_pbar:
                if isinstance(na_entry, dict):
                    username_normalized = str(na_entry.get("username_normalized") or "").lower().lstrip("@")
                    username_display = str(na_entry.get("username") or username_normalized or "unknown")
                    profile_url = str(na_entry.get("profile_url") or "")
                else:
                    username_normalized = str(na_entry).lower().lstrip("@")
                    username_display = username_normalized or "unknown"
                    profile_url = ""

                na_pbar.set_description(f"NA: @{username_display}")
                
                _logger.info(f"Writing NA values for @{username_display} (empty/NA URL)")
                na_success = False
                for na_attempt in range(5):  # 5 attempts for NA values write
                    try:
                        if write_na_to_sheet(f"@{username_display}", profile_url=profile_url):
                            _logger.info(f"   NA values written to Google Sheets for @{username_display}")
                            na_success = True
                            analytics['na_url_profiles'] += 1
                            break
                        else:
                            if na_attempt < 4:
                                _logger.warning(f"   Retry {na_attempt + 2}/5 for NA values write for @{username_display}")
                                time.sleep(2)
                    except Exception as na_e:
                        _logger.error(f"   NA values write attempt {na_attempt + 1} failed: {na_e}")
                        if na_attempt < 4:
                            time.sleep(2)
                
                if not na_success:
                    _logger.error(f"   ERROR: Failed to write NA values to Google Sheets for @{username_display} after 5 attempts")

    # Only show scraping summary if we actually scraped
    if RUN_PROFILE_SCRAPING and len(PROFILE_URLS) > 0:
        log_and_print("\n" + "=" * 80)
        log_and_print("SCRAPING SUMMARY")
        log_and_print("=" * 80)
        
        if not all_rows:
            log_and_print("No recent items found.", "WARNING")
        else:
            log_and_print(f"Total items collected: {len(all_rows)}")
            
            # Show the target video date (15th if available, otherwise last)
            if all_rows:
                # Group by profile to get target video date
                profiles = {}
                for row in all_rows:
                    profile = row.get("profile", "unknown")
                    if profile not in profiles:
                        profiles[profile] = []
                    profiles[profile].append(row)
                
                for profile, profile_rows in profiles.items():
                    if len(profile_rows) >= 15:
                        # Don't sort - keep original order to get the correct 15th video
                        target_video = profile_rows[14]  # 0-indexed, so 14 is the 15th
                        if target_video["create_ts"] > 0:
                            target_date = datetime.fromtimestamp(target_video["create_ts"], tz=timezone.utc)
                            log_and_print(f"15th video date for {profile}: {target_date.strftime('%Y-%m-%d')}")
                        else:
                            log_and_print(f"15th video date for {profile}: Unknown")
                    else:
                        # Use last video date if less than 15 videos
                        last_video = profile_rows[-1]  # Last video
                        if last_video["create_ts"] > 0:
                            last_date = datetime.fromtimestamp(last_video["create_ts"], tz=timezone.utc)
                            log_and_print(f"Last video date for {profile}: {last_date.strftime('%Y-%m-%d')} ({len(profile_rows)} videos)")
                        else:
                            log_and_print(f"Last video date for {profile}: Unknown ({len(profile_rows)} videos)")
                
                # Show view statistics
                view_counts = [row["views"] for row in all_rows if row["views"] > 0]
                if view_counts:
                    total_views = sum(view_counts)
                    avg_views = total_views / len(view_counts)
                    max_views = max(view_counts)
                    log_and_print(f"View statistics:")
                    log_and_print(f"   Total views: {total_views:,}")
                    log_and_print(f"   Average views: {avg_views:,.0f}")
                    log_and_print(f"   Highest views: {max_views:,}")
            
            # Group label from first username or generic
            if PROFILE_URLS:
                first_label = extract_username_from_profile_url(PROFILE_URLS[0]) or "tiktok"
            else:
                first_label = "tiktok"
            out_file = write_csv(all_rows, out_dir=os.getcwd(), label=first_label)
            log_and_print(f"Saved CSV: {out_file}")
            
            # Display summary statistics for each profile
            log_and_print("\n" + "=" * 80)
            log_and_print("PROFILE SUMMARY STATISTICS")
            log_and_print("=" * 80)
            
            # Group rows by profile for summary display
            profiles = {}
            for row in all_rows:
                profile = row.get("profile", "unknown")
                if profile not in profiles:
                    profiles[profile] = []
                profiles[profile].append(row)
            
            for profile, profile_rows in profiles.items():
                stats = calculate_profile_stats(profile_rows)
                log_and_print(f"\nProfile: {stats['profile']}")
                if len(profile_rows) >= 15:
                    log_and_print(f"   15th video date: {stats['fifteenth_video_date']}")
                else:
                    log_and_print(f"   Last video date: {stats['fifteenth_video_date']} ({len(profile_rows)} videos)")
                log_and_print(f"   Median views: {stats['median_views']:,}")
                log_and_print(f"   View distribution:")
                log_and_print(f"      Under 10k views: {stats['videos_under_10k']} videos")
                log_and_print(f"      10k-100k views: {stats['videos_10k_to_100k']} videos")
                log_and_print(f"      100k+ views: {stats['videos_100k_plus']} videos")
                log_and_print(f"   Total videos processed: {stats['total_videos']}")
        
        log_and_print("=" * 80)
        log_and_print("SCRAPING COMPLETED")
        if consecutive_failures >= max_consecutive_failures:
            log_and_print(f"Execution stopped after {consecutive_failures} consecutive failures", "WARNING")
            log_and_print("Data integrity preserved - only successfully processed profiles included")
        else:
            log_and_print("All profiles processed successfully")
        log_and_print("=" * 80)
    
    # Check if run was successful
    run_successful = consecutive_failures < max_consecutive_failures
    
    # Feature toggle: Update Google Sheets headers with current date AFTER successful run
    if run_successful and RUN_UPDATE_HEADERS:
        log_and_print("\n" + "=" * 80)
        log_and_print("UPDATING GOOGLE SHEETS HEADERS")
        log_and_print("=" * 80)
        log_and_print("Updating Google Sheets headers with current date...")
        if update_sheet_headers():
            log_and_print("Sheet headers updated successfully")
        else:
            log_and_print("WARNING: Failed to update sheet headers, continuing anyway...", "WARNING")
        log_and_print("=" * 80)
    elif run_successful and not RUN_UPDATE_HEADERS:
        log_and_print("\nSheet headers update is disabled (RUN_UPDATE_HEADERS = False)", "WARNING")
        
    # Feature toggle: Classify and move accounts to appropriate tabs after header update
    if run_successful and RUN_ACCOUNT_CLASSIFICATION:
        classify_and_move_accounts()
    elif run_successful and not RUN_ACCOUNT_CLASSIFICATION:
        log_and_print("\nAccount classification is disabled (RUN_ACCOUNT_CLASSIFICATION = False)", "WARNING")
    
    # Calculate time for analytics display
    run_end_time = time.time()
    total_run_time = run_end_time - run_start_time
    end_cairo_time = get_cairo_time()
    
    # Display comprehensive analytics at the END (after bad accounts processing)
    log_and_print("\n" + "=" * 80)
    log_and_print("RUN ANALYTICS")
    log_and_print("=" * 80)
    
    # Profile statistics
    log_and_print("\n📊 PROFILE STATISTICS:")
    log_and_print(f"   Total profiles in sheet: {analytics['total_profiles']}")
    log_and_print(f"   Successfully retrieved: {analytics['successful_profiles']} ({analytics['successful_profiles']/analytics['total_profiles']*100:.1f}%)")
    log_and_print(f"   Failed to retrieve: {analytics['failed_profiles']} ({analytics['failed_profiles']/analytics['total_profiles']*100:.1f}%)")
    log_and_print(f"   Invalid/Not found: {analytics['invalid_or_not_found_profiles']} ({analytics['invalid_or_not_found_profiles']/analytics['total_profiles']*100:.1f}%)")
    log_and_print(f"   Empty profiles (no videos): {analytics['empty_profiles']} ({analytics['empty_profiles']/analytics['total_profiles']*100:.1f}%)")
    log_and_print(f"   Empty/NA URLs (NA values): {analytics['na_url_profiles']} ({analytics['na_url_profiles']/analytics['total_profiles']*100:.1f}%)")
    
    # Time statistics
    log_and_print("\n⏱️  TIME STATISTICS:")
    hours = int(total_run_time // 3600)
    minutes = int((total_run_time % 3600) // 60)
    seconds = int(total_run_time % 60)
    log_and_print(f"   Total run time: {hours}h {minutes}m {seconds}s ({total_run_time:.2f} seconds)")
    
    if analytics['profile_times']:
        avg_profile_time = sum(analytics['profile_times']) / len(analytics['profile_times'])
        min_profile_time = min(analytics['profile_times'])
        max_profile_time = max(analytics['profile_times'])
        log_and_print(f"   Average profile time: {avg_profile_time:.2f} seconds ({avg_profile_time/60:.1f} minutes)")
        log_and_print(f"   Fastest profile: {min_profile_time:.2f} seconds ({min_profile_time/60:.1f} minutes)")
        log_and_print(f"   Slowest profile: {max_profile_time:.2f} seconds ({max_profile_time/60:.1f} minutes)")
    else:
        log_and_print(f"   Average profile time: N/A (no successful profiles)")
    
    if analytics['video_times']:
        avg_video_time = sum(analytics['video_times']) / len(analytics['video_times'])
        log_and_print(f"   Average video time: {avg_video_time:.2f} seconds")
    else:
        log_and_print(f"   Average video time: N/A (no videos processed)")
    
    # Video statistics
    log_and_print("\n🎬 VIDEO STATISTICS:")
    log_and_print(f"   Total videos processed: {analytics['total_videos_processed']}")
    if analytics['successful_profiles'] > 0:
        avg_videos_per_profile = analytics['total_videos_processed'] / analytics['successful_profiles']
        log_and_print(f"   Average videos per profile: {avg_videos_per_profile:.1f}")
    else:
        log_and_print(f"   Average videos per profile: N/A")
    
    # Processing rate
    log_and_print("\n📈 PROCESSING RATE:")
    if total_run_time > 0:
        profiles_per_hour = (analytics['successful_profiles'] / total_run_time) * 3600
        profiles_per_minute = analytics['successful_profiles'] / (total_run_time / 60)
        log_and_print(f"   Profiles per hour: {profiles_per_hour:.1f}")
        log_and_print(f"   Profiles per minute: {profiles_per_minute:.2f}")
        
        if analytics['total_videos_processed'] > 0:
            videos_per_minute = analytics['total_videos_processed'] / (total_run_time / 60)
            log_and_print(f"   Videos per minute: {videos_per_minute:.2f}")
    
    log_and_print("\n" + "=" * 80)
    log_and_print(f"RUN COMPLETED - {end_cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    log_and_print("=" * 80)
    
    # Final log separator
    _logger.info("=" * 100)
    _logger.info("")
    
    # Return success status
    flush_pending_sheet_writes("run_scraping_job_end")
    return run_successful


def apply_cost_formula_to_tabs(spreadsheet_id, tabs_to_process):
    """
    Apply the Cost for 100k Views formula to all specified tabs.
    Formula: =(100000*G{row})/J{row}
    Where G is TikTok Price and J is Median Views.
    """
    try:
        service = setup_google_sheets_service()
        if not service:
            log_and_print("ERROR: Could not setup Google Sheets service", "ERROR")
            return False
        
        log_and_print("\nApplying Cost for 100k Views formula to all tabs...")
        
        for tab_name in tabs_to_process:
            try:
                # Read all data from the tab to find rows with data
                result = sheets_get_values(
                    service,
                    spreadsheet_id,
                    f"{tab_name}!A:U"
                )
                
                values = result.get('values', [])
                if len(values) <= HEADER_ROW:  # Only headers, no data
                    continue
                
                # Prepare formulas for all data rows (starting from row 3)
                formulas = []
                data_row_count = 0
                for i, row in enumerate(values[HEADER_ROW:], start=DATA_START_ROW):
                    # Skip if USERNAME or Link is empty
                    username = row[COLUMNS['USERNAME']].strip() if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                    link = row[COLUMNS['LINK']].strip() if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                    
                    if not username and not link:
                        continue
                    
                    # Add formula: =(100000*G{row})/J{row}
                    formula = f"=(100000*G{i})/J{i}"
                    formulas.append([formula])
                    data_row_count += 1
                
                if formulas:
                    # Write formulas to column I (Cost for 100k Views)
                    formula_range = f"{tab_name}!I{DATA_START_ROW}:I{DATA_START_ROW + data_row_count - 1}"
                    sheets_update_values(
                        service,
                        spreadsheet_id,
                        formula_range,
                        formulas,
                        value_input_option='USER_ENTERED'
                    )
                    log_and_print(f"  Applied formula to {data_row_count} rows in '{tab_name}'")
                
            except Exception as e:
                log_and_print(f"  WARNING: Could not apply formula to '{tab_name}': {e}", "WARNING")
                continue
        
        return True
        
    except Exception as e:
        log_and_print(f"ERROR applying cost formula: {e}", "ERROR")
        return False


def classify_and_move_accounts():
    """
    Classify and move accounts to appropriate tabs based on criteria.
    Skips rows with empty TikTok USERNAME or Link columns.
    
    Manual Approve (BYPASSES ALL CHECKS):
    - If Manual Approve checkbox (Column F) is checked → AUTOMATICALLY RELIABLE
    - Bypasses ALL checks: Bad, Good, and Reliable criteria
    
    Bad Accounts (ANY ONE condition = bad):
    - Latest Video Date (Column P) NOT within 15 days from current day → BAD
      (Only if TikTok Price exists; if date is > 15 days but TikTok Price is missing → moved to Initial Reachout instead)
    - Cost for 100k Views (Column I) > 30 (if numeric) → BAD
      (No fallback; Perf Based Price column is display-only)
    - Median Views (Column J) is 0, has error values (#DIV/0!, #VALUE!, etc.), or text like NA → BAD
      Only completely empty cells are acceptable and won't mark as bad
    
    Good Accounts (MUST PASS ALL):
    - NOT bad (passes all bad account checks, including date within 15 days)
    - Cost for 100k Views (Column I) <= 30
    - TikTok Price (Column G) is not empty and is a valid number
    
    Reliable Accounts (MUST PASS ALL OR Manual Approve):
    - Passes good account checks AND # of Times Booked (Column D) is integer > 0
    - OR Manual Approve (Column F) checkbox is checked (bypasses ALL checks - Bad, Good, and Reliable criteria)
    
    Move Back to Initial Reachout:
    - Accounts in Good, Reliable, or Bad Accounts tabs that are NOT bad but don't meet Good criteria
      (e.g., missing TikTok Price, Cost > 30)
    - Accounts with date > 15 days old but TikTok Price is missing
      (These are NOT marked as bad due to date, but moved to Initial Reachout)
    
    Processes all enabled tabs and moves accounts to appropriate destinations.
    """
    service = setup_google_sheets_service()
    if not service:
        log_and_print("ERROR: Could not setup Google Sheets service for account classification", "ERROR")
        return False
    
    spreadsheet_id = SPREADSHEET_ID
    
    try:
        log_and_print("\n" + "=" * 80)
        log_and_print("CLASSIFYING AND MOVING ACCOUNTS")
        log_and_print("=" * 80)
        
        # Determine which tabs to process based on toggles
        tabs_to_process = []
        if PROCESS_INITIAL_REACHOUT:
            tabs_to_process.append(TAB_NAMES['MAIN'])
        if PROCESS_BAD_ACCOUNTS:
            tabs_to_process.append(TAB_NAMES['BAD'])
        if PROCESS_GOOD_ACCOUNTS:
            tabs_to_process.append(TAB_NAMES['GOOD'])
        if PROCESS_RELIABLE_ACCOUNTS:
            tabs_to_process.append(TAB_NAMES['RELIABLE'])
        
        if not tabs_to_process:
            log_and_print("No tabs enabled for processing", "WARNING")
            return False
        
        log_and_print(f"Processing {len(tabs_to_process)} tab(s): {', '.join(tabs_to_process)}")
        
        # CRITICAL: Apply Cost for 100k Views formula BEFORE classification
        log_and_print("\n" + "=" * 80)
        log_and_print("APPLYING COST FORMULA (classification depends on this)")
        log_and_print("=" * 80)
        apply_cost_formula_to_tabs(spreadsheet_id, tabs_to_process)
        
        log_and_print("\n" + "=" * 80)
        log_and_print("STARTING CLASSIFICATION")
        log_and_print("=" * 80)
        
        # Get current date for comparison
        current_date = datetime.now(timezone.utc)
        log_and_print(f"Current date for comparison: {current_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log_and_print(f"Checking for dates within 15 days (after {(current_date - timedelta(days=15)).strftime('%Y-%m-%d')})")
        
        # Store classified accounts from all tabs
        all_bad_accounts = []      # List of (source_tab, row_index, row_data, reasons) tuples
        all_good_accounts = []     # List of (source_tab, row_index, row_data) tuples
        all_reliable_accounts = [] # List of (source_tab, row_index, row_data) tuples
        accounts_to_initial = []   # List of (source_tab, row_index, row_data) tuples - accounts that don't meet Good criteria
        all_empty_rows = []        # List of (source_tab, row_index) tuples
        
        # Process each enabled tab
        for tab_name in tabs_to_process:
            log_and_print(f"\nProcessing tab: {tab_name}")
            
            # Read entire sheet including all columns we need (A through W)
            range_name = f"{tab_name}!A:U"
            result = sheets_get_values(
                service,
                spreadsheet_id,
                range_name,
                value_render_option='UNFORMATTED_VALUE'
            )
            
            values = result.get('values', [])
            if not values or len(values) < HEADER_ROW:  # No data or only header
                log_and_print(f"  No data found in tab '{tab_name}'", "WARNING")
                continue
            
            header_row = values[HEADER_ROW - 1]  # Header is on row 2 (index 1)
            data_rows = values[HEADER_ROW:]       # Data starts from row 3
            
            # Classify each row in this tab
            for i, row in enumerate(data_rows, start=DATA_START_ROW):  # Start from row 3
                if not row or len(row) == 0:
                    # Empty row - mark for deletion
                    all_empty_rows.append((tab_name, i))
                    continue
                
                # Check if row is completely empty (no data in any column)
                is_completely_empty = True
                for cell in row:
                    if cell and str(cell).strip():
                        is_completely_empty = False
                        break
                
                # Completely empty rows get deleted from source tab
                if is_completely_empty:
                    all_empty_rows.append((tab_name, i))
                    continue
                
                # Skip rows with empty TikTok USERNAME or Link - treat as if they don't exist
                username = row[COLUMNS['USERNAME']].strip() if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                link = row[COLUMNS['LINK']].strip() if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                
                if not username or not link or link.upper() in ['NA', 'N/A', 'NULL', 'EMPTY']:
                    # Skip this row - missing username or link
                    _logger.debug(f"   Row {i}: Skipping (empty username or link)")
                    continue
                
                # Row has data, now classify it
                # FIRST: Check if Manual Approve checkbox is checked (Column F, index 5)
                # If checked, automatically goes to Reliable tab - BYPASSES ALL CHECKS (Bad, Good, Reliable)
                is_manually_approved = False
                if len(row) > COLUMNS['MANUAL_APPROVE'] and row[COLUMNS['MANUAL_APPROVE']] is not None:
                    manual_approve_value = row[COLUMNS['MANUAL_APPROVE']]
                    # Checkbox values in Google Sheets can be TRUE, "TRUE", True, or boolean True
                    if isinstance(manual_approve_value, bool):
                        is_manually_approved = manual_approve_value
                    elif isinstance(manual_approve_value, str):
                        is_manually_approved = manual_approve_value.upper() in ['TRUE', '1', 'YES', 'CHECKED']
                    elif isinstance(manual_approve_value, (int, float)):
                        is_manually_approved = bool(manual_approve_value)
                
                # If Manual Approve is checked, skip ALL checks (Bad, Good, Reliable) and mark as Reliable
                if is_manually_approved:
                    all_reliable_accounts.append((tab_name, i, row))
                    _logger.info(f"   Row {i} (@{username}): Classified as RELIABLE (Manual Approve checked - BYPASSES ALL CHECKS)")
                    continue  # Skip ALL classification logic - Bad, Good, and Reliable checks
                
                # Manual Approve is not checked, proceed with normal classification
                # First check if it's BAD (any one condition = bad)
                bad_reasons = []
                
                # Bad Check 1: Median Views (Column J, index 9) is 0, has error values, or has text like NA
                # Only completely empty cells are acceptable
                is_bad_median = False
                if len(row) > COLUMNS['MEDIAN_VIEWS'] and row[COLUMNS['MEDIAN_VIEWS']] is not None:
                    median_str = str(row[COLUMNS['MEDIAN_VIEWS']]).strip()
                    # If the cell has any content, check it
                    if median_str:  # Not empty string
                        # Check for Excel error values
                        if median_str.startswith('#'):  # #VALUE!, #DIV/0!, #N/A, #REF!, #NAME?, etc.
                            bad_reasons.append(f"Median Views has error: {median_str}")
                            is_bad_median = True
                        # Check for text values like NA
                        elif median_str.upper() in ['NA', 'N/A', 'NULL', 'EMPTY']:
                            bad_reasons.append(f"Median Views is '{median_str}'")
                            is_bad_median = True
                        else:
                            # Try to parse as number
                            try:
                                median_value = None
                                if isinstance(row[COLUMNS['MEDIAN_VIEWS']], (int, float)):
                                    median_value = float(row[COLUMNS['MEDIAN_VIEWS']])
                                else:
                                    median_cleaned = median_str.replace(',', '').strip()
                                    median_value = float(median_cleaned)
                                
                                # Check if median is 0
                                if median_value == 0:
                                    bad_reasons.append(f"Median Views is 0")
                                    is_bad_median = True
                            except (ValueError, TypeError):
                                # Not a number - this is an error
                                bad_reasons.append(f"Median Views is not a number: '{median_str}'")
                                is_bad_median = True
                # If column is missing or completely empty, that's acceptable - don't mark as bad
                
                # Bad Check 2: Cost for 100k Views (Column I, index 8) > 30 (if it's a number)
                # (Perf Based Price column is display-only, no fallback used)
                is_bad_cost = False
                cost_value = None
                cost_source = None  # Track which column was used: 'cost_100k' only
                cost_100k_valid = False  # Track if Cost for 100k Views was successfully parsed (even if 0)
                
                # Get cost from Cost for 100k Views only
                if len(row) > COLUMNS['COST_100K'] and row[COLUMNS['COST_100K']] is not None:
                    cost_str = str(row[COLUMNS['COST_100K']]).strip()
                    # Only check if it's a valid number (errors/NA skip cost check)
                    if not cost_str.startswith('#') and cost_str.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                        try:
                            # Handle both numeric values (from UNFORMATTED_VALUE) and string values
                            if isinstance(row[COLUMNS['COST_100K']], (int, float)):
                                temp_cost = float(row[COLUMNS['COST_100K']])
                            else:
                                # String value - parse it
                                cost_str_cleaned = cost_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').strip()
                                # Handle negative numbers in accounting format
                                if cost_str_cleaned.startswith('(') and cost_str_cleaned.endswith(')'):
                                    cost_str_cleaned = '-' + cost_str_cleaned[1:-1]
                                temp_cost = float(cost_str_cleaned)
                            
                            # Mark as valid if we successfully parsed a number (including 0)
                            if temp_cost is not None:
                                cost_100k_valid = True
                                # Only use if it's a valid positive number (not 0)
                                if temp_cost > 0:
                                    cost_value = temp_cost
                                    cost_source = 'cost_100k'
                        except (ValueError, TypeError):
                            cost_100k_valid = False
                            pass
                # # Perf Based Price no longer used as fallback (column is display-only)
                # if cost_value is None or (cost_100k_valid and cost_value == 0):
                #     if perf_price_available: ... cost_value = temp_perf; cost_source = 'perf_based_price'
                
                # Check if cost value > 30 (only from Cost for 100k Views)
                if cost_value is not None and cost_value > 30:
                    bad_reasons.append(f"Cost for 100k Views ({cost_value:.2f}) > 30")
                    is_bad_cost = True
                
                # Bad Check 3: Latest Video Date (Column P, index 15) NOT within 15 days
                # Only mark as bad if date is > 15 days AND at least one pricing field exists (TikTok Price or Perf Based Price)
                is_bad_date = False
                is_within_15_days = False
                if len(row) > COLUMNS['LATEST_DATE'] and row[COLUMNS['LATEST_DATE']]:
                    latest_date = None
                    original_value = row[COLUMNS['LATEST_DATE']]
                    
                    # Log the raw value for debugging
                    _logger.debug(f"   Row {i} (@{username}): Latest Video Date (Column P) raw value = {original_value} (type: {type(original_value).__name__})")
                    
                    # Handle different date formats
                    if isinstance(original_value, (int, float)):
                        # Serial number format (Excel/Sheets date format)
                        # Excel epoch starts at December 30, 1899
                        try:
                            excel_epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
                            latest_date = excel_epoch + timedelta(days=float(original_value))
                            _logger.debug(f"   Row {i} (@{username}): Parsed serial number {original_value} as {latest_date.strftime('%Y-%m-%d')}")
                        except (ValueError, TypeError, OverflowError) as e:
                            _logger.warning(f"   Row {i} (@{username}): Failed to parse serial number {original_value}: {e}")
                    else:
                        # String format
                        latest_date_str = str(original_value).strip()
                        if latest_date_str and latest_date_str.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', ''] and not latest_date_str.startswith('#'):
                            try:
                                # Try to parse date in YYYY-MM-DD format
                                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                                _logger.debug(f"   Row {i} (@{username}): Parsed string '{latest_date_str}' as {latest_date.strftime('%Y-%m-%d')}")
                            except (ValueError, TypeError) as e:
                                # If date parsing fails, skip this check
                                _logger.warning(f"   Row {i} (@{username}): Failed to parse date string '{latest_date_str}': {e}")
                    
                    # Check if date is more than 15 days old
                    if latest_date:
                        days_old = (current_date - latest_date).days
                        _logger.debug(f"   Row {i} (@{username}): Date is {days_old} days old")
                        if days_old > 15:
                            # Check if at least one pricing field exists before marking as bad
                            has_pricing_field = False
                            if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                                tiktok_price_str = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                                if tiktok_price_str and tiktok_price_str.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                                    has_pricing_field = True
                            # # Perf Based Price no longer used in logic
                            # if not has_pricing_field and len(row) > COLUMNS['PERF_BASED_PRICE'] ...
                            
                            if has_pricing_field:
                                date_display = latest_date.strftime('%Y-%m-%d')
                                bad_reasons.append(f"Latest video date ({date_display}) is {days_old} days old (> 15 days)")
                                _logger.info(f"   Row {i} (@{username}): MARKED AS BAD - Date is {days_old} days old (pricing field exists)")
                            else:
                                # Date is old but no pricing fields - don't mark as bad, will go to Initial Reachout
                                _logger.info(f"   Row {i} (@{username}): Date is {days_old} days old but no pricing fields - will move to Initial Reachout")
                        else:
                            # Date is within 15 days
                            is_within_15_days = True
                    else:
                        # Could not parse date - treat as bad
                        bad_reasons.append("Latest Video Date could not be parsed")
                        is_bad_date = True
                else:
                    # No date available - treat as bad
                    bad_reasons.append("Latest Video Date is missing")
                    is_bad_date = True
                
                # CLASSIFICATION DECISION
                if bad_reasons:
                    # This is a BAD account
                    all_bad_accounts.append((tab_name, i, row, bad_reasons))
                    _logger.debug(f"   Row {i} (@{username}): Classified as BAD - {', '.join(bad_reasons)}")
                else:
                    # Not bad, check if it's GOOD or RELIABLE
                    # Good Account criteria (must pass ALL):
                    # - NOT bad (already passed)
                    # - Latest Video Date IS within 15 days
                    # - Cost for 100k Views <= 30 (or Perf Based Price <= 30 as fallback)
                    # - TikTok Price (Column G) is not empty and is a valid number
                    
                    # cost_value is already set from Bad Check 2 with fallback logic
                    # If cost_value is None here, both Cost for 100k Views and Perf Based Price are invalid
                    
                    # Check if TikTok Price is valid (not empty and is a number)
                    has_valid_tiktok_price = False
                    if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                        price_str = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                        if price_str:  # Not empty string
                            try:
                                # Try to parse as number
                                if isinstance(row[COLUMNS['TIKTOK_PRICE']], (int, float)):
                                    price_value = float(row[COLUMNS['TIKTOK_PRICE']])
                                    has_valid_tiktok_price = True
                                else:
                                    # String value - parse it
                                    price_str_cleaned = price_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').strip()
                                    price_value = float(price_str_cleaned)
                                    has_valid_tiktok_price = True
                            except (ValueError, TypeError):
                                # Not a valid number
                                has_valid_tiktok_price = False
                    
                    # # Perf Based Price no longer used in logic (column is display-only)
                    # has_perf_based_price = False
                    # if len(row) > COLUMNS['PERF_BASED_PRICE'] ...
                    has_perf_based_price = False
                    
                    is_good = False
                    # Good Account criteria (MUST PASS ALL):
                    # 1. Date is within 15 days (is_within_15_days)
                    # 2. TikTok Price is available (Perf Based Price not used)
                    # 3. Cost <= 30 (from Cost for 100k Views only)
                    
                    can_determine_cost = False
                    if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                        tiktok_price_str = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                        if tiktok_price_str and tiktok_price_str.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                            can_determine_cost = True
                    # # Perf Based Price no longer used
                    # if not can_determine_cost and len(row) > COLUMNS['PERF_BASED_PRICE'] ...
                    
                    # Good if ALL criteria are met:
                    # 1. Date is within 15 days
                    # 2. TikTok Price is available (Perf Based Price not used)
                    # 3. Cost can be determined AND cost <= 30
                    if is_within_15_days:
                        has_pricing_field = has_valid_tiktok_price or has_perf_based_price  # has_perf_based_price is always False now
                        if has_pricing_field and can_determine_cost:
                            if cost_value is not None and cost_value <= 30:
                                is_good = True
                        # If TikTok Price is empty, cannot determine cost, so is_good stays False
                    
                    # Log detailed evaluation for accounts in Reliable or Good tabs
                    if tab_name in [TAB_NAMES['RELIABLE'], TAB_NAMES['GOOD']]:
                        cost_source_display = "Cost for 100k Views" if cost_source == 'cost_100k' else "None"
                        pricing_field_display = "TikTok Price" if has_valid_tiktok_price else "None"
                        _logger.info(f"   Row {i} (@{username}): Evaluating from '{tab_name}' - is_good={is_good} (date_ok={is_within_15_days}, cost={cost_value} from {cost_source_display}, pricing_field={pricing_field_display})")
                    
                    if is_good:
                        # Check if it's RELIABLE (must also have # of Times Booked > 0)
                        is_reliable = False
                        times_booked_status = "missing"
                        if len(row) > COLUMNS['TIMES_BOOKED'] and row[COLUMNS['TIMES_BOOKED']] is not None:
                            times_booked_str = str(row[COLUMNS['TIMES_BOOKED']]).strip()
                            # Only check if not empty
                            if times_booked_str:  # Not empty string
                                try:
                                    times_booked_value = None
                                    if isinstance(row[COLUMNS['TIMES_BOOKED']], (int, float)):
                                        times_booked_value = int(row[COLUMNS['TIMES_BOOKED']])
                                    else:
                                        times_booked_cleaned = times_booked_str.replace(',', '').strip()
                                        times_booked_value = int(float(times_booked_cleaned))
                                    
                                    if times_booked_value > 0:
                                        is_reliable = True
                                        times_booked_status = f"valid ({times_booked_value})"
                                    else:
                                        times_booked_status = f"zero or negative ({times_booked_value})"
                                except (ValueError, TypeError) as e:
                                    # Not a valid number, so not reliable
                                    times_booked_status = f"invalid ('{times_booked_str}')"
                            else:
                                times_booked_status = "empty string"
                        
                        # Log detailed info for accounts in Reliable tab
                        if tab_name == TAB_NAMES['RELIABLE']:
                            _logger.info(f"   Row {i} (@{username}): Evaluating Reliable status - Times Booked: {times_booked_status}")
                        
                        if is_reliable:
                            # This is a RELIABLE account
                            all_reliable_accounts.append((tab_name, i, row))
                            _logger.debug(f"   Row {i} (@{username}): Classified as RELIABLE")
                        else:
                            # This is a GOOD account (but not reliable)
                            all_good_accounts.append((tab_name, i, row))
                            if tab_name == TAB_NAMES['RELIABLE']:
                                _logger.info(f"   Row {i} (@{username}): Moving from Reliable to Good (Times Booked not > 0)")
                            else:
                                _logger.debug(f"   Row {i} (@{username}): Classified as GOOD")
                    else:
                        # Not bad, but also not good enough
                        # Account is NOT bad (date is within 15 days)
                        stay_reasons = []
                        # Check if we can determine cost
                        can_determine_cost_check = False
                        if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                            tiktok_price_str_check = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                            if tiktok_price_str_check and tiktok_price_str_check.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                                can_determine_cost_check = True
                        # # Perf Based Price no longer used in logic
                        # if not can_determine_cost_check and len(row) > COLUMNS['PERF_BASED_PRICE'] ...
                        
                        # Check if TikTok Price is available
                        has_valid_tiktok_price_check = False
                        if len(row) > COLUMNS['TIKTOK_PRICE'] and row[COLUMNS['TIKTOK_PRICE']] is not None:
                            tiktok_price_str_check2 = str(row[COLUMNS['TIKTOK_PRICE']]).strip()
                            if tiktok_price_str_check2 and tiktok_price_str_check2.upper() not in ['NA', 'N/A', 'NULL', 'EMPTY', '']:
                                try:
                                    if isinstance(row[COLUMNS['TIKTOK_PRICE']], (int, float)):
                                        has_valid_tiktok_price_check = True
                                    else:
                                        price_str_cleaned_check = tiktok_price_str_check2.replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').strip()
                                        float(price_str_cleaned_check)
                                        has_valid_tiktok_price_check = True
                                except (ValueError, TypeError):
                                    pass
                        
                        # # Perf Based Price no longer used in logic
                        # has_perf_based_price_check = False
                        # if len(row) > COLUMNS['PERF_BASED_PRICE'] ...
                        has_perf_based_price_check = False
                        
                        if not has_valid_tiktok_price_check:
                            stay_reasons.append("TikTok Price is empty")
                        elif not can_determine_cost_check:
                            stay_reasons.append("cannot determine cost (Cost for 100k Views invalid)")
                        elif cost_value is None:
                            stay_reasons.append("no cost value (Cost for 100k Views invalid)")
                        elif cost_value > 30:
                            stay_reasons.append(f"Cost for 100k Views {cost_value:.2f} > 30")
                        
                        # If account is in Good, Reliable, or Bad Accounts tabs but doesn't meet Good criteria,
                        # move it back to Initial Reachout
                        # (These accounts are NOT bad - they just don't qualify as Good)
                        if tab_name in [TAB_NAMES['GOOD'], TAB_NAMES['RELIABLE'], TAB_NAMES['BAD']]:
                            accounts_to_initial.append((tab_name, i, row))
                            _logger.info(f"   Row {i} (@{username}): Moving from '{tab_name}' to Initial Reachout (doesn't meet Good criteria: {', '.join(stay_reasons)})")
                        else:
                            _logger.debug(f"   Row {i} (@{username}): Not classified (stays in {tab_name})")
        
        # Summary logging
        log_and_print("\n" + "=" * 80)
        log_and_print("CLASSIFICATION SUMMARY")
        log_and_print("=" * 80)
        log_and_print(f"Bad Accounts: {len(all_bad_accounts)}")
        log_and_print(f"Good Accounts: {len(all_good_accounts)}")
        log_and_print(f"Reliable Accounts: {len(all_reliable_accounts)}")
        log_and_print(f"Accounts to Initial Reachout: {len(accounts_to_initial)}")
        log_and_print(f"Empty Rows: {len(all_empty_rows)}")
        
        if not all_bad_accounts and not all_good_accounts and not all_reliable_accounts and not accounts_to_initial and not all_empty_rows:
            log_and_print("No accounts to move or rows to delete")
            return True
        
        # Log details of classified accounts
        if all_bad_accounts:
            log_and_print(f"\nBad Accounts ({len(all_bad_accounts)}):")
            for source_tab, row_index, row, reasons in all_bad_accounts:
                username = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] else "Unknown"
                log_and_print(f"   @{username} (from {source_tab}, row {row_index}): {', '.join(reasons)}")
        
        if all_good_accounts:
            log_and_print(f"\nGood Accounts ({len(all_good_accounts)}):")
            for source_tab, row_index, row in all_good_accounts:
                username = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] else "Unknown"
                log_and_print(f"   @{username} (from {source_tab}, row {row_index})")
        
        if all_reliable_accounts:
            log_and_print(f"\nReliable Accounts ({len(all_reliable_accounts)}):")
            for source_tab, row_index, row in all_reliable_accounts:
                username = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] else "Unknown"
                log_and_print(f"   @{username} (from {source_tab}, row {row_index})")
        
        if accounts_to_initial:
            log_and_print(f"\nAccounts to Initial Reachout ({len(accounts_to_initial)}):")
            for source_tab, row_index, row in accounts_to_initial:
                username = row[COLUMNS['USERNAME']] if len(row) > COLUMNS['USERNAME'] else "Unknown"
                log_and_print(f"   @{username} (from {source_tab}, row {row_index})")
        
        # Get spreadsheet metadata to check which tabs exist and get sheet IDs
        spreadsheet_metadata = sheets_get_metadata(service, spreadsheet_id)
        sheets = spreadsheet_metadata.get('sheets', [])
        sheet_names = [sheet['properties']['title'] for sheet in sheets]
        
        # Ensure all destination tabs exist, create if not
        tabs_to_create = []
        if all_bad_accounts and TAB_NAMES['BAD'] not in sheet_names:
            tabs_to_create.append(TAB_NAMES['BAD'])
        if all_good_accounts and TAB_NAMES['GOOD'] not in sheet_names:
            tabs_to_create.append(TAB_NAMES['GOOD'])
        if all_reliable_accounts and TAB_NAMES['RELIABLE'] not in sheet_names:
            tabs_to_create.append(TAB_NAMES['RELIABLE'])
        
        if tabs_to_create:
            log_and_print(f"\nCreating new tabs: {', '.join(tabs_to_create)}")
            for tab_name in tabs_to_create:
                requests = [{
                    'addSheet': {
                        'properties': {
                            'title': tab_name
                        }
                    }
                }]
                sheets_batch_update(
                    service,
                    spreadsheet_id,
                    requests
                )
                
                # Copy header row to new sheet (header is on row 2, so we need row 1 and row 2)
                # Get the header from the main tab
                main_tab_range = f"{TAB_NAMES['MAIN']}!A1:W2"
                header_result = sheets_get_values(
                    service,
                    spreadsheet_id,
                    main_tab_range
                )
                if header_result:
                    header_values = header_result.get('values', [])
                    
                    if header_values:
                        sheets_update_values(
                            service,
                            spreadsheet_id,
                            f"{tab_name}!A1",
                            header_values,
                            value_input_option='USER_ENTERED'
                        )
                    log_and_print(f"  Created tab '{tab_name}' with headers")
            
            # Refresh spreadsheet metadata after creating tabs
            spreadsheet_metadata = sheets_get_metadata(service, spreadsheet_id)
            sheets = spreadsheet_metadata.get('sheets', [])
        
        # Build a mapping of sheet names to sheet IDs (needed for formatting copy)
        sheet_id_map = {}
        for sheet in sheets:
            sheet_id_map[sheet['properties']['title']] = sheet['properties']['sheetId']
        
        # Function to append rows to a destination tab with formulas
        def append_to_tab(dest_tab_name, rows_list, classification_type):
            """Copy rows from source to destination tab using copyPaste to preserve everything."""
            if not rows_list:
                return
            
            log_and_print(f"\nCopying {len(rows_list)} {classification_type} accounts to '{dest_tab_name}' tab...")
            
            # Get destination sheet ID
            dest_sheet_id = sheet_id_map.get(dest_tab_name)
            if dest_sheet_id is None:
                log_and_print(f"  ERROR: Could not find sheet ID for '{dest_tab_name}'", "ERROR")
                return
            
            # Get current number of rows WITH DATA in destination tab
            existing_data = sheets_get_values(
                service,
                spreadsheet_id,
                f"{dest_tab_name}!A:U"
            )
            existing_values = existing_data.get('values', [])
            
            # Find first truly empty row (where USERNAME and Link columns are empty)
            # Start from row 3 (after headers in rows 1-2)
            next_row = DATA_START_ROW
            for i, row in enumerate(existing_values[HEADER_ROW:], start=DATA_START_ROW):
                # Check if USERNAME (Column A) and Link (Column B) are both empty
                username = row[COLUMNS['USERNAME']].strip() if len(row) > COLUMNS['USERNAME'] and row[COLUMNS['USERNAME']] else ""
                link = row[COLUMNS['LINK']].strip() if len(row) > COLUMNS['LINK'] and row[COLUMNS['LINK']] else ""
                
                if not username and not link:
                    # Found first empty row
                    next_row = i
                    break
            else:
                # All rows have data, use next row after last row
                next_row = len(existing_values) + 1
            
            # Get destination sheet properties to check available rows
            dest_sheet_props = None
            for sheet in sheets:
                if sheet['properties']['sheetId'] == dest_sheet_id:
                    dest_sheet_props = sheet['properties']
                    break
            
            if dest_sheet_props:
                current_row_count = dest_sheet_props.get('gridProperties', {}).get('rowCount', 1000)
                rows_needed = next_row - 1 + len(rows_list)  # Total rows needed (including existing)
                
                log_and_print(f"  Current sheet has {current_row_count} rows, {len(existing_values)} rows with data")
                log_and_print(f"  Will write to rows {next_row} to {next_row + len(rows_list) - 1} using existing empty rows")
                
                # Only expand if we truly don't have enough rows
                if current_row_count < rows_needed:
                    log_and_print(f"  WARNING: Need to expand sheet from {current_row_count} to {rows_needed} rows")
                    expand_request = {
                        'updateSheetProperties': {
                            'properties': {
                                'sheetId': dest_sheet_id,
                                'gridProperties': {
                                    'rowCount': rows_needed
                                }
                            },
                            'fields': 'gridProperties.rowCount'
                        }
                    }
                    sheets_batch_update(
                        service,
                        spreadsheet_id,
                        [expand_request]
                    )
                else:
                    log_and_print(f"  Sheet has enough empty rows, no expansion needed")
            
            # Copy each row from source to destination using copyPaste with PASTE_NORMAL
            copy_requests = []
            
            for idx, item in enumerate(rows_list):
                if classification_type == "bad":
                    source_tab, source_row_index, row, reasons = item
                else:
                    source_tab, source_row_index, row = item
                
                dest_row_index = next_row + idx
                
                # Get source sheet ID
                source_sheet_id = sheet_id_map.get(source_tab)
                if source_sheet_id is None:
                    continue
                
                # Copy entire row (values, formulas, formatting) from source to destination
                copy_requests.append({
                    'copyPaste': {
                        'source': {
                            'sheetId': source_sheet_id,
                            'startRowIndex': source_row_index - 1,  # Convert to 0-based
                            'endRowIndex': source_row_index,
                            'startColumnIndex': 0,
                            'endColumnIndex': 22  # Columns A-W
                        },
                        'destination': {
                            'sheetId': dest_sheet_id,
                            'startRowIndex': dest_row_index - 1,  # Convert to 0-based
                            'endRowIndex': dest_row_index,
                            'startColumnIndex': 0,
                            'endColumnIndex': 22
                        },
                        'pasteType': 'PASTE_NORMAL'  # Copy everything: values, formulas, formatting
                    }
                })
            
            # Execute copy requests in batches
            if copy_requests:
                batch_size = 100
                for i in range(0, len(copy_requests), batch_size):
                    batch = copy_requests[i:i + batch_size]
                    sheets_batch_update(
                        service,
                        spreadsheet_id,
                        batch
                    )
                log_and_print(f"  Successfully copied {len(copy_requests)} rows to existing empty rows (values, formulas, formatting preserved)")
        
        # Append accounts to their destination tabs
        if all_bad_accounts:
            append_to_tab(TAB_NAMES['BAD'], all_bad_accounts, "bad")
        if all_good_accounts:
            append_to_tab(TAB_NAMES['GOOD'], all_good_accounts, "good")
        if all_reliable_accounts:
            append_to_tab(TAB_NAMES['RELIABLE'], all_reliable_accounts, "reliable")
        if accounts_to_initial:
            append_to_tab(TAB_NAMES['MAIN'], accounts_to_initial, "initial")
        
        # Delete moved accounts and empty rows from their source tabs
        # Group rows by source tab
        rows_to_delete_by_tab = {}
        
        # Add bad accounts
        for source_tab, row_index, row, reasons in all_bad_accounts:
            if source_tab not in rows_to_delete_by_tab:
                rows_to_delete_by_tab[source_tab] = []
            rows_to_delete_by_tab[source_tab].append(row_index)
        
        # Add good accounts
        for source_tab, row_index, row in all_good_accounts:
            if source_tab not in rows_to_delete_by_tab:
                rows_to_delete_by_tab[source_tab] = []
            rows_to_delete_by_tab[source_tab].append(row_index)
        
        # Add reliable accounts
        for source_tab, row_index, row in all_reliable_accounts:
            if source_tab not in rows_to_delete_by_tab:
                rows_to_delete_by_tab[source_tab] = []
            rows_to_delete_by_tab[source_tab].append(row_index)
        
        # Add accounts moving back to initial
        for source_tab, row_index, row in accounts_to_initial:
            if source_tab not in rows_to_delete_by_tab:
                rows_to_delete_by_tab[source_tab] = []
            rows_to_delete_by_tab[source_tab].append(row_index)
        
        # Add empty rows
        for source_tab, row_index in all_empty_rows:
            if source_tab not in rows_to_delete_by_tab:
                rows_to_delete_by_tab[source_tab] = []
            rows_to_delete_by_tab[source_tab].append(row_index)
        
        # Delete rows from each source tab
        log_and_print("\nDeleting moved accounts and empty rows from source tabs:")
        for source_tab, row_indices in rows_to_delete_by_tab.items():
            if source_tab not in sheet_id_map:
                log_and_print(f"  ERROR: Could not find sheet ID for '{source_tab}'", "ERROR")
                continue
            
            sheet_id = sheet_id_map[source_tab]
            
            # Sort in descending order (delete from bottom to top to maintain indices)
            row_indices_sorted = sorted(set(row_indices), reverse=True)
            
            if row_indices_sorted:
                delete_requests = []
                for row_index in row_indices_sorted:
                    delete_requests.append({
                        'deleteDimension': {
                            'range': {
                                'sheetId': sheet_id,
                                'dimension': 'ROWS',
                                'startIndex': row_index - 1,  # Convert to 0-based index
                                'endIndex': row_index
                            }
                        }
                    })
                
                # Batch delete requests (Google Sheets API allows up to 100 requests per batch)
                batch_size = 100
                for i in range(0, len(delete_requests), batch_size):
                    batch = delete_requests[i:i + batch_size]
                    sheets_batch_update(
                        service,
                        spreadsheet_id,
                        batch
                    )
                
                log_and_print(f"  Deleted {len(delete_requests)} rows from '{source_tab}'")
        
        log_and_print("\n" + "=" * 80)
        log_and_print("CLASSIFICATION COMPLETE")
        log_and_print(f"  Bad Accounts moved: {len(all_bad_accounts)}")
        log_and_print(f"  Good Accounts moved: {len(all_good_accounts)}")
        log_and_print(f"  Reliable Accounts moved: {len(all_reliable_accounts)}")
        log_and_print(f"  Accounts moved to Initial Reachout: {len(accounts_to_initial)}")
        log_and_print(f"  Empty rows deleted: {len(all_empty_rows)}")
        log_and_print("=" * 80)
        return True
        
    except Exception as e:
        log_and_print(f"ERROR moving bad accounts: {e}", "ERROR")
        import traceback
        log_and_print(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False


def run():
    """
    Main entry point. Sets up browser, logs in once, and runs scraping job(s).
    Supports processing multiple sheets while keeping the browser open.
    """
    # Track overall run statistics
    overall_run_start_time = time.time()
    overall_analytics = {
        'total_sheets': 0,
        'successful_sheets': 0,
        'failed_sheets': 0,
        'total_profiles': 0,
        'successful_profiles': 0,
        'failed_profiles': 0,
        'invalid_or_not_found_profiles': 0,
        'empty_profiles': 0,
        'na_url_profiles': 0,
        'total_videos_processed': 0,
        'total_run_time': 0,
        'profile_times': [],
        'video_times': []
    }
    
    ensure_dependencies()
    
    # Initialize logging system
    global _logger, SPREADSHEET_ID
    _logger = setup_logging()
    
    # Get Cairo time for the run
    cairo_time = get_cairo_time()
    
    # Log run separator and metadata
    _logger.info("=" * 100)
    _logger.info("=" * 100)
    _logger.info(f"TIKTOK SCRAPER STARTED - {cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    if PROCESS_MULTIPLE_SHEETS:
        _logger.info("Multi-sheet mode enabled - will process multiple sheets with one browser session")
    else:
        _logger.info("Single-sheet mode - processing one sheet")
    _logger.info("=" * 100)
    
    log_and_print("\n" + "=" * 80)
    log_and_print("=" * 80)
    log_and_print("TIKTOK SCRAPER - FULL RUN STARTED")
    log_and_print("=" * 80)
    log_and_print("=" * 80)
    log_and_print(f"Start Time: {cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    if PROCESS_MULTIPLE_SHEETS:
        log_and_print("Mode: Multi-sheet orchestration")
    else:
        log_and_print("Mode: Single-sheet processing")
    log_and_print("=" * 80)
    
    # Log current mode and toggle states
    log_and_print("\n" + "=" * 80)
    log_and_print("CURRENT MODE AND TOGGLE STATES")
    log_and_print("=" * 80)
    
    # Mode information
    log_and_print(f"\nMode Configuration:")
    log_and_print(f"  MANUAL_MODE: {MANUAL_MODE}")
    if not MANUAL_MODE:
        log_and_print(f"  FULL_RUN: {FULL_RUN}")
        if FULL_RUN:
            log_and_print(f"    → Full run mode: All features enabled")
        else:
            log_and_print(f"    → Limited run mode: Only basic features enabled")
    else:
        log_and_print(f"  → Manual mode: Individual toggles are used")
    
    # Feature toggles
    log_and_print(f"\nFeature Toggles:")
    log_and_print(f"  RUN_PROFILE_SCRAPING: {RUN_PROFILE_SCRAPING}")
    log_and_print(f"  RUN_NA_PROFILES: {RUN_NA_PROFILES}")
    log_and_print(f"  SKIP_TIKTOK_LOGIN: {SKIP_TIKTOK_LOGIN}")
    log_and_print(f"  RUN_UPDATE_HEADERS: {RUN_UPDATE_HEADERS}")
    log_and_print(f"  RUN_ACCOUNT_CLASSIFICATION: {RUN_ACCOUNT_CLASSIFICATION}")
    
    # Tab processing toggles
    log_and_print(f"\nTab Processing Toggles:")
    log_and_print(f"  PROCESS_INITIAL_REACHOUT: {PROCESS_INITIAL_REACHOUT}")
    log_and_print(f"  PROCESS_BAD_ACCOUNTS: {PROCESS_BAD_ACCOUNTS}")
    log_and_print(f"  PROCESS_GOOD_ACCOUNTS: {PROCESS_GOOD_ACCOUNTS}")
    log_and_print(f"  PROCESS_RELIABLE_ACCOUNTS: {PROCESS_RELIABLE_ACCOUNTS}")
    
    # Multi-sheet toggle
    log_and_print(f"\nOther Settings:")
    log_and_print(f"  PROCESS_MULTIPLE_SHEETS: {PROCESS_MULTIPLE_SHEETS}")
    log_and_print(f"  RUN_BUFFER_WARMUP: {RUN_BUFFER_WARMUP}")
    log_and_print(f"  FAST_STARTUP: {FAST_STARTUP}")
    
    log_and_print("=" * 80 + "\n")
    
    # Also log to file
    _logger.info("=" * 100)
    _logger.info("CURRENT MODE AND TOGGLE STATES")
    _logger.info("=" * 100)
    _logger.info(f"MANUAL_MODE: {MANUAL_MODE}")
    if not MANUAL_MODE:
        _logger.info(f"FULL_RUN: {FULL_RUN}")
    _logger.info(f"RUN_PROFILE_SCRAPING: {RUN_PROFILE_SCRAPING}")
    _logger.info(f"RUN_NA_PROFILES: {RUN_NA_PROFILES}")
    _logger.info(f"SKIP_TIKTOK_LOGIN: {SKIP_TIKTOK_LOGIN}")
    _logger.info(f"RUN_UPDATE_HEADERS: {RUN_UPDATE_HEADERS}")
    _logger.info(f"RUN_ACCOUNT_CLASSIFICATION: {RUN_ACCOUNT_CLASSIFICATION}")
    _logger.info(f"PROCESS_INITIAL_REACHOUT: {PROCESS_INITIAL_REACHOUT}")
    _logger.info(f"PROCESS_BAD_ACCOUNTS: {PROCESS_BAD_ACCOUNTS}")
    _logger.info(f"PROCESS_GOOD_ACCOUNTS: {PROCESS_GOOD_ACCOUNTS}")
    _logger.info(f"PROCESS_RELIABLE_ACCOUNTS: {PROCESS_RELIABLE_ACCOUNTS}")
    _logger.info(f"PROCESS_MULTIPLE_SHEETS: {PROCESS_MULTIPLE_SHEETS}")
    _logger.info(f"RUN_BUFFER_WARMUP: {RUN_BUFFER_WARMUP}")
    _logger.info(f"FAST_STARTUP: {FAST_STARTUP}")
    _logger.info("=" * 100)
    
    # Determine which sheets to process
    sheets_to_process = []
    if PROCESS_MULTIPLE_SHEETS:
        sheets_to_process = read_sheets_to_process()
        if not sheets_to_process:
            log_and_print("ERROR: No sheets to process. Exiting.", "ERROR")
            return
    else:
        # Single sheet mode - use the configured SPREADSHEET_ID
        sheets_to_process = [{
            'name': 'Default Sheet',
            'spreadsheet_id': SPREADSHEET_ID,
            'url': f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
        }]
    
    log_and_print(f"\nTotal sheets to process: {len(sheets_to_process)}")
    log_and_print("=" * 80)
    
    # Check if we need the browser (only for profile scraping)
    need_browser = RUN_PROFILE_SCRAPING or RUN_NA_PROFILES
    
    if need_browser:
        log_and_print("Browser is needed for scraping. Setting up browser...")
        log_and_print("=" * 80)
        p, context, page = launch_browser_with_profile()
        
        # Login once before scraping (optional toggle)
        if SKIP_TIKTOK_LOGIN:
            log_and_print("\nSkipping TikTok login (SKIP_TIKTOK_LOGIN = True).")
            log_and_print("Using existing browser profile session as-is.")
        else:
            log_and_print("\nLogging into TikTok (one-time login)...")
            if not login_to_tiktok(page):
                log_and_print("WARNING: Login failed, but continuing anyway...", "WARNING")
                log_and_print("You may need to manually complete the login in the browser window", "WARNING")
            else:
                log_and_print("Login successful!")

        if RUN_BUFFER_WARMUP:
            warmup_buffer_profile(page)
        else:
            log_and_print("Skipping buffer profile warm-up (RUN_BUFFER_WARMUP = False).")
        
        log_and_print("\n" + "=" * 80)
        log_and_print("Browser logged in and will remain open for all sheets")
        log_and_print("=" * 80)
    else:
        log_and_print("Browser not needed (RUN_PROFILE_SCRAPING and RUN_NA_PROFILES are both False)")
        log_and_print("Skipping browser setup and login...")
        log_and_print("=" * 80)
        page = None  # No browser page needed
    
    # Process each sheet
    overall_analytics['total_sheets'] = len(sheets_to_process)
    
    for idx, sheet_info in enumerate(sheets_to_process, start=1):
        sheet_name = sheet_info['name']
        sheet_id = sheet_info['spreadsheet_id']
        
        log_and_print("\n" + "=" * 80)
        log_and_print(f"PROCESSING SHEET {idx}/{len(sheets_to_process)}")
        log_and_print(f"Name: {sheet_name}")
        log_and_print(f"ID: {sheet_id}")
        cairo_now = get_cairo_time()
        log_and_print(f"Time: {cairo_now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
        log_and_print("=" * 80)
        
        # Set the global SPREADSHEET_ID to the current sheet
        SPREADSHEET_ID = sheet_id
        
        try:
            sheet_start_time = time.time()
            run_scraping_job(page, manager_label=sheet_name)
            flush_pending_sheet_writes(f"post_sheet_success:{sheet_name}")
            sheet_elapsed = time.time() - sheet_start_time
            
            log_and_print("\n" + "=" * 80)
            log_and_print(f"Sheet '{sheet_name}' completed successfully in {sheet_elapsed:.1f} seconds ({sheet_elapsed/60:.1f} minutes)")
            log_and_print("=" * 80)
            overall_analytics['successful_sheets'] += 1
            
            # Note: Analytics from run_scraping_job are logged per sheet, but we can't easily aggregate them
            # without modifying run_scraping_job to return analytics. For now, we'll track sheet-level stats.
            
        except Exception as e:
            log_and_print(f"ERROR processing sheet '{sheet_name}': {e}", "ERROR")
            _logger.error(f"ERROR processing sheet '{sheet_name}': {e}")
            import traceback
            _logger.error(f"Traceback: {traceback.format_exc()}")
            flush_pending_sheet_writes(f"post_sheet_exception:{sheet_name}")
            overall_analytics['failed_sheets'] += 1

        # In multi-sheet mode, close browser immediately after the last sheet scrape completes.
        if PROCESS_MULTIPLE_SHEETS and need_browser and idx == len(sheets_to_process) and page is not None:
            flush_pending_sheet_writes("final_flush_before_browser_close")
            close_browser_resources(p, context, page, reason="all_sheets_complete")
            page = None
            context = None
            p = None
            log_and_print("Last sheet done — browser closed.")
        
        # Small pause between sheets (optional)
        if idx < len(sheets_to_process):
            log_and_print("\nPausing 2 seconds before next sheet...")
            time.sleep(2)
    
    # Calculate total run time
    overall_run_end_time = time.time()
    overall_run_duration = overall_run_end_time - overall_run_start_time
    overall_analytics['total_run_time'] = overall_run_duration
    end_cairo_time = get_cairo_time()
    
    # Final summary
    log_and_print("\n" + "=" * 80)
    log_and_print("=" * 80)
    log_and_print("ALL SHEETS PROCESSING COMPLETE")
    log_and_print("=" * 80)
    log_and_print(f"Total sheets processed: {overall_analytics['total_sheets']}")
    log_and_print(f"Successful sheets: {overall_analytics['successful_sheets']}")
    log_and_print(f"Failed sheets: {overall_analytics['failed_sheets']}")
    if overall_analytics['total_sheets'] > 0:
        success_rate = (overall_analytics['successful_sheets'] / overall_analytics['total_sheets']) * 100
        log_and_print(f"Success rate: {success_rate:.1f}%")
    log_and_print("=" * 80)
    
    # Aggregate Reliable Accounts and Good Accounts to combined sheet (only in multi-sheet mode)
    if PROCESS_MULTIPLE_SHEETS and overall_analytics['successful_sheets'] > 0:
        # Aggregate Reliable Accounts only if both RUN_ACCOUNT_CLASSIFICATION and PROCESS_RELIABLE_ACCOUNTS are True
        if RUN_ACCOUNT_CLASSIFICATION and PROCESS_RELIABLE_ACCOUNTS:
            try:
                aggregate_reliable_accounts_to_combined(sheets_to_process)
            except Exception as e:
                log_and_print(f"ERROR during Reliable Accounts aggregation: {e}", "ERROR")
                _logger.error(f"ERROR during Reliable Accounts aggregation: {e}")
        
        # Aggregate Good Accounts only if both RUN_ACCOUNT_CLASSIFICATION and PROCESS_GOOD_ACCOUNTS are True
        if RUN_ACCOUNT_CLASSIFICATION and PROCESS_GOOD_ACCOUNTS:
            try:
                aggregate_good_accounts_to_combined(sheets_to_process)
            except Exception as e:
                log_and_print(f"ERROR during Good Accounts aggregation: {e}", "ERROR")
                _logger.error(f"ERROR during Good Accounts aggregation: {e}")
    
    # Close browser if it was opened
    if need_browser:
        if page is not None:
            log_and_print("\nClosing browser...")
            close_browser_resources(p, context, page, reason="run_end")
            log_and_print("Browser closed and Playwright stopped.")
        else:
            log_and_print("\nBrowser already closed.")
    else:
        log_and_print("\nNo browser to close.")
    
    # Final comprehensive run summary
    log_and_print("\n" + "=" * 80)
    log_and_print("=" * 80)
    log_and_print("FULL RUN SUMMARY")
    log_and_print("=" * 80)
    log_and_print("=" * 80)
    log_and_print(f"End Time: {end_cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    log_and_print(f"Start Time: {cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    
    # Calculate duration breakdown
    hours = int(overall_run_duration // 3600)
    minutes = int((overall_run_duration % 3600) // 60)
    seconds = int(overall_run_duration % 60)
    
    log_and_print("\n⏱️  TOTAL RUN TIME:")
    log_and_print(f"   Duration: {hours}h {minutes}m {seconds}s ({overall_run_duration:.2f} seconds)")
    if overall_analytics['total_sheets'] > 0:
        avg_sheet_time = overall_run_duration / overall_analytics['total_sheets']
        avg_sheet_hours = int(avg_sheet_time // 3600)
        avg_sheet_minutes = int((avg_sheet_time % 3600) // 60)
        avg_sheet_seconds = int(avg_sheet_time % 60)
        log_and_print(f"   Average per sheet: {avg_sheet_hours}h {avg_sheet_minutes}m {avg_sheet_seconds}s ({avg_sheet_time:.1f} seconds)")
    
    log_and_print("\n📊 SHEET STATISTICS:")
    log_and_print(f"   Total sheets: {overall_analytics['total_sheets']}")
    log_and_print(f"   Successful: {overall_analytics['successful_sheets']}")
    log_and_print(f"   Failed: {overall_analytics['failed_sheets']}")
    if overall_analytics['total_sheets'] > 0:
        success_rate = (overall_analytics['successful_sheets'] / overall_analytics['total_sheets']) * 100
        log_and_print(f"   Success rate: {success_rate:.1f}%")
    
    log_and_print("\n📈 PROCESSING RATE:")
    if overall_run_duration > 0 and overall_analytics['total_sheets'] > 0:
        sheets_per_hour = (overall_analytics['total_sheets'] / overall_run_duration) * 3600
        log_and_print(f"   Sheets per hour: {sheets_per_hour:.2f}")
    
    log_and_print("\n" + "=" * 80)
    log_and_print("=" * 80)
    log_and_print("FULL RUN COMPLETED")
    log_and_print("=" * 80)
    log_and_print("=" * 80)
    
    # Also log to file
    _logger.info("=" * 100)
    _logger.info("=" * 100)
    _logger.info("FULL RUN SUMMARY")
    _logger.info("=" * 100)
    _logger.info(f"Start Time: {cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    _logger.info(f"End Time: {end_cairo_time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    _logger.info(f"Total Duration: {hours}h {minutes}m {seconds}s ({overall_run_duration:.2f} seconds)")
    _logger.info(f"Total Sheets: {overall_analytics['total_sheets']}")
    _logger.info(f"Successful Sheets: {overall_analytics['successful_sheets']}")
    _logger.info(f"Failed Sheets: {overall_analytics['failed_sheets']}")
    if overall_analytics['total_sheets'] > 0:
        success_rate_log = (overall_analytics['successful_sheets'] / overall_analytics['total_sheets']) * 100
        _logger.info(f"Success Rate: {success_rate_log:.1f}%")
    _logger.info("=" * 100)
    _logger.info("=" * 100)


if __name__ == "__main__":
    # Self-tests for strict profile URL validation.
    assert is_valid_profile_url("https://www.tiktok.com/@klipolahraga1")
    assert not is_valid_profile_url("https://vt.tiktok.com/ZSfcRUnDN/")
    assert not is_valid_profile_url("https://www.tiktok.com/@user/video/123")
    assert not is_valid_profile_url("https://m.tiktok.com/v/123.html")
    assert not is_valid_profile_url("https://google.com")
    try:
        run()
    finally:
        flush_pending_sheet_writes("process_exit_finally")
        # Safety net to avoid lingering browser processes on unhandled failures.
        close_browser_resources(reason="process_exit_finally")