"""
Update multiple Google Sheets: remove columns G, H, J and add "Last Payment Date"
to the right of column D on every tab. Uses auto_auth.json for authentication.
"""

import re
import time
from typing import Any, Dict, List, Optional

# Pause between spreadsheets to avoid "Write requests per minute" quota (60/min)
DELAY_BETWEEN_SPREADSHEETS_SEC = 8
# Retry batchUpdate on rate limit (429), wait this many seconds before retry
RATE_LIMIT_RETRY_WAIT_SEC = 65
RATE_LIMIT_MAX_RETRIES = 2

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Spreadsheet URLs to process
SHEET_URLS = [
    "https://docs.google.com/spreadsheets/d/15EgJgngT1GUCfnZdyB1iCw8DL3CI-2S0Npkh8BUHYF8/edit?usp=sharing",
]

# Column indices (0-based): A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9
# Delete G (6), H (7), J (9) — must delete right-to-left: J, H, G
# Then insert one column at index 4 (to the right of D)
COL_DELETE_INDICES = [(9, 10), (7, 8), (6, 7)]  # J, H, G
COL_INSERT_INDEX = 4  # new column E (right of D)
HEADER_NAME = "Last Payment Date"
# Headers in these sheets are on row 2 (row 1 is often a title/empty)
HEADER_ROW = 2


def extract_spreadsheet_id(url: str) -> Optional[str]:
    """Extract spreadsheet ID from a Google Sheets URL."""
    if not url:
        return None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None


def setup_sheets_service():
    """Build Google Sheets API service using auto_auth.json."""
    creds = Credentials.from_service_account_file(
        "auto_auth.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds)


def get_sheet_metadata(service, spreadsheet_id: str) -> Optional[Dict[str, Any]]:
    """Get spreadsheet metadata including all sheet (tab) IDs and titles."""
    try:
        return service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    except Exception as e:
        print(f"  ERROR get metadata: {e}")
        return None


def process_spreadsheet(service, spreadsheet_id: str) -> bool:
    """
    For one spreadsheet: on every tab, delete columns G/H/J and insert
    'Last Payment Date' to the right of column D.
    """
    meta = get_sheet_metadata(service, spreadsheet_id)
    if not meta:
        return False

    sheets = meta.get("sheets", [])
    if not sheets:
        print("  No tabs found, skipping.")
        return True

    requests: List[Dict[str, Any]] = []
    for sheet in sheets:
        props = sheet.get("properties", {})
        sheet_id = props.get("sheetId")
        title = props.get("title", "")
        if sheet_id is None:
            continue
        # Delete columns J, H, G (right to left)
        for start, end in COL_DELETE_INDICES:
            requests.append({
                "deleteDimension": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": start,
                        "endIndex": end,
                    }
                }
            })
        # Insert one column at index 4 (right of D)
        requests.append({
            "insertDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": COL_INSERT_INDEX,
                    "endIndex": COL_INSERT_INDEX + 1,
                }
            }
        })

    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": requests},
            ).execute()
            break
        except HttpError as e:
            if e.resp.status == 429 and attempt < RATE_LIMIT_MAX_RETRIES:
                print("  Rate limit hit, waiting {}s before retry...".format(RATE_LIMIT_RETRY_WAIT_SEC))
                time.sleep(RATE_LIMIT_RETRY_WAIT_SEC)
            else:
                print("  ERROR batchUpdate: {}".format(e))
                return False

    # Set header "Last Payment Date" in new column E, header row for each tab
    for sheet in sheets:
        props = sheet.get("properties", {})
        title = props.get("title", "")
        if not title:
            continue
        # Escape single quotes in sheet name for A1 notation
        safe_title = title.replace("'", "''")
        range_a1 = "'{}'!E{}".format(safe_title, HEADER_ROW)
        try:
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_a1,
                valueInputOption="USER_ENTERED",
                body={"values": [[HEADER_NAME]]},
            ).execute()
        except Exception as e:
            print(f"  ERROR writing header to tab '{title}': {e}")

    print(f"  Updated {len(sheets)} tab(s).")
    return True


def main():
    service = setup_sheets_service()
    if not service:
        print("ERROR: Could not create Google Sheets service. Check auto_auth.json.")
        return

    total = len(SHEET_URLS)
    ok = 0
    for i, url in enumerate(SHEET_URLS, start=1):
        sid = extract_spreadsheet_id(url)
        if not sid:
            print(f"[{i}/{total}] Invalid URL, skipping: {url[:60]}...")
            continue
        print(f"[{i}/{total}] Processing spreadsheet {sid[:20]}...")
        if process_spreadsheet(service, sid):
            ok += 1
        else:
            print("  Failed.")

        if i < total:
            time.sleep(DELAY_BETWEEN_SPREADSHEETS_SEC)

    print("\nDone. Success: {}/{} spreadsheets.".format(ok, total))


if __name__ == "__main__":
    main()
