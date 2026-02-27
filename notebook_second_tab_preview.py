"""
Notebook helper: load only the 2nd tab from the input payments sheet as a DataFrame.

Usage in a notebook:
    from notebook_second_tab_preview import load_second_tab_df
    df, details = load_second_tab_df()
    details
    df.head(20)
"""

import os
from typing import Any, Dict, List, Tuple

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pandas is required for this notebook helper. Install with: pip install pandas"
    ) from exc


# Same source sheet used by monthly_sheet_reader.py
SOURCE_SPREADSHEET_ID = os.environ.get(
    "INPUT_SPREADSHEET_ID",
    "1EXrm0FbudIu44LmgSKXZt2lf96XYdOuwyRWf-6yhXus",
)
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "auto_auth.json")


def _setup_sheets_service():
    if not os.path.isfile(CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"Credentials file not found: {CREDENTIALS_PATH}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or place auto_auth.json in project root."
        )
    creds = Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return build("sheets", "v4", credentials=creds)


def _normalize_headers(raw_headers: List[Any], width: int) -> List[str]:
    out: List[str] = []
    used: Dict[str, int] = {}

    for i in range(width):
        h = str(raw_headers[i]).strip() if i < len(raw_headers) else ""
        if not h:
            h = f"Column_{i + 1}"
        base = h
        n = used.get(base, 0) + 1
        used[base] = n
        if n > 1:
            h = f"{base}_{n}"
        out.append(h)
    return out


def load_second_tab_df() -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Reads only the 2nd tab (index 1) from SOURCE_SPREADSHEET_ID.
    Returns:
      - df: pandas DataFrame (header row inferred from first row in tab)
      - details: metadata and quick diagnostics
    """
    service = _setup_sheets_service()

    meta = service.spreadsheets().get(spreadsheetId=SOURCE_SPREADSHEET_ID).execute()
    sheets = meta.get("sheets", [])
    if len(sheets) < 2:
        raise ValueError(
            f"Spreadsheet has {len(sheets)} tab(s); cannot load 2nd tab."
        )

    tab_props = sheets[1].get("properties", {})
    tab_name = tab_props.get("title", "")
    tab_id = tab_props.get("sheetId")
    tab_index = tab_props.get("index", 1)

    range_name = f"'{tab_name}'!A:Z"
    resp = (
        service.spreadsheets()
        .values()
        .get(
            spreadsheetId=SOURCE_SPREADSHEET_ID,
            range=range_name,
            valueRenderOption="UNFORMATTED_VALUE",
            dateTimeRenderOption="SERIAL_NUMBER",
        )
        .execute()
    )
    values = resp.get("values", [])

    if not values:
        df = pd.DataFrame()
        details = {
            "spreadsheet_id": SOURCE_SPREADSHEET_ID,
            "tab_name": tab_name,
            "tab_id": tab_id,
            "tab_index": tab_index,
            "rows_total": 0,
            "columns_total": 0,
            "note": "2nd tab is empty",
        }
        print(details)
        return df, details

    width = max(len(r) for r in values)
    headers = _normalize_headers(values[0], width)
    rows = []
    for r in values[1:]:
        padded = list(r) + [""] * (width - len(r))
        rows.append(padded[:width])

    df = pd.DataFrame(rows, columns=headers)

    non_empty_rows = int((df.astype(str).apply(lambda x: x.str.strip()).ne("").any(axis=1)).sum())
    details = {
        "spreadsheet_id": SOURCE_SPREADSHEET_ID,
        "tab_name": tab_name,
        "tab_id": tab_id,
        "tab_index": tab_index,
        "rows_total_excluding_header": int(df.shape[0]),
        "rows_non_empty_excluding_header": non_empty_rows,
        "columns_total": int(df.shape[1]),
        "columns": list(df.columns),
    }

    print("Second tab loaded:")
    print(details)
    print("\nPreview:")
    print(df.head(10))
    return df, details


if __name__ == "__main__":
    load_second_tab_df()
