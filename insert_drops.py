from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# === CONFIG ===
SPREADSHEET_ID = "1t21VCweRezB0MkgzXimqBKLh7mI6Ri4tEsljFG9dmcA"
SHEET_ID = 0          # from the gid in the URL
SERVICE_ACCOUNT_FILE = "auto_auth.json"  # path to your credentials

# === AUTH ===
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=scopes
)
service = build("sheets", "v4", credentials=creds)

# === DROPDOWN LISTS ===
first_list = [
    "Faceless 👤",
    "Influencer 🎭",
    "Niche Leader 👑",
    "Quality 👑",
]

second_list = [
    "Phonk",
    "Ambient",
    "Speed Garage",
]

third_list = [
    "Anime",
    "Soccer",
    "Memes",
    "Kpop",
    "Videogames",
    "Visualizers",
    "Movies / Cartoons",
    "Lyrics",
    "Celebrities (US)",
    "Motivation / Luxury",
    "Capcut Templates",
    "IA",
    "Cars",
    "Wrestling",
    "Streamers",
    "Celebrities",
    "Astronomy",
    "Reactions",
    "Gym",
    "Travel",
    "Fashion",
    "Sad",
    "Mindfulness",
    "Pixel Art",
    "Quotes",
    "Cartoons",
    "Dance",
    "BLENDER",
    "Influencer",
    "NOSTALGIA",
    "DREAMCORE",
    "Cosplayer",
]

def make_rule(values):
    return {
        "condition": {
            "type": "ONE_OF_LIST",
            "values": [{"userEnteredValue": v} for v in values],
        },
        "showCustomUi": True,
        "strict": True,
    }

requests = [
    # A5  (row 5, col A -> 0-based: row 4, col 0)
    {
        "setDataValidation": {
            "range": {
                "sheetId": SHEET_ID,
                "startRowIndex": 11,
                "endRowIndex": 12,
                "startColumnIndex": 10,
                "endColumnIndex": 11,
            },
            "rule": make_rule(first_list),
        }
    },
    # B5
    {
        "setDataValidation": {
            "range": {
                "sheetId": SHEET_ID,
                "startRowIndex": 4,
                "endRowIndex": 5,
                "startColumnIndex": 1,
                "endColumnIndex": 2,
            },
            "rule": make_rule(second_list),
        }
    },
    # C5
    {
        "setDataValidation": {
            "range": {
                "sheetId": SHEET_ID,
                "startRowIndex": 4,
                "endRowIndex": 5,
                "startColumnIndex": 2,
                "endColumnIndex": 3,
            },
            "rule": make_rule(third_list),
        }
    },
]

body = {"requests": requests}
response = service.spreadsheets().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body=body
).execute()

print("Done, dropdowns created.")
