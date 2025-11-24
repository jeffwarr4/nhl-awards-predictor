import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
# You can narrow this while testing, e.g. 2008–2010, then expand to 2024.
START_SEASON = 2008   # inclusive
END_SEASON = 2025     # inclusive

TEAM_NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VEG",  # matches your skater table output
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}


# ---------------------------------------------
# PATH SETUP (always relative to project root)
# ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_RAW_DIR / "nhl_historical_training.csv"


# ---------------------------------------------
# FIX BAD TEXT ENCODING (Hockey-Reference UTF8/LATIN1 bug)
# ---------------------------------------------
def fix_name_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        # Many corrupted names are UTF-8 decoded as Latin-1; this reverses that.
        return text.encode("latin1").decode("utf8")
    except Exception:
        return text


# ---------------------------------------------
# HTTP FETCH WITH RETRY + BACKOFF
# ---------------------------------------------
def fetch_html(url: str, max_retries: int = 8, base_delay: int = 10) -> str:
    """
    Fetch HTML with retry + exponential backoff to handle 429 (rate limiting)
    and transient 5xx server errors.
    """
    for attempt in range(max_retries):
        print(f"[DEBUG] Fetching URL (attempt {attempt + 1}/{max_retries}): {url}")
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
        except requests.RequestException as e:
            wait = base_delay * (2 ** attempt)
            print(f"[WARN] Request error {e}. Backing off {wait} seconds...")
            time.sleep(wait)
            continue

        status = r.status_code

        # Rate limited
        if status == 429:
            wait = base_delay * (2 ** attempt)
            print(f"[WARN] 429 Too Many Requests for {url}. Backing off {wait} seconds...")
            time.sleep(wait)
            continue

        # Retryable server errors
        if 500 <= status < 600:
            wait = base_delay * (2 ** attempt)
            print(f"[WARN] Server error {status} for {url}. Retrying after {wait} seconds...")
            time.sleep(wait)
            continue

        if status != 200:
            raise ValueError(f"[ERROR] Failed to fetch {url} — status {status}")

        return r.text

    raise ValueError(f"[ERROR] Failed to fetch {url} after {max_retries} retries.")


# ---------------------------------------------
# COLUMN HELPERS
# ---------------------------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten multi-level NHL column headers into simple strings.

    Examples:
      ('Unnamed: 0_level_0', 'Rk')   -> 'Rk'
      ('Scoring', 'G')              -> 'G'
      ('Goals', 'PTS')              -> 'PTS'
      ('Player', '')                -> 'Player'
    """
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Keep the LAST meaningful non-'Unnamed' label (bottom header row)
            cleaned = [str(c).strip() for c in col
                       if c and not str(c).startswith("Unnamed")]
            if cleaned:
                flat.append(cleaned[-1])   # <--- key change: use last, not first
            else:
                flat.append(str(col[-1]).strip())
        else:
            flat.append(str(col).strip())
    df.columns = flat
    return df


def _normalize_team_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'Team' column for team name / abbrev.
    Tries 'Team', 'Tm', then first 'Unnamed' column.
    """
    df = df.copy()
    cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in cols]

    team_col = None

    if "team" in cols_lower:
        team_col = cols[cols_lower.index("team")]
    elif "tm" in cols_lower:
        team_col = cols[cols_lower.index("tm")]
    elif cols and str(cols[0]).startswith("Unnamed"):
        team_col = cols[0]

    if team_col is None:
        print("[WARN] Could not find team column in df; columns:", cols)
        return df

    if team_col != "Team":
        df = df.rename(columns={team_col: "Team"})

    return df

def parse_toi_to_minutes(value):
    """
    Convert 'MM:SS' or 'HH:MM:SS' TOI strings into minutes (float).
    Returns NaN if unparsable.
    """
    import numpy as np
    if not isinstance(value, str) or ":" not in value:
        return np.nan

    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            m, s = parts
            return int(m) + int(s) / 60.0
        elif len(parts) == 3:
            h, m, s = parts
            return 60 * int(h) + int(m) + int(s) / 60.0
    except Exception:
        return np.nan

    return np.nan

# ---------------------------------------------
# GET SKATERS DATA
# ---------------------------------------------
def get_season_skaters(season_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_skaters.html"
    html = fetch_html(url)

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"[ERROR] No skater tables found for {season_year}")

    tables = pd.read_html(html)
    df = tables[0]
    df = _flatten_columns(df)
    # NEW: drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print(f"[DEBUG] Skater columns for {season_year}: {list(df.columns)}")

    if "Player" not in df.columns:
        raise ValueError(
            f"[ERROR] Skater table for {season_year} has no 'Player' column. "
            f"Columns: {list(df.columns)}"
        )

    # Drop repeated header rows inside the table
    df = df[df["Player"] != "Player"]
    df = df[df["Player"].notna()]

    df["Season"] = season_year

    # Ensure position is present
    if "Pos" not in df.columns:
        df["Pos"] = "NA"

    return df


# ---------------------------------------------
# GET GOALIES DATA
# ---------------------------------------------
def get_season_goalies(season_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_goalies.html"
    html = fetch_html(url)

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"[ERROR] No goalie tables found for {season_year}")

    tables = pd.read_html(html)
    df = tables[0]
    df = _flatten_columns(df)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print(f"[DEBUG] Goalie columns for {season_year}: {list(df.columns)}")

    if "Player" not in df.columns:
        raise ValueError(
            f"[ERROR] Goalie table for {season_year} has no 'Player' column. "
            f"Columns: {list(df.columns)}"
        )

    df = df[df["Player"] != "Player"]
    df = df[df["Player"].notna()]

    df["Season"] = season_year
    df["Pos"] = "G"

    return df


# ---------------------------------------------
# GET STANDINGS
# ---------------------------------------------
def get_season_standings(season_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_standings.html"
    html = fetch_html(url)

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"[ERROR] No standings tables found for {season_year}")

    candidates = []
    for idx, t in enumerate(tables):
        t = _flatten_columns(t)
        # drop duplicate columns
        t = t.loc[:, ~t.columns.duplicated()].copy()
        t = _normalize_team_column(t)

        cols = [str(c).strip().lower() for c in t.columns]


        # Require at least Team, W, L, PTS
        has_team = "team" in cols
        has_w = "w" in cols
        has_l = "l" in cols
        has_pts = "pts" in cols

        if has_team and has_w and has_l and has_pts:
            print(f"[DEBUG] Standings candidate table {idx} for {season_year}: {list(t.columns)}")
            candidates.append(t)

    if not candidates:
        print(
            f"[DEBUG] No usable standings tables found for {season_year}. "
            f"First table columns: {list(tables[0].columns)}"
        )
        raise ValueError(f"[ERROR] Standings table missing required columns for {season_year}")

    df = pd.concat(candidates, ignore_index=True)

    # Drop rows that are clearly not teams (like division headers or totals)
    df = df[df["Team"].notna()]
    df = df[~df["Team"].isin(["Division", "Conference", "League", "Overall"])]

    # Drop duplicate teams, keep first
    df = df.drop_duplicates(subset=["Team"], keep="first")

    # 🔹 Map full team names to abbreviations so they match skater/goalie tables
    df["Team"] = (
        df["Team"]
        .astype(str)
        .str.strip()
        .replace(TEAM_NAME_TO_ABBR)
    )

    # 🔹 Rename team points so it doesn't collide with player PTS
    if "PTS" in df.columns:
        df = df.rename(columns={"PTS": "PTS_team"})

    keep_cols = [c for c in ["Team", "PTS", "W", "L"] if c in df.columns]
    df = df[keep_cols].copy()

    
    df["Season"] = season_year
    return df



# ---------------------------------------------
# GET HART VOTING (MVP)
# ---------------------------------------------
from io import StringIO

def get_season_hart_voting(season_year: int) -> pd.DataFrame:
    """
    Fetch Hart Trophy voting for a given season and return a tidy dataframe with:
      Season, Player_clean, VotePoints, VoteTeam
    """
    url = f"https://www.hockey-reference.com/awards/voting-{season_year}.html"
    html = fetch_html(url)

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"[ERROR] No voting tables found for {season_year}")

    voting_df = None

    for i, t in enumerate(tables):
        t = _flatten_columns(t)
        # Drop duplicate columns that can appear from multi-level headers
        t = t.loc[:, ~t.columns.duplicated()].copy()

        cols_lower = [str(c).strip().lower() for c in t.columns]

        # Look for the main Hart table:
        # must have a Player column and either a Votes or PTS/points column
        if "player" in cols_lower and (
            "votes" in cols_lower or "pts" in cols_lower or "points" in cols_lower
        ):
            voting_df = t
            print(
                f"[DEBUG] Hart voting table candidate {i} for {season_year}: "
                f"{list(t.columns)}"
            )
            break

    if voting_df is None:
        raise ValueError(f"[ERROR] No Hart voting table found for {season_year}")

    # --- Normalize column names we care about ---

    # Player column
    player_col = None
    for c in voting_df.columns:
        if str(c).strip().lower() == "player":
            player_col = c
            break
    if player_col is None:
        raise ValueError(f"[ERROR] No 'Player' column in Hart voting for {season_year}")

    voting_df = voting_df[voting_df[player_col].notna()].copy()
    voting_df = voting_df.rename(columns={player_col: "Player"})

    # Fix encoding and strip asterisks
    voting_df["Player"] = voting_df["Player"].apply(fix_name_encoding)
    voting_df["Player_clean"] = (
        voting_df["Player"].astype(str).str.replace(r"\*", "", regex=True).str.strip()
    )

    # Vote points column – Hockey-Reference uses "Votes" as the point total
    vote_col = None
    for c in voting_df.columns:
        cl = str(c).strip().lower()
        if cl in ["votes", "pts", "points"]:
            vote_col = c
            break
    if vote_col is None:
        raise ValueError(
            f"[ERROR] Could not find votes/points column in Hart voting for {season_year}"
        )

    voting_df["VotePoints"] = pd.to_numeric(
        voting_df[vote_col], errors="coerce"
    ).fillna(0)

    # Team column (Tm or Team)
    team_col = None
    for c in voting_df.columns:
        cl = str(c).strip().lower()
        if cl in ["tm", "team"]:
            team_col = c
            break

    if team_col is not None:
        voting_df["VoteTeam"] = voting_df[team_col]
    else:
        voting_df["VoteTeam"] = ""

    # Add Season so build_historical_training_data can select it
    voting_df["Season"] = season_year

    # Return only what the training pipeline expects
    return voting_df[["Season", "Player_clean", "VotePoints", "VoteTeam"]].copy()

# ---------------------------------------------
# BUILD FULL TRAINING SET
# ---------------------------------------------
def build_historical_training_data():
    all_seasons = []

    for season in range(START_SEASON, END_SEASON + 1):
        print("=" * 50)
        print(f"[INFO] Building season {season}")
        print("=" * 50)

        # Players: skaters + goalies
        skaters = get_season_skaters(season)
        goalies = get_season_goalies(season)

        skaters = skaters.loc[:, ~skaters.columns.duplicated()].copy()
        goalies = goalies.loc[:, ~goalies.columns.duplicated()].copy()

        players = pd.concat([skaters, goalies], ignore_index=True, sort=False)
        players["Season"] = season

        # Normalize team column & clean player names
        players = _normalize_team_column(players)

        players["Player"] = players["Player"].apply(fix_name_encoding)
        players["Player_clean"] = (
            players["Player"]
            .str.replace(r"\*", "", regex=True)
            .str.strip()
        )

        # Flag goalies
        players["is_goalie"] = players["Pos"].eq("G").astype(int)

        # Standings
        standings = get_season_standings(season)

        # Hart voting
        voting = get_season_hart_voting(season)

        # Merge voting into players by Player + Season
        players = players.merge(
            voting[["Season", "Player_clean", "VotePoints", "VoteTeam"]],
            on=["Season", "Player_clean"],
            how="left",
        )

        # --- Remove 'League Average' or bogus rows ---
        bad_names = ["league", "lgavg", "average"]
        players = players[
            ~players["Player"].str.lower().str.contains("|".join(bad_names), na=False)
        ]

        # Drop non-player rows
        players = players[players["Player"].notna()]
        players = players[players["Player"].str.strip() != ""]



        # --- Handle players who played for multiple teams (2TM scenario) ---
        # Multi-team per player/season?
        multi_team = players.groupby(["Season", "Player_clean"])["Team"].transform("nunique") > 1
        has_2tm = players["Team"].eq("2TM")

        # If player has multiple teams and a 2TM row, keep only the 2TM row
        keep_mask = ~multi_team | (multi_team & has_2tm)
        players = players[keep_mask].copy()

        # For 2TM rows with a Hart ballot team, use that as their team
        mask_2tm_with_vote = players["Team"].eq("2TM") & players["VoteTeam"].notna()
        players.loc[mask_2tm_with_vote, "Team"] = players.loc[mask_2tm_with_vote, "VoteTeam"]

        # Merge team standings into players by Team + Season
        if "Team" in players.columns and "Team" in standings.columns:
            # Use PTS_team (renamed in get_season_standings) instead of PTS
            standings_cols = [c for c in ["Season", "Team", "PTS_team", "W", "L"] if c in standings.columns]
            players = players.merge(
                standings[standings_cols],
                on=["Season", "Team"],
                how="left",
                suffixes=("", "_team"),
            )
        else:
            print(f"[WARN] Skipping standings merge for {season} due to missing 'Team' column.")
            players["PTS_team"] = pd.NA
            players["W"] = pd.NA
            players["L"] = pd.NA


        all_seasons.append(players)

        # Season-level throttle to avoid being blocked
        print(f"[INFO] Completed season {season}. Sleeping 10 seconds before next season...")
        time.sleep(10)

        full_df = pd.concat(all_seasons, ignore_index=True)

    # ============================================================
    # Add Hart voting labels: vote_rank, vote_share, is_winner, is_top5
    # ============================================================

    # Ensure VotePoints exists and is numeric
    if "VotePoints" not in full_df.columns:
        full_df["VotePoints"] = 0
    full_df["VotePoints"] = pd.to_numeric(full_df["VotePoints"], errors="coerce").fillna(0)

    # Total Hart points per season (for vote_share)
    season_totals = full_df.groupby("Season")["VotePoints"].transform("sum")

    # vote_share: player's share of all Hart points that season
    full_df["vote_share"] = np.where(
        season_totals > 0,
        full_df["VotePoints"] / season_totals,
        0.0,
    )

    # vote_rank: rank among players WITH > 0 vote points in that season
    # (others get NaN and will NOT be treated as top5/winner)
    full_df["vote_rank"] = np.nan

    # Only rank players who actually received at least one vote
    has_votes = full_df["VotePoints"] > 0

    full_df.loc[has_votes, "vote_rank"] = (
        full_df[has_votes]
        .sort_values(["Season", "VotePoints"], ascending=[True, False])
        .groupby("Season")["VotePoints"]
        .rank(method="first", ascending=False)
    )

    # Binary labels
    full_df["is_winner"] = ((full_df["vote_rank"] == 1).astype(int))
    full_df["is_top5"] = ((full_df["vote_rank"] >= 1) & (full_df["vote_rank"] <= 5)).astype(int)

    # Any seasons where nobody had VotePoints > 0 => all zeros for winner/top5
    full_df["is_winner"] = full_df["is_winner"].fillna(0).astype(int)
    full_df["is_top5"] = full_df["is_top5"].fillna(0).astype(int)

    print(f"[INFO] Saving final dataset → {OUTPUT_FILE}")
    full_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("[DONE] Historical NHL dataset created successfully.")



# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    build_historical_training_data()
