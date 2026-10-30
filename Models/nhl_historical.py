"""
nhl_historical.py

Builds Data/Raw/nhl_historical_training.csv: one row per player-season
(TRAIN_START..TRAIN_END) with Hockey-Reference stats, team record and Hart
voting labels (VotePoints, vote_share, vote_rank, is_winner, is_top5).

Player rows go through hart_common, the same cleaning nhl_predict_current.py
uses, so training and live features line up.
"""

import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from hart_common import (
    add_team_context, collapse_multi_team, fix_name_encoding, parse_standings, team_abbr,
)
from config import DATA_RAW, TRAIN_END, TRAIN_START, TRAINING_CSV

START_SEASON = TRAIN_START   # inclusive
END_SEASON = TRAIN_END       # inclusive

DATA_RAW.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = TRAINING_CSV


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
                flat.append(cleaned[-1])
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


def _player_table(url: str, season_year: int, kind: str) -> pd.DataFrame:
    """First table on a skaters/goalies page, cleaned to one row per player."""
    tables = pd.read_html(StringIO(fetch_html(url)))
    if not tables:
        raise ValueError(f"[ERROR] No {kind} tables found for {season_year}")

    df = _flatten_columns(tables[0])
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _normalize_team_column(df)
    print(f"[DEBUG] {kind.title()} columns for {season_year}: {list(df.columns)}")

    if "Player" not in df.columns:
        raise ValueError(
            f"[ERROR] {kind.title()} table for {season_year} has no 'Player' column. "
            f"Columns: {list(df.columns)}"
        )

    # Drop repeated header rows and league-average rows inside the table
    df = df[df["Player"].notna() & (df["Player"] != "Player")]
    df = df[~df["Player"].astype(str).str.lower().str.contains("league|lgavg|average")]
    df["Player"] = df["Player"].apply(fix_name_encoding).astype(str).str.replace("*", "", regex=False).str.strip()
    df["Team"] = df["Team"].map(team_abbr)
    return collapse_multi_team(df)


# ---------------------------------------------
# GET SKATERS DATA
# ---------------------------------------------
def get_season_skaters(season_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_skaters.html"
    df = _player_table(url, season_year, "skater")
    if "Pos" not in df.columns:
        df["Pos"] = "NA"
    # The skaters page also lists goalies (TOI, points, no W/SV%); their row comes from the goalies page
    return df[df["Pos"] != "G"].copy()


# ---------------------------------------------
# GET GOALIES DATA
# ---------------------------------------------
def get_season_goalies(season_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_goalies.html"
    df = _player_table(url, season_year, "goalie")
    df["Pos"] = "G"
    return df


# ---------------------------------------------
# GET STANDINGS
# ---------------------------------------------
def get_season_standings(season_year: int) -> pd.DataFrame:
    """Team record (team_gp, W_team, L_team, OL_team, PTS_team) by team abbreviation.
    Playoff teams are marked '*' on Hockey-Reference; parse_standings strips it
    (an earlier version didn't, so every playoff team lost its record)."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_standings.html"
    st = parse_standings(pd.read_html(StringIO(fetch_html(url))))
    unmapped = [t for t in st["Team"] if len(str(t)) != 3]
    if unmapped:
        raise ValueError(f"[ERROR] Unmapped standings teams for {season_year}: {unmapped}")
    print(f"[DEBUG] Standings for {season_year}: {len(st)} teams")
    return st


# ---------------------------------------------
# GET HART VOTING (MVP)
# ---------------------------------------------
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
def build_season(season: int) -> pd.DataFrame:
    skaters = get_season_skaters(season)
    goalies = get_season_goalies(season)
    players = pd.concat([skaters, goalies], ignore_index=True, sort=False)
    players["Season"] = season
    players["Player_clean"] = players["Player"]
    players["is_goalie"] = players["Pos"].eq("G").astype(int)

    players = add_team_context(players, get_season_standings(season))

    # Hart voting by name. If two players share a name that season (two Sebastian
    # Ahos in 2019), the votes stay only with the one on the ballot team, else
    # the higher scorer.
    voting = get_season_hart_voting(season)
    players = players.merge(voting[["Player_clean", "VotePoints", "VoteTeam"]],
                            on="Player_clean", how="left")
    dup = players["VotePoints"].notna() & players.duplicated("Player_clean", keep=False)
    if dup.any():
        same_team = players["Team"].eq(players["VoteTeam"])
        pick = (players[dup].assign(_m=same_team[dup], _p=pd.to_numeric(players.loc[dup, "PTS"], errors="coerce"))
                .sort_values(["_m", "_p"], ascending=False).drop_duplicates("Player_clean").index)
        players.loc[dup & ~players.index.isin(pick), ["VotePoints", "VoteTeam"]] = np.nan

    if "TOI" in players.columns:
        players["TOI_min"] = players["TOI"].apply(parse_toi_to_minutes)
    if "ATOI" in players.columns:
        players["ATOI_min"] = players["ATOI"].apply(parse_toi_to_minutes)

    missing = players.loc[players["VotePoints"].gt(0) & players["team_gp"].isna(), "Player"].tolist()
    if missing:
        print(f"[WARN] {season}: vote-getters without a team record: {missing}")
    return players


def build_historical_training_data():
    all_seasons = []

    for season in range(START_SEASON, END_SEASON + 1):
        print("=" * 50)
        print(f"[INFO] Building season {season}")
        print("=" * 50)
        all_seasons.append(build_season(season))

        # Season-level throttle to avoid being blocked
        print(f"[INFO] Completed season {season}. Sleeping 10 seconds before next season...")
        time.sleep(10)

    full_df = pd.concat(all_seasons, ignore_index=True)

    # ============================================================
    # Add Hart voting labels: vote_rank, vote_share, is_winner, is_top5
    # ============================================================

    full_df["VotePoints"] = pd.to_numeric(full_df["VotePoints"], errors="coerce").fillna(0)

    # vote_share: player's share of all Hart points that season
    season_totals = full_df.groupby("Season")["VotePoints"].transform("sum")
    full_df["vote_share"] = np.where(season_totals > 0, full_df["VotePoints"] / season_totals, 0.0)

    # vote_rank: rank among players WITH > 0 vote points in that season
    # (others get NaN and will NOT be treated as top5/winner)
    full_df["vote_rank"] = np.nan
    has_votes = full_df["VotePoints"] > 0
    full_df.loc[has_votes, "vote_rank"] = (
        full_df[has_votes]
        .sort_values(["Season", "VotePoints"], ascending=[True, False])
        .groupby("Season")["VotePoints"]
        .rank(method="first", ascending=False)
    )

    full_df["is_winner"] = (full_df["vote_rank"] == 1).astype(int)
    full_df["is_top5"] = full_df["vote_rank"].between(1, 5).astype(int)

    print(f"[INFO] Saving final dataset → {OUTPUT_FILE}")
    full_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("[DONE] Historical NHL dataset created successfully.")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    build_historical_training_data()
