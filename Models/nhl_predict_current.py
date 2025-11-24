"""
nhl_predict_current.py

Builds a current-season NHL player dataset, loads the trained Hart model,
and produces Hart Trophy probabilities for all players, plus a top-15
leaderboard and a CSV in Data/Processed.

Assumptions:
- Model + features from nhl_train.py are saved in:
    Models/Artifacts/hart_model.pkl
    Models/Artifacts/hart_features.json
- Project structure:
    /Data/Processed
    /Models/Artifacts
    /Models/nhl_predict_current.py
"""

import json
import re
import time
import unicodedata
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# ============================================================================
# CONFIG
# ============================================================================

# Season *ending* year. For the 2025–26 season, Hockey-Reference uses 2026.
SEASON_END_YEAR = 2026  # adjust as needed in future

# Project-relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = PROJECT_ROOT / "Models" / "Artifacts"
MODEL_PATH = ARTIFACTS_DIR / "hart_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "hart_features.json"

MODEL_REG_V2_PATH = ARTIFACTS_DIR / "hart_vote_reg_v2.pkl"
MODEL_TOP5_V2_PATH = ARTIFACTS_DIR / "hart_top5_clf_v2.pkl"
FEATURES_V2_PATH = ARTIFACTS_DIR / "hart_features_v2.json"

# ============================================================================
# TEAM NAME NORMALIZATION
# ============================================================================

TEAM_NAME_TO_ABBR = {
    # Standard names
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
    "Vegas Golden Knights": "VEG",  # Hockey-Reference uses VEG
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "Utah Mammoth": "UTA",
    "Utah Hockey Club": "UTA",

    # Weird truncated / typo variants seen on H-R
    "Calgar Flames": "CGY",
    "New Jerse Devils": "NJD",
    "New ork Islanders": "NYI",
    "New ork Rangers": "NYR",
    "Philadelphia Flers": "PHI",
    "Tampa Ba Lightning": "TBL",
    "Utah Hocke Club": "UTA",
    "NR": "NYR",  # truncated New York Rangers

    # Abbreviations as "names"
    "ANA": "ANA",
    "ARI": "ARI",
    "BOS": "BOS",
    "BUF": "BUF",
    "CGY": "CGY",
    "CAR": "CAR",
    "CHI": "CHI",
    "COL": "COL",
    "CBJ": "CBJ",
    "DAL": "DAL",
    "DET": "DET",
    "EDM": "EDM",
    "FLA": "FLA",
    "LAK": "LAK",
    "MIN": "MIN",
    "MTL": "MTL",
    "NSH": "NSH",
    "NJD": "NJD",
    "NYI": "NYI",
    "NYR": "NYR",
    "OTT": "OTT",
    "PHI": "PHI",
    "PIT": "PIT",
    "SJS": "SJS",
    "SEA": "SEA",
    "STL": "STL",
    "TBL": "TBL",
    "TOR": "TOR",
    "VAN": "VAN",
    "VEG": "VEG",
    "VGK": "VEG",
    "WSH": "WSH",
    "WPG": "WPG",
    "UTA": "UTA",
}

# ============================================================================
# HELPERS
# ============================================================================


def fetch_html(url: str, max_retries: int = 5, base_delay: int = 5) -> str:
    """Fetch HTML with basic retry logic for 429s and transient errors."""
    for attempt in range(max_retries):
        print(f"[DEBUG] Fetching URL (attempt {attempt + 1}/{max_retries}): {url}")
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
        except requests.RequestException as exc:
            wait = base_delay * (2**attempt)
            print(f"[WARN] Request error {exc}. Backing off {wait} seconds...")
            time.sleep(wait)
            continue

        if r.status_code == 429 or 500 <= r.status_code < 600:
            wait = base_delay * (2**attempt)
            print(
                f"[WARN] Status {r.status_code} for {url}. "
                f"Backing off {wait} seconds..."
            )
            time.sleep(wait)
            continue

        if r.status_code != 200:
            raise ValueError(f"[ERROR] Failed to fetch {url} — status {r.status_code}")

        return r.text

    raise ValueError(f"[ERROR] Failed to fetch {url} after {max_retries} retries.")


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column labels are unique by dropping duplicate columns."""
    df = df.copy()
    df.columns = pd.Index(map(str, df.columns))
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten multi-level NHL column headers into simple strings.
    """
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
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
    """Try to detect and rename the team column to 'Team'."""
    df = df.copy()
    cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in cols]

    team_col = None
    if "team" in cols_lower:
        team_col = cols[cols_lower.index("team")]
    elif "tm" in cols_lower:
        team_col = cols[cols_lower.index("tm")]
    else:
        # Sometimes team names are in an unnamed first column
        if cols and str(cols[0]).startswith("Unnamed"):
            team_col = cols[0]

    if team_col is None:
        return df

    if team_col != "Team":
        df = df.rename(columns={team_col: "Team"})
    return df


def fix_name_encoding(text):
    """
    Match the historical pipeline's name-fix behavior:
    - Reverse the common Hockey-Reference UTF-8/Latin-1 decoding issue.
    """
    if not isinstance(text, str):
        return text
    try:
        # Many corrupted names are UTF-8 decoded as Latin-1; this reverses that.
        return text.encode("latin1").decode("utf8")
    except Exception:
        return text


def clean_team_name(name):
    """Normalize team names by stripping clinch markers and fixing spacing."""
    if not isinstance(name, str):
        return name
    name = re.sub(r"[\*\+xXyYzZ†‡]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

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


# ============================================================================
# SCRAPERS — CURRENT SEASON
# ============================================================================


def get_current_skaters(season_year: int) -> pd.DataFrame:
    """
    Fetch current-season skater stats from Hockey-Reference for the given
    season end year (e.g., 2026 for the 2025–26 season).

    Returns a cleaned DataFrame with at least:
      Player, Team, Pos, GP, G, A, PTS, SOG, PPG, SHG, etc.
    """
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_skaters.html"
    html = fetch_html(url)
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise ValueError(f"[ERROR] No tables found on skaters page for {season_year}")

    chosen = None
    for idx, t in enumerate(tables):
        df = flatten_columns(t)
        df = dedupe_columns(df)

        cols_lower = [str(c).strip().lower() for c in df.columns]
        has_player = "player" in cols_lower
        has_team = "team" in cols_lower or "tm" in cols_lower
        has_pts = "pts" in cols_lower
        has_g = "g" in cols_lower
        has_a = "a" in cols_lower

        if has_player and has_team and has_pts and has_g and has_a:
            print(f"[DEBUG] Skater table candidate {idx} for {season_year}: {list(df.columns)}")
            chosen = df
            break

    if chosen is None:
        raise ValueError(
            f"[ERROR] Could not find a skater stats table with Player/G/A/PTS for {season_year}"
        )

    df = chosen.copy()
    df = _normalize_team_column(df)

    if "Player" in df.columns:
        df = df[df["Player"].notna()].copy()
        df = df[df["Player"].astype(str).str.strip().str.lower() != "player"]
        df["Player"] = (
            df["Player"]
            .astype(str)
            .str.replace(r"\*", "", regex=True)
            .str.strip()
            .apply(fix_name_encoding)
        )

    if "Pos" not in df.columns:
        df["Pos"] = ""

    return df


def get_current_goalies(season_year: int) -> pd.DataFrame:
    """Fetch current-season goalie stats from Hockey-Reference."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_goalies.html"

    html = fetch_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"[ERROR] No goalie tables found for {season_year}")

    df = tables[0]
    df = flatten_columns(df)
    df = dedupe_columns(df)
    df = _normalize_team_column(df)

    if "Player" not in df.columns:
        raise ValueError(f"[ERROR] Goalie table missing 'Player' column for {season_year}")

    df = df[df["Player"].notna()]
    df = df[df["Player"] != "Player"]

    bad_names = ["league", "lgavg", "average"]
    df = df[
        ~df["Player"]
        .astype(str)
        .str.lower()
        .str.contains("|".join(bad_names), na=False)
    ]

    df["Season"] = season_year
    df["Pos"] = "G"

    df["Player"] = df["Player"].apply(fix_name_encoding)
    df["Player_clean"] = (
        df["Player"].astype(str).str.replace(r"\*", "", regex=True).str.strip()
    )

    return df


def get_current_standings(season_year: int) -> pd.DataFrame:
    """
    Best-effort fetch of current-season team standings.

    Returns columns: Team, PTS, W, L, OL (if available)
    """
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_standings.html"
    print(f"[INFO] Fetching current standings from: {url}")

    try:
        html = fetch_html(url)
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[WARN] Failed to download or parse standings page: {e}")
        return pd.DataFrame(columns=["Team", "PTS", "W", "L", "OL"])

    if not tables:
        print(f"[WARN] No tables found on standings page for {season_year}")
        return pd.DataFrame(columns=["Team", "PTS", "W", "L", "OL"])

    candidates = []
    for idx, t in enumerate(tables):
        df = flatten_columns(t)
        df = dedupe_columns(df)
        df = _normalize_team_column(df)

        cols_lower = [str(c).strip().lower() for c in df.columns]
        has_team = "team" in cols_lower
        has_pts = any("pts" == cl or "points" in cl for cl in cols_lower)

        if not (has_team and has_pts):
            continue

        print(f"[DEBUG] Standings candidate table {idx} for {season_year}: {list(df.columns)}")

        # Identify columns
        team_col = "Team"
        pts_col = next(
            (c for c in df.columns if str(c).strip().lower() in ("pts", "points")), None
        )
        w_col = next((c for c in df.columns if str(c).strip().lower() == "w"), None)
        l_col = next((c for c in df.columns if str(c).strip().lower() == "l"), None)
        ol_col = next(
            (c for c in df.columns if str(c).strip().lower() in ("ol", "otl")),
            None,
        )

        if pts_col is None:
            continue

        keep_map = {"Team": team_col, "PTS": pts_col}
        if w_col is not None:
            keep_map["W"] = w_col
        if l_col is not None:
            keep_map["L"] = l_col
        if ol_col is not None:
            keep_map["OL"] = ol_col

        sub = df[list(keep_map.values())].copy()
        sub.columns = list(keep_map.keys())
        candidates.append(sub)

    if not candidates:
        print(
            f"[WARN] Could not find any usable standings tables for {season_year}. "
            f"PTS_team and team_win_pct will be NaN."
        )
        return pd.DataFrame(columns=["Team", "PTS", "W", "L", "OL"])

    standings = pd.concat(candidates, ignore_index=True)

    standings = standings[standings["Team"].notna()].copy()
    standings = standings.drop_duplicates(subset=["Team"], keep="first")

    for col in ["PTS", "W", "L", "OL"]:
        if col in standings.columns:
            standings[col] = pd.to_numeric(standings[col], errors="coerce")

    return standings

# ============================================================================
# FEATURE ENGINEERING (MATCHES TRAINING LOGIC)
# ============================================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror nhl_train.engineer_features for current-season data.

    - Filter GP >= 20
    - Compute per-game rates (G/A/PTS/SOG/PPG/SHG)
    - Compute team_win_pct
    - Ensure is_goalie flag
    - Fill obvious NaNs
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    # Filter by GP >= 20 (same as training)
    if "GP" in df.columns:
        df = df[df["GP"].fillna(0) >= 20].copy()

    # Ensure is_goalie exists
    if "is_goalie" not in df.columns:
        if "Pos" in df.columns:
            df["is_goalie"] = (df["Pos"] == "G").astype(int)
        else:
            df["is_goalie"] = 0

    # Ensure PTS (player points) exists
    if "PTS" not in df.columns and {"G", "A"}.issubset(df.columns):
        df["PTS"] = df["G"].fillna(0) + df["A"].fillna(0)

    # Compute team_win_pct
    # Prefer W_team/L_team (from standings), fallback to W/L if that's all we have
    if {"W_team", "L_team"}.issubset(df.columns):
        denom = (df["W_team"] + df["L_team"]).replace(0, np.nan)
        df["team_win_pct"] = df["W_team"] / denom
    elif {"W", "L"}.issubset(df.columns):
        denom = (df["W"] + df["L"]).replace(0, np.nan)
        df["team_win_pct"] = df["W"] / denom
    else:
        df["team_win_pct"] = np.nan

    # Per-game ratios (same as training)
    def safe_ratio(num_col, den_col, new_col):
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / df[den_col].replace(0, np.nan)
        else:
            df[new_col] = np.nan

    safe_ratio("G", "GP", "G_per_GP")
    safe_ratio("A", "GP", "A_per_GP")
    safe_ratio("PTS", "GP", "PTS_per_GP")
    safe_ratio("SOG", "GP", "SOG_per_GP")
    safe_ratio("PPG", "GP", "PPG_per_GP")
    safe_ratio("SHG", "GP", "SHG_per_GP")

    # Age numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Fill obvious NaNs with 0 for stats that are naturally 0 if missing
    for col in ["G", "A", "PTS", "PIM", "SOG", "PPG", "SHG"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Team W/L analogues for NaN filling (training used W, L)
    if "W_team" in df.columns:
        df["W_team"] = df["W_team"].fillna(0)
    if "L_team" in df.columns:
        df["L_team"] = df["L_team"].fillna(0)
    if "PTS_team" in df.columns:
        df["PTS_team"] = df["PTS_team"].fillna(0)

    # ==================================================
    # Clean TOI and ATOI into numeric minutes for reporting
    # ==================================================
    if "TOI" in df.columns:
        df["TOI_min"] = df["TOI"].apply(parse_toi_to_minutes)

    if "ATOI" in df.columns:
        df["ATOI_min"] = df["ATOI"].apply(parse_toi_to_minutes)


    return df

# ============================================================================
# BUILD FULL CURRENT-SEASON DATAFRAME
# ============================================================================


def build_current_season_df(season_year: int) -> pd.DataFrame:
    print(f"[INFO] Building current season dataset for {season_year}...")

    TEAM_FIX_MAP = {
        "Calgar Flames": "CGY",
        "New Jerse Devils": "NJD",
        "New ork Islanders": "NYI",
        "New ork Rangers": "NYR",
        "Philadelphia Flers": "PHI",
        "Tampa Ba Lightning": "TBL",
        "Utah Hocke Club": "UTA",
        "Utah Mammoth": "UTA",
    }

    def normalize_team_abbrev(team: str) -> str:
        if not isinstance(team, str):
            return team
        t = team.strip()
        # First handle weird raw strings
        if t in TEAM_FIX_MAP:
            return TEAM_FIX_MAP[t]
        # Then map full names/known variants to abbreviations
        if t in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[t]
        return t

    # Fetch player stats
    skaters = get_current_skaters(season_year).reset_index(drop=True)
    goalies = get_current_goalies(season_year).reset_index(drop=True)

    skaters = skaters.loc[:, ~skaters.columns.duplicated()].copy()
    goalies = goalies.loc[:, ~goalies.columns.duplicated()].copy()

    players = pd.concat([skaters, goalies], ignore_index=True, sort=False)

    # Normalize player team abbreviations
    players["Team"] = players["Team"].apply(normalize_team_abbrev)

    # Fetch standings
    standings = get_current_standings(season_year).copy()
    if not standings.empty:
        standings["Team"] = standings["Team"].apply(clean_team_name).apply(normalize_team_abbrev)

        # Rename standings columns to avoid collision with goalie W/L
        standings = standings.rename(
            columns={
                "PTS": "PTS_team",
                "W": "W_team",
                "L": "L_team",
            }
        )

        merge_cols = ["Team"]
        for col in ["PTS_team", "W_team", "L_team"]:
            if col in standings.columns:
                merge_cols.append(col)

        players = players.merge(
            standings[merge_cols],
            how="left",
            on="Team",
        )

    print("==== DEBUG: PLAYERS COLUMNS BEFORE FEATURE ENGINEERING ====")
    print(players.columns.tolist())

    # Feature engineering to match training logic
    players = engineer_features(players)

    return players

def export_flat_block(
    df_current: pd.DataFrame,
    season_year: int,
    start_rank: int,
    end_rank: int,
    label: str,
):
    """
    Export a flattened CSV covering ranks [start_rank, end_rank].

    Example:
      label="top5"   -> top5.csv   (r1–r5)
      label="6to10"  -> top6to10.csv  (r6–r10)

    Each row = one run_date, with columns like:
      r1_player, r1_team, ..., r5_player, ...
      or r6_player, ..., r10_player, ...
    """

    run_date = datetime.today().strftime("%Y-%m-%d")
    filename = f"nhl_hart_{label}_flat_{season_year}.csv"
    flat_path = DATA_PROCESSED_DIR / filename

    # Ensure sorted by Hart_Rank
    df_sorted = df_current.sort_values("Hart_Rank", ascending=True).reset_index(drop=True)

    # Slice the requested rank window
    # Convert ranks (1-based) to 0-based indices
    start_idx = start_rank - 1
    end_idx = end_rank - 1

    block = df_sorted[(df_sorted["Hart_Rank"] >= start_rank) &
                      (df_sorted["Hart_Rank"] <= end_rank)].copy()

    # If for some reason we don't have enough rows yet, pad logically
    block = block.reset_index(drop=True)

    row = {
        "run_date": run_date,
        "season_year": season_year,
        "label": label,
    }

    # For each row in the block, add prefixed columns
    for i in range(len(block)):
        rank = int(block.loc[i, "Hart_Rank"])
        prefix = f"r{rank}_"

        row[prefix + "player"] = block.loc[i, "Player"] if "Player" in block.columns else ""
        row[prefix + "team"] = block.loc[i, "Team"] if "Team" in block.columns else ""
        row[prefix + "pos"] = block.loc[i, "Pos"] if "Pos" in block.columns else ""
        row[prefix + "gp"] = float(block.loc[i, "GP"]) if "GP" in block.columns else np.nan
        row[prefix + "g"] = float(block.loc[i, "G"]) if "G" in block.columns else np.nan
        row[prefix + "a"] = float(block.loc[i, "A"]) if "A" in block.columns else np.nan
        row[prefix + "pts"] = float(block.loc[i, "PTS"]) if "PTS" in block.columns else np.nan

        row[prefix + "pts_team"] = float(block.loc[i, "PTS_team"]) if "PTS_team" in block.columns else np.nan
        row[prefix + "team_win_pct"] = float(block.loc[i, "team_win_pct"]) if "team_win_pct" in block.columns else np.nan

        row[prefix + "vote_share_pred"] = float(block.loc[i, "Hart_VoteShare_Pred"]) if "Hart_VoteShare_Pred" in block.columns else np.nan
        row[prefix + "top5_prob"] = float(block.loc[i, "Hart_Top5_Prob"]) if "Hart_Top5_Prob" in block.columns else np.nan
        row[prefix + "win_prob"] = float(block.loc[i, "Hart_Win_Prob"]) if "Hart_Win_Prob" in block.columns else np.nan

    new_row_df = pd.DataFrame([row])

    if flat_path.exists():
        existing = pd.read_csv(flat_path)
        # Optional: keep only latest row per run_date+label
        existing = existing[existing["run_date"] != run_date]
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_csv(flat_path, index=False, encoding="utf-8-sig")
    print(f"[INFO][v2] Saved flattened block {label} → {flat_path}")


# ============================================================================
# PREDICTION
# ============================================================================


def predict_current_hart_v2(season_year: int, top_k_softmax: int = 25):
    """
    v2 Hart prediction pipeline using:
      - Regression on vote_share (hart_vote_reg_v2.pkl)
      - Classification on is_top5 (hart_top5_clf_v2.pkl)

    Outputs:
      - Hart_VoteShare_Pred  (continuous score ~ expected vote share)
      - Hart_Top5_Prob       (probability of being a top-5 finisher)
      - Hart_Win_Prob        (softmax over top_k_softmax candidates)
    """

    print(f"[INFO][v2] Loading regression model from: {MODEL_REG_V2_PATH}")
    reg_model = joblib.load(MODEL_REG_V2_PATH)

    print(f"[INFO][v2] Loading top-5 classifier from: {MODEL_TOP5_V2_PATH}")
    clf_model = joblib.load(MODEL_TOP5_V2_PATH)

    print(f"[INFO][v2] Building current season dataset for {season_year}...")
    df_current = build_current_season_df(season_year)

    print("[DEBUG] Any McDavid rows?", df_current["Player"].str.contains("McDavid", case=False, na=False).any())
    print(df_current[df_current["Player"].str.contains("McDavid", case=False, na=False)][
    ["Player", "Team", "GP", "G", "A", "PTS", "Hart_VoteShare_Pred", "Hart_Top5_Prob"]
] if "Hart_VoteShare_Pred" in df_current.columns else "McDavid not in current DF (probably filtered by GP)")


    # The pipelines know which columns to use (they have the ColumnTransformer inside),
    # so we can just pass the full dataframe; extra columns are ignored.
    X_current = df_current.copy()

    print("[INFO][v2] Predicting vote_share (continuous Hart-ness)...")
    vote_share_pred = reg_model.predict(X_current)

    print("[INFO][v2] Predicting top-5 probabilities...")
    top5_prob = clf_model.predict_proba(X_current)[:, 1]

    df_current["Hart_VoteShare_Pred"] = vote_share_pred
    df_current["Hart_Top5_Prob"] = top5_prob

    # -------------------------
    # Derive winner probability via softmax on vote_share_pred
    # -------------------------
    # Restrict to top_k_softmax players when computing the softmax,
    # then assign zero to everyone else. This keeps the winner probability
    # focused on realistic contenders.
    df_soft = df_current.sort_values(
        "Hart_VoteShare_Pred", ascending=False
    ).head(top_k_softmax)

    scores = df_soft["Hart_VoteShare_Pred"].values
    # Guard against all zeros / negative scores
    shifted = scores - scores.max()
    exps = np.exp(shifted)
    softmax = exps / exps.sum()

    # Initialize column
    df_current["Hart_Win_Prob"] = 0.0
    df_current.loc[df_soft.index, "Hart_Win_Prob"] = softmax

        # -------------------------
    # Rank players using a composite score
    #   - Scale vote_share within season
    #   - Blend vote_share, top5_prob, and win_prob
    # -------------------------

    # 1) Min-max scale Hart_VoteShare_Pred → [0, 1] for this run
    vs = df_current["Hart_VoteShare_Pred"].values
    vs_min = np.nanmin(vs)
    vs_max = np.nanmax(vs)

    if np.isclose(vs_max, vs_min):
        # Edge case: all equal → just use zeros so ranking falls back to probs
        vote_share_scaled = np.zeros_like(vs)
    else:
        vote_share_scaled = (vs - vs_min) / (vs_max - vs_min)

    df_current["Hart_VoteShare_Scaled"] = vote_share_scaled

    # 2) Composite final score
    # You can tweak these weights later if you want:
    #   - 0.50: strength of MVP case (vote share)
    #   - 0.40: probability of finishing top 5
    #   - 0.10: probability of actually winning
    df_current["Hart_FinalScore"] = (
        0.50 * df_current["Hart_VoteShare_Scaled"]
        + 0.40 * df_current["Hart_Top5_Prob"]
        + 0.10 * df_current["Hart_Win_Prob"]
    )

    # 3) Rank by composite score (then vote_share + points as soft tie-breakers)
    df_current = df_current.sort_values(
        ["Hart_FinalScore", "Hart_VoteShare_Pred", "PTS"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_current["Hart_Rank"] = np.arange(1, len(df_current) + 1)


    # Save full CSV
    output_path = DATA_PROCESSED_DIR / f"nhl_hart_predictions_v2_{season_year}.csv"
    print(f"[INFO][v2] Saving v2 predictions → {output_path}")
    df_current.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ⭐ NEW: export flattened blocks for 1–5 and 6–10
    export_flat_block(df_current, season_year, start_rank=1, end_rank=5, label="top5")
    export_flat_block(df_current, season_year, start_rank=6, end_rank=10, label="6to10")

    # Columns to display
    preferred_cols = [
        "Hart_Rank",
        "Player",
        "Team",
        "Pos",
        "GP",
        "G",
        "A",
        "PTS",
        "PTS_team",
        "team_win_pct",
        "Hart_VoteShare_Pred",
        "Hart_Top5_Prob",
        "Hart_Win_Prob",
    ]
    display_cols = [c for c in preferred_cols if c in df_current.columns]

    print("\n=========== TOP 15 HART CANDIDATES (v2) ===========")
    print(df_current[display_cols].head(15).to_string(index=False))
    print("===================================================\n")

    return df_current


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"[INFO] Running Hart v2 prediction for season ending {SEASON_END_YEAR}")
    predict_current_hart_v2(SEASON_END_YEAR)


