"""
nhl_predict_current.py

Scores the live NHL season with the v2 Hart models and writes:
  Data/Processed/nhl_hart_predictions_v2_{season}.csv   (full board; the workflow copies
                                                         it to pages/ for the Google Sheet)
  Data/Processed/nhl_hart_top5_flat_{season}.csv        (one row per run, ranks 1–5)
  Data/Processed/nhl_hart_6to10_flat_{season}.csv       (ranks 6–10)

Pipeline
1. Hockey-Reference skaters + goalies, one row per player (hart_common: traded
   players keep their season-total row under their current team; goalies come
   from the goalies page only).
2. Team record from the standings → team_gp, team_win_pct, team_pts_pct.
3. Eligibility: played ≥ MIN_TEAM_GAME_SHARE of his team's games (works from
   the first week; the old GP >= 20 gate emptied the board until late November).
4. Counting features projected to an 82-game pace before scoring — the models
   were trained on full seasons. Displayed stats stay the actual season-to-date values.
5. hart_common.hart_scores → Hart_FinalScore → Hart_Rank.

6. publish/board.py → predictions/{season}/ weekly board, movement, headshot
   candidates, chips/Whys, board + bubble graphics, carousel caption.

Run:
    python Models/nhl_predict_current.py
    python Models/nhl_predict_current.py --outdir <scratch> --no-publish   # dry run
"""

import argparse
import sys
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from hart_common import (
    add_team_context, collapse_multi_team, engineer_features, feature_frame, fix_name_encoding,
    hart_scores, parse_standings, project_to_pace, team_abbr,
)
from config import (
    DATA_PROCESSED, MIN_TEAM_GAME_SHARE, MODEL_REG_PATH, MODEL_TOP5_PATH, SEASON_END_YEAR,
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR = DATA_PROCESSED

# Columns of nhl_hart_predictions_v2_{season}.csv, in order. The workflow copies it
# to pages/nhl_hart_predictions_v2_latest.csv, which the website's Google Sheet
# IMPORTDATAs — a contract: don't add, drop or reorder without switching the Sheet.
PAGES_COLUMNS = [
    "Rk", "Player", "Age", "Team", "Pos", "GP", "G", "A", "PTS", "+/-", "PIM", "EVG", "PPG", "SHG",
    "GWG", "EV", "PP", "SH", "SOG", "SPCT", "TSA", "TOI", "ATOI", "FOW", "FOL", "FO%", "BLK", "HIT",
    "TAKE", "GIVE", "Awards", "GS", "W", "L", "T/O", "GA", "Shots", "SV", "SV%", "GAA", "SO", "MIN",
    "QS", "QS%", "RBS", "GA%-", "GSAA", "GAA/A", "GPS", "Season", "Player_clean", "PTS_team",
    "W_team", "L_team", "is_goalie", "team_win_pct", "G_per_GP", "A_per_GP", "PTS_per_GP",
    "SOG_per_GP", "PPG_per_GP", "SHG_per_GP", "TOI_min", "ATOI_min", "model_run_ts",
    "model_run_label", "Hart_VoteShare_Pred", "Hart_Top5_Prob", "Hart_Win_Prob",
    "Hart_VoteShare_Scaled", "Hart_FinalScore", "Hart_Rank",
]


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


# ============================================================================
# SCRAPERS — CURRENT SEASON
# ============================================================================


def _clean_player_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop header/league-average rows, fix names, map teams, one row per player."""
    df = _normalize_team_column(df)
    df = df[df["Player"].notna() & (df["Player"].astype(str).str.strip().str.lower() != "player")]
    df = df[~df["Player"].astype(str).str.lower().str.contains("league|lgavg|average")]
    df["Player"] = (
        df["Player"].apply(fix_name_encoding).astype(str).str.replace("*", "", regex=False).str.strip()
    )
    df["Team"] = df["Team"].map(team_abbr)
    return collapse_multi_team(df)


def get_current_skaters(season_year: int) -> pd.DataFrame:
    """
    Current-season skater stats from Hockey-Reference (season end year, e.g. 2027
    for 2026–27). Goalies are dropped here; their row comes from the goalies page.
    """
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_skaters.html"
    tables = pd.read_html(StringIO(fetch_html(url)))

    chosen = None
    for idx, t in enumerate(tables):
        df = dedupe_columns(flatten_columns(t))
        cols_lower = {str(c).strip().lower() for c in df.columns}
        if {"player", "pts", "g", "a"} <= cols_lower and cols_lower & {"team", "tm"}:
            print(f"[DEBUG] Skater table candidate {idx} for {season_year}: {list(df.columns)}")
            chosen = df
            break

    if chosen is None:
        raise ValueError(
            f"[ERROR] Could not find a skater stats table with Player/G/A/PTS for {season_year}"
        )

    df = _clean_player_table(chosen.copy())
    if "Pos" not in df.columns:
        df["Pos"] = ""
    return df[df["Pos"] != "G"].copy()


def get_current_goalies(season_year: int) -> pd.DataFrame:
    """Fetch current-season goalie stats from Hockey-Reference."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_goalies.html"
    tables = pd.read_html(StringIO(fetch_html(url)))
    if not tables:
        raise ValueError(f"[ERROR] No goalie tables found for {season_year}")

    df = dedupe_columns(flatten_columns(tables[0]))
    if "Player" not in df.columns:
        raise ValueError(f"[ERROR] Goalie table missing 'Player' column for {season_year}")

    df = _clean_player_table(df)
    df["Pos"] = "G"
    return df


def get_current_standings(season_year: int) -> pd.DataFrame:
    """Team record (team_gp, W_team, L_team, OL_team, PTS_team) by abbreviation.
    Raises if the page can't be read: without team games played there is no
    eligibility or pace, and a board built without them must not be published."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_standings.html"
    print(f"[INFO] Fetching current standings from: {url}")
    st = parse_standings(pd.read_html(StringIO(fetch_html(url))))
    unmapped = [t for t in st["Team"] if len(str(t)) != 3]
    if st.empty or unmapped:
        raise ValueError(f"[ERROR] Unusable standings for {season_year} (unmapped: {unmapped})")
    return st


def get_season_calendar(season_year: int) -> tuple[date, date]:
    """(first game date, last date with a completed game) from the schedule page.
    Weeks on the board count from opening night; 'stats through' is the last game day."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_year}_games.html"
    games = dedupe_columns(flatten_columns(pd.read_html(StringIO(fetch_html(url)))[0]))
    days = pd.to_datetime(games["Date"], errors="coerce")
    played = days[pd.to_numeric(games["G"], errors="coerce").notna()]
    if days.dropna().empty or played.empty:
        raise ValueError(f"[ERROR] No played games on the {season_year} schedule")
    return days.min().date(), played.max().date()


# ============================================================================
# BUILD FULL CURRENT-SEASON DATAFRAME
# ============================================================================


def build_current_season_df(season_year: int) -> pd.DataFrame:
    """All eligible players with actual season-to-date stats and engineered features."""
    print(f"[INFO] Building current season dataset for {season_year}...")

    skaters = get_current_skaters(season_year)
    goalies = get_current_goalies(season_year)
    players = pd.concat([skaters, goalies], ignore_index=True, sort=False)
    players["Season"] = season_year
    players["Player_clean"] = players["Player"]

    players = add_team_context(players, get_current_standings(season_year))
    players = engineer_features(players)

    if "TOI" in players.columns:
        players["TOI_min"] = players["TOI"].apply(parse_toi_to_minutes)
    if "ATOI" in players.columns:
        players["ATOI_min"] = players["ATOI"].apply(parse_toi_to_minutes)

    no_team = players["team_gp"].isna()
    if no_team.any():
        print(f"[WARN] {int(no_team.sum())} player(s) without a team record dropped: "
              f"{players.loc[no_team, ['Player', 'Team']].head(10).values.tolist()}")
    players = players[~no_team & (players["team_gp"] > 0)]
    eligible = players["GP"] / players["team_gp"] >= MIN_TEAM_GAME_SHARE
    print(f"[INFO] {int(eligible.sum())} of {len(players)} players eligible "
          f"(GP >= {MIN_TEAM_GAME_SHARE:.0%} of team games)")
    return players[eligible].reset_index(drop=True)


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
        row[prefix + "pts"] = float(block.loc[i, "PTS"]) if "PTS" in block.columns else np.nan

        row[prefix + "team_win_pct"] = float(block.loc[i, "team_win_pct"]) if "team_win_pct" in block.columns else np.nan

        # 🔹 Rate stat (added back)
        row[prefix + "ppg"] = (
            float(block.loc[i, "PTS_per_GP"])
            if "PTS_per_GP" in block.columns
            else np.nan
        )

        # 🔹 Primary model stat for IG
        row[prefix + "top5_prob"] = (
            float(block.loc[i, "Hart_Top5_Prob"])
            if "Hart_Top5_Prob" in block.columns
            else np.nan
        )

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
    Both models score pace-projected features; every displayed stat is actual.
    """
    reg_model = joblib.load(MODEL_REG_PATH)
    clf_model = joblib.load(MODEL_TOP5_PATH)

    df_current = build_current_season_df(season_year)
    if df_current.empty:
        # Fail loudly — never overwrite the last good board with an empty one.
        raise SystemExit(f"ERROR: no eligible players for {season_year}; nothing written.")

    df_current["model_run_ts"] = datetime.now(timezone.utc).isoformat()
    df_current["model_run_label"] = datetime.now().strftime("As of %m/%d/%Y")

    X = feature_frame(project_to_pace(df_current))
    vote_share_pred = reg_model.predict(X)
    top5_prob = clf_model.predict_proba(X)[:, 1]
    scores = hart_scores(vote_share_pred, top5_prob, top_k_softmax)

    df_current["Hart_VoteShare_Pred"] = vote_share_pred
    df_current["Hart_Top5_Prob"] = top5_prob
    df_current["Hart_Win_Prob"] = scores["win_prob"]
    df_current["Hart_VoteShare_Scaled"] = scores["vote_share_scaled"]
    df_current["Hart_FinalScore"] = scores["final"]

    # Rank by composite score (then vote_share + points as soft tie-breakers)
    df_current = df_current.sort_values(
        ["Hart_FinalScore", "Hart_VoteShare_Pred", "PTS"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df_current["Hart_Rank"] = np.arange(1, len(df_current) + 1)

    output_path = DATA_PROCESSED_DIR / f"nhl_hart_predictions_v2_{season_year}.csv"
    print(f"[INFO][v2] Saving v2 predictions → {output_path}")
    df_current.reindex(columns=PAGES_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")

    export_flat_block(df_current, season_year, start_rank=1, end_rank=5, label="top5")
    export_flat_block(df_current, season_year, start_rank=6, end_rank=10, label="6to10")

    display_cols = ["Hart_Rank", "Player", "Team", "Pos", "GP", "team_gp", "G", "A", "PTS",
                    "team_pts_pct", "Hart_VoteShare_Pred", "Hart_Top5_Prob"]
    print("\n=========== TOP 15 HART CANDIDATES (v2) ===========")
    print(df_current[display_cols].head(15).to_string(index=False))
    print("===================================================\n")

    return df_current


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # ↑/↓ arrows and accented names on a cp1252 console
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=None,
                        help="write the weekly board here instead of predictions/{season}/")
    parser.add_argument("--no-publish", action="store_true",
                        help="board CSVs only: skip candidates, chips, graphics and caption")
    args = parser.parse_args()

    print(f"[INFO] Running Hart v2 prediction for season ending {SEASON_END_YEAR}")
    first_game, through = get_season_calendar(SEASON_END_YEAR)
    scored = predict_current_hart_v2(SEASON_END_YEAR)

    from publish.board import publish_board
    publish_board(scored, SEASON_END_YEAR, first_game, through,
                  outdir=args.outdir, publish=not args.no_publish)


if __name__ == "__main__":
    main()
