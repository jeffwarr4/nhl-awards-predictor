"""
Shared Hockey-Reference cleaning + feature engineering for the Hart pipeline.

nhl_historical.py (training rows) and nhl_predict_current.py (live rows) both
build their player tables through these functions, so a feature can't be
computed one way in training and another way live.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PACE_COLS, SEASON_GAMES  # noqa: E402


# ============================================================================
# TEAMS
# ============================================================================

TEAM_NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Atlanta Thrashers": "ATL",
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
    "Phoenix Coyotes": "PHX",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VEG",   # Hockey-Reference uses VEG
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "VGK": "VEG",
}

# Season-total row for a player who played for several teams ("2TM", "3TM"; "TOT" in older pages)
MULTI_TEAM_RE = r"^(\dTM|TOT)$"


def clean_team_name(name):
    """Strip Hockey-Reference markers: '*' (made playoffs), '+', daggers, and a
    leading clinch prefix like 'x-'. Only markers — never letters inside a name."""
    if not isinstance(name, str):
        return name
    name = re.sub(r"[\*\+†‡]", "", name)
    name = re.sub(r"^[a-z]\s*-\s*", "", name.strip())
    return re.sub(r"\s+", " ", name).strip()


def team_abbr(name):
    name = clean_team_name(name)
    return TEAM_NAME_TO_ABBR.get(name, name) if isinstance(name, str) else name


# ============================================================================
# TABLE CLEANING
# ============================================================================


def fix_name_encoding(text):
    """Reverse Hockey-Reference's UTF-8-read-as-Latin-1 names."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf8")
    except Exception:
        return text


def collapse_multi_team(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player. A traded player keeps his season-total row (2TM/TOT),
    labelled with the team of his last per-team row (Hockey-Reference lists them
    in the order played), so his team record and headshot key are his current club.
    Must run on a single page's table in scraped order (Rk is per page)."""
    df = df.copy()
    key = df["Player"].astype(str) + "|" + (df["Rk"].astype(str) if "Rk" in df.columns else "")
    total = df["Team"].astype(str).str.match(MULTI_TEAM_RE)
    last_team = df[~total].groupby(key[~total])["Team"].last()
    traded = set(key[total])
    df = df[total | ~key.isin(traded)].copy()
    tot_rows = df["Team"].astype(str).str.match(MULTI_TEAM_RE)
    df.loc[tot_rows, "Team"] = key[tot_rows.index[tot_rows]].map(last_team)
    return df


def parse_standings(tables: list) -> pd.DataFrame:
    """Team record from the standings page tables → Team, team_gp, W_team, L_team,
    OL_team, PTS_team (one row per team, abbreviations)."""
    from_cols = {"gp": "team_gp", "w": "W_team", "l": "L_team",
                 "ol": "OL_team", "otl": "OL_team", "pts": "PTS_team"}
    parts = []
    for t in tables:
        cols = [str(c[-1] if isinstance(c, tuple) else c).strip() for c in t.columns]
        t = t.copy()
        t.columns = cols
        t = t.loc[:, ~t.columns.duplicated()]
        lower = {c.lower(): c for c in cols}
        team_col = lower.get("team") or lower.get("tm") or (cols[0] if cols[0].startswith("Unnamed") else None)
        if team_col is None or not {"w", "l", "pts"} <= set(lower):
            continue
        keep = {team_col: "Team"}
        keep.update({lower[k]: v for k, v in from_cols.items() if k in lower})
        parts.append(t[list(keep)].rename(columns=keep))
    if not parts:
        return pd.DataFrame(columns=["Team", "team_gp", "W_team", "L_team", "OL_team", "PTS_team"])

    st = pd.concat(parts, ignore_index=True)
    st = st[st["Team"].notna() & ~st["Team"].astype(str).str.contains("Division|Conference|League|Overall")]
    st["Team"] = st["Team"].map(team_abbr)
    for c in ["team_gp", "W_team", "L_team", "OL_team", "PTS_team"]:
        st[c] = pd.to_numeric(st[c], errors="coerce") if c in st.columns else np.nan
    st["OL_team"] = st["OL_team"].fillna(0)
    st["team_gp"] = st["team_gp"].fillna(st["W_team"] + st["L_team"] + st["OL_team"])
    st = st.dropna(subset=["W_team"]).drop_duplicates(subset=["Team"], keep="first")
    return st[["Team", "team_gp", "W_team", "L_team", "OL_team", "PTS_team"]].reset_index(drop=True)


def add_team_context(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Join the team record and derive team_win_pct (W/(W+L), the published display
    stat) and team_pts_pct (the model feature)."""
    players = players.drop(columns=[c for c in standings.columns if c != "Team" and c in players.columns])
    players = players.merge(standings, on="Team", how="left")
    players["team_win_pct"] = players["W_team"] / (players["W_team"] + players["L_team"]).replace(0, np.nan)
    players["team_pts_pct"] = players["PTS_team"] / (2 * players["team_gp"]).replace(0, np.nan)
    return players


# ============================================================================
# FEATURES
# ============================================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric coercion, per-game rates and the goalie flag. No row filtering."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in ["Age", "GP", "G", "A", "PTS", "SOG", "PPG", "SHG", "PIM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["G", "A", "PTS", "SOG", "PPG", "SHG", "PIM"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    if "PTS" not in df.columns and {"G", "A"} <= set(df.columns):
        df["PTS"] = df["G"] + df["A"]

    gp = df["GP"].replace(0, np.nan)
    for num, new in [("G", "G_per_GP"), ("A", "A_per_GP"), ("PTS", "PTS_per_GP"),
                     ("SOG", "SOG_per_GP"), ("PPG", "PPG_per_GP"), ("SHG", "SHG_per_GP")]:
        df[new] = df[num] / gp if num in df.columns else np.nan

    df["is_goalie"] = (df["Pos"] == "G").astype(int)
    if "team_pts_pct" not in df.columns:
        df["team_pts_pct"] = np.nan
    return df


def project_to_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Scale counting features to an 82-game pace by team games played. The model
    was trained on full seasons; raw partial totals rank anyone who missed games
    far too low (2025-26 replay: Kucherov 25th in mid-December on raw totals, 4th on pace)."""
    df = df.copy()
    factor = SEASON_GAMES / df["team_gp"].replace(0, np.nan)
    for c in PACE_COLS:
        df[c] = df[c] * factor
    df["GP"] = df["GP"].clip(upper=SEASON_GAMES)
    return df


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def hart_scores(vote_share_pred: np.ndarray, top5_prob: np.ndarray, top_k_softmax: int = 25) -> dict:
    """The composite used to rank the board (training backtest and live scoring):
    0.50 × min-max-scaled vote share + 0.40 × top-5 probability + 0.10 × win
    probability, where win probability is a softmax over the top_k_softmax vote
    shares (0 for everyone else). The blend beat either model alone in the
    2008–2026 leave-one-season-out backtest on mean winner rank."""
    vs = np.asarray(vote_share_pred, dtype=float)
    win = np.zeros_like(vs)
    top = np.argsort(-vs)[:top_k_softmax]
    e = np.exp(vs[top] - vs[top].max())
    win[top] = e / e.sum()
    span = np.nanmax(vs) - np.nanmin(vs)
    scaled = (vs - np.nanmin(vs)) / span if span > 0 else np.zeros_like(vs)
    return {"vote_share_scaled": scaled, "win_prob": win,
            "final": 0.50 * scaled + 0.40 * np.asarray(top5_prob, dtype=float) + 0.10 * win}
