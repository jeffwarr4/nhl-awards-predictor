"""Board stat columns, display labels and formatting shared by chips, captions and the graphics."""

from __future__ import annotations

import pandas as pd

# Columns of board_hart_latest.csv that describe performance. Goalies get their own set;
# games is context, not a standout.
SKATER_STAT_COLS = ["points", "goals", "assists", "ppg", "pp_goals", "shots", "plus_minus", "team_win_pct"]
GOALIE_STAT_COLS = ["save_pct", "gaa", "goalie_wins", "team_win_pct"]
LOWER_IS_BETTER = {"gaa"}

STAT_LABELS = {
    "points": "PTS", "goals": "G", "assists": "A", "ppg": "PTS/GP", "pp_goals": "PP Goals",
    "shots": "Shots", "plus_minus": "+/-", "team_win_pct": "Team W%",
    "save_pct": "SV%", "gaa": "GAA", "goalie_wins": "Wins", "games": "GP",
}

# Forwards are compared with forwards, defensemen with defensemen, goalies with goalies.
POSITION_GROUP = {"C": "F", "LW": "F", "RW": "F", "F": "F", "W": "F", "D": "D", "G": "G"}

# Hockey-Reference team code → stinger-assets logo filename (ESPN-named), where they differ
LOGO_CODES = {"LAK": "la"}


def is_goalie(player: pd.Series) -> bool:
    return str(player.get("position", "")) == "G"


def stat_cols(player: pd.Series) -> list[str]:
    return GOALIE_STAT_COLS if is_goalie(player) else SKATER_STAT_COLS


# The reel's labels, read aloud and on big cards. gpg / apg are per-game values derived by per_game() for the
# then-vs-now comparison.
REEL_LABELS = {**STAT_LABELS, "points": "Points", "goals": "Goals", "assists": "Assists", "ppg": "Points / game",
               "gpg": "Goals / game", "apg": "Assists / game", "save_pct": "Save %", "goalie_wins": "Wins"}

# Then-vs-now compares rates, not season totals (totals only go up).
SKATER_RATE_COLS = ["ppg", "gpg", "apg"]
GOALIE_RATE_COLS = ["save_pct", "gaa"]


def rate_cols(player: pd.Series) -> list[str]:
    return GOALIE_RATE_COLS if is_goalie(player) else SKATER_RATE_COLS


def per_game(row: pd.Series) -> pd.Series:
    """row plus gpg and apg (season goals / assists per game), for comparing two weekly boards."""
    row = row.copy()
    try:
        g = float(row["games"])
        row["gpg"] = round(float(row["goals"]) / g, 2) if g else None
        row["apg"] = round(float(row["assists"]) / g, 2) if g else None
    except (KeyError, TypeError, ValueError):
        pass
    return row


def position_group(pos) -> str:
    return POSITION_GROUP.get(str(pos), "F")


def format_stat(col: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if col in ("team_win_pct", "save_pct"):
        return f"{v:.3f}".lstrip("0") if v < 1 else "1.000"
    if col == "plus_minus":
        return f"{int(round(v)):+d}"
    if col in ("ppg", "gaa", "gpg", "apg"):
        return f"{v:.2f}"
    return f"{int(round(v)):,}"


def stat_line(player: pd.Series, cols: list[str] | None = None) -> str:
    """'PTS: 36, G: 8, ...' — only stats present for this player."""
    cols = cols or stat_cols(player)
    return ", ".join(f"{STAT_LABELS.get(c, c)}: {format_stat(c, player[c])}"
                     for c in cols if c in player and pd.notna(player[c]))
