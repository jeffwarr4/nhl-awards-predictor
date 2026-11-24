"""
Pick the one story the weekly reel tells, from this week's board and last week's ranks.

The first rule that matches wins:
  1. new_no1              the rank-1 player changed (back_to_no1 if he was No. 1 earlier this season)
  2. top5_climber         biggest climb (>= REEL_MIN_CLIMB) by a player who landed in the top REEL_TOP_SPOTS
  3. bubble_breakthrough  from rank 11+ last week into the top 10 (biggest jump)
  4. top5_drop            biggest drop (>= REEL_MIN_DROP) by a player who was top REEL_TOP_SPOTS last week
  5. profile_no1          fallback: the current #1. Always used when there is no previous board
                          (first board of a season, when everyone is NEW).

Movement is within the saved top 30 (BOARD_DEPTH): a player who wasn't on last week's board has no
previous rank, so only rules 1 and 5 can feature him; a former top-5 player who fell off the board
entirely has no row to show and is skipped by rule 4.

pick_story is pure: board rows + {player_key: previous rank} in, Story out.

Hooks are short and lead with the number, because the reel has about two seconds to stop the scroll. When a
climber's path this season is bigger than last week's move (season_lows: No. 30 three weeks ago, No. 7 last week,
No. 3 now), the hook tells the path: "No. 30 → No. 3 in 3 weeks". The renderer draws "→" as a shape (the reel
fonts have no arrow glyph).

Ported from nfl-awards-predictor/src/publish/reel_story.py, reworked with the NBA reel. Players are matched by
player_key, as in .prev_board_hart.csv; a player traded since last week has a new key (his new team) and reads as NEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from config import REEL_LOOKBACK_WEEKS, REEL_MIN_CLIMB, REEL_MIN_DROP, REEL_TOP_SPOTS, TOP_N


@dataclass
class Story:
    story_type: str
    player: pd.Series             # the featured board row
    prev_rank: Optional[int]      # None when he wasn't on last week's board (or there was none)
    cur_rank: int
    hook: str
    reason: str                   # one line on why this rule fired (printed, saved in reel_meta.json)
    labels: tuple = ("LAST WEEK", "THIS WEEK")   # over the rank, before and after it moves


def load_prev_ranks(prev_path: Path) -> Dict[str, int]:
    """{player_key: rank} from .prev_board_hart.csv; empty when there is no previous board."""
    if not Path(prev_path).exists():
        return {}
    p = pd.read_csv(prev_path, dtype={"player_key": str})
    return dict(zip(p["player_key"].astype(str), p["rank"].astype(int)))


PATH_STORIES = ("new_no1", "top5_climber", "bubble_breakthrough")   # climbs, where a longer path is the better hook


def _hook(story_type: str, prev: Optional[int], cur: int, low: Optional[Tuple[int, int]] = None) -> str:
    """low: (his worst rank on an earlier weekly board, how many weeks ago). Used for a climb when it's further
    back than last week's rank and at least 2 weeks ago."""
    if story_type in PATH_STORIES and low and prev and low[0] > prev and low[1] >= 2:
        return f"No. {low[0]} → No. {cur} in {low[1]} weeks"
    if story_type == "new_no1":
        return "New No. 1"
    if story_type == "back_to_no1":
        return "Back on top"
    if story_type == "top5_climber":
        return f"Up {prev - cur} spots"
    if story_type == "bubble_breakthrough":
        return "Into the top 10"
    if story_type == "top5_drop":
        return f"Down {cur - prev}. What happened?"
    return "Why he's No. 1"


def _story(story_type: str, row: pd.Series, prev: Optional[int], reason: str,
           lows: Optional[Dict[str, Tuple[int, int]]] = None) -> Story:
    cur = int(row["rank"])
    low = (lows or {}).get(str(row["player_key"]))
    return Story(story_type, row, prev, cur, _hook(story_type, prev, cur, low), reason)


def load_history(outdir: Path, before_week: int) -> pd.DataFrame:
    """Every row of the season's weekly boards (board_hart_wkNN.csv, written by scripts/snapshot_board.py) for weeks
    before before_week: week, player_key, player_name, rank and the stats, for the then-vs-now comparison. Empty when
    there are none (e.g. a replay folder)."""
    frames = []
    for f in sorted(Path(outdir).glob("board_hart_wk*.csv")):
        d = pd.read_csv(f, dtype={"player_key": str})
        if len(d) and int(d["week"].iloc[0]) < before_week:
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["week", "player_key", "player_name", "rank"])


def rank_history(history: pd.DataFrame, player_key: str) -> str:
    """'Week 6: No. 1, Week 7: No. 3, ...' for one player, or '' with no history."""
    if history.empty:
        return ""
    weeks = sorted(history["week"].unique())
    mine = dict(zip(*history[history["player_key"].astype(str) == str(player_key)][["week", "rank"]].values.T)) \
        if (history["player_key"].astype(str) == str(player_key)).any() else {}
    return ", ".join(f"Week {int(w)}: " + (f"No. {int(mine[w])}" if w in mine else "not on the board") for w in weeks)


def season_lows(history: pd.DataFrame, week: int) -> Dict[str, Tuple[int, int]]:
    """{player_key: (worst rank on an earlier weekly board, weeks before `week`)}; the latest week on a tie.
    Only boards he was on count (30 deep), so "off the board" never reads as a rank."""
    if history.empty:
        return {}
    h = history.assign(player_key=history["player_key"].astype(str)).sort_values(["rank", "week"])
    worst = h.groupby("player_key").tail(1)
    return {k: (int(r), int(week) - int(w)) for k, r, w in zip(worst["player_key"], worst["rank"], worst["week"])}


def comparison_row(history: pd.DataFrame, player_key: str, week: int, lookback: int = REEL_LOOKBACK_WEEKS,
                   prefer_week: Optional[int] = None, prev_rank: Optional[int] = None,
                   cur_rank: Optional[int] = None) -> Optional[pd.Series]:
    """His row on an earlier weekly board to compare with this week ("then vs now"), or None.
    prefer_week (the week of a path hook, "No. 30 → No. 3 in 3 weeks") wins when he's on that board; otherwise
    the earliest board he's on within `lookback` weeks where he was at least as far from this week's rank as he
    was last week (prev_rank -> cur_rank), so the window covers the whole move the hook names. (2027 NHL Week 8:
    Kaprizov was No. 7, 5, 15, then 3; comparing with Week 5, when he was No. 7, explained a different move than
    the hook's "No. 15 -> No. 3".) The comparison explains the move: what changed, his numbers or his team's."""
    if history.empty:
        return None
    mine = history[history["player_key"].astype(str) == str(player_key)]
    if prefer_week is not None and (mine["week"] == prefer_week).any():
        return mine[mine["week"] == prefer_week].iloc[0]
    recent = mine[mine["week"] >= week - lookback].sort_values("week")
    if prev_rank and cur_rank and prev_rank != cur_rank:
        recent = recent[recent["rank"] >= prev_rank] if cur_rank < prev_rank else recent[recent["rank"] <= prev_rank]
    return None if recent.empty else recent.iloc[0]


def pick_story(board: pd.DataFrame, prev_ranks: Dict[str, int], past_no1: frozenset = frozenset(),
               lows: Optional[Dict[str, Tuple[int, int]]] = None) -> Story:
    """past_no1: player_keys who were No. 1 on any earlier week's board this season. A No. 1 who held the
    spot before is "back_to_no1", not "new_no1" (ported from the NFL reel). lows: season_lows(), for the
    path hook."""
    b = board.sort_values("rank").reset_index(drop=True).copy()
    b["player_key"] = b["player_key"].astype(str)
    b["prev_rank"] = b["player_key"].map(prev_ranks)
    no1 = b.iloc[0]

    def prev_of(row) -> Optional[int]:
        return None if pd.isna(row["prev_rank"]) else int(row["prev_rank"])

    # 0. No baseline (first board of the season) or nobody carried over: everyone is NEW
    if not prev_ranks or b["prev_rank"].isna().all():
        return _story("profile_no1", no1, None, "no previous board to compare with (everyone is NEW)", lows)

    # 1. New #1, or back at #1 if he held it earlier this season
    prev_no1 = min(prev_ranks, key=prev_ranks.get)
    if prev_no1 != no1["player_key"]:
        p = prev_of(no1)
        was = f"No. {p}" if p else "off our top 30"
        if no1["player_key"] in past_no1:
            return _story("back_to_no1", no1, p,
                          f"{no1['player_name']} is back at No. 1 (was {was} last week; No. 1 earlier this season)", lows)
        return _story("new_no1", no1, p, f"{no1['player_name']} took over No. 1 (was {was} last week)", lows)

    known = b[b["prev_rank"].notna()].copy()
    known["delta"] = known["prev_rank"] - known["rank"]     # + = climbed

    # 2. Biggest climber who landed in the top 5
    climbers = known[(known["rank"] <= REEL_TOP_SPOTS) & (known["delta"] >= REEL_MIN_CLIMB)]
    if not climbers.empty:
        r = climbers.sort_values(["delta", "rank"], ascending=[False, True]).iloc[0]
        return _story("top5_climber", r, prev_of(r),
                      f"{r['player_name']} climbed {int(r['delta'])} spots into the top {REEL_TOP_SPOTS}", lows)

    # 3. Bubble breakthrough: 11+ last week, top 10 now
    risers = known[(known["prev_rank"] > TOP_N) & (known["rank"] <= TOP_N)]
    if not risers.empty:
        r = risers.sort_values(["delta", "rank"], ascending=[False, True]).iloc[0]
        return _story("bubble_breakthrough", r, prev_of(r),
                      f"{r['player_name']} moved from No. {prev_of(r)} into the top {TOP_N}", lows)

    # 4. Biggest drop by a former top-5 player (still on the board)
    drops = known[(known["prev_rank"] <= REEL_TOP_SPOTS) & (known["delta"] <= -REEL_MIN_DROP)]
    if not drops.empty:
        r = drops.sort_values(["delta", "rank"], ascending=[True, True]).iloc[0]
        return _story("top5_drop", r, prev_of(r),
                      f"{r['player_name']} fell {-int(r['delta'])} spots from the top {REEL_TOP_SPOTS}", lows)

    # 5. Fallback
    return _story("profile_no1", no1, prev_of(no1), "quiet week at the top: no rule above matched", lows)
