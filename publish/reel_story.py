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

Ported from nfl-awards-predictor/src/publish/reel_story.py. Players are matched by player_key, as in
.prev_board_hart.csv; a player traded since last week has a new key (his new team) and reads as NEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import REEL_MIN_CLIMB, REEL_MIN_DROP, REEL_TOP_SPOTS, TOP_N


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


def _hook(story_type: str, prev: Optional[int], cur: int) -> str:
    if story_type == "new_no1":
        return "There's a new No. 1 in our MVP model"
    if story_type == "back_to_no1":
        return "Back at No. 1"
    if story_type == "top5_climber":
        return f"Up {prev - cur} spots into the top 5"
    if story_type == "bubble_breakthrough":
        return f"From No. {prev} to No. {cur}: into the top 10"
    if story_type == "top5_drop":
        return f"Down {cur - prev} spots. What happened?"
    return "Who leads our MVP model, and why?"


def _story(story_type: str, row: pd.Series, prev: Optional[int], reason: str) -> Story:
    cur = int(row["rank"])
    return Story(story_type, row, prev, cur, _hook(story_type, prev, cur), reason)


def load_history(outdir: Path, before_week: int) -> pd.DataFrame:
    """week, player_key, player_name, rank from the season's weekly boards (board_hart_wkNN.csv, written by
    scripts/snapshot_board.py) for weeks before before_week. Empty when there are none (e.g. a replay folder)."""
    frames = []
    for f in sorted(Path(outdir).glob("board_hart_wk*.csv")):
        d = pd.read_csv(f, dtype={"player_key": str}, usecols=lambda c: c in ("week", "player_key", "player_name", "rank"))
        if len(d) and int(d["week"].iloc[0]) < before_week:
            frames.append(d.rename(columns={"week": "week"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["week", "player_key", "player_name", "rank"])


def rank_history(history: pd.DataFrame, player_key: str) -> str:
    """'Week 6: No. 1, Week 7: No. 3, ...' for one player, or '' with no history."""
    if history.empty:
        return ""
    weeks = sorted(history["week"].unique())
    mine = dict(zip(*history[history["player_key"].astype(str) == str(player_key)][["week", "rank"]].values.T)) \
        if (history["player_key"].astype(str) == str(player_key)).any() else {}
    return ", ".join(f"Week {int(w)}: " + (f"No. {int(mine[w])}" if w in mine else "not on the board") for w in weeks)


def pick_story(board: pd.DataFrame, prev_ranks: Dict[str, int], past_no1: frozenset = frozenset()) -> Story:
    """past_no1: player_keys who were No. 1 on any earlier week's board this season. A No. 1 who held the
    spot before is "back_to_no1", not "new_no1" (ported from the NFL reel)."""
    b = board.sort_values("rank").reset_index(drop=True).copy()
    b["player_key"] = b["player_key"].astype(str)
    b["prev_rank"] = b["player_key"].map(prev_ranks)
    no1 = b.iloc[0]

    def prev_of(row) -> Optional[int]:
        return None if pd.isna(row["prev_rank"]) else int(row["prev_rank"])

    # 0. No baseline (first board of the season) or nobody carried over: everyone is NEW
    if not prev_ranks or b["prev_rank"].isna().all():
        return _story("profile_no1", no1, None, "no previous board to compare with (everyone is NEW)")

    # 1. New #1, or back at #1 if he held it earlier this season
    prev_no1 = min(prev_ranks, key=prev_ranks.get)
    if prev_no1 != no1["player_key"]:
        p = prev_of(no1)
        was = f"No. {p}" if p else "off our top 30"
        if no1["player_key"] in past_no1:
            return _story("back_to_no1", no1, p,
                          f"{no1['player_name']} is back at No. 1 (was {was} last week; No. 1 earlier this season)")
        return _story("new_no1", no1, p, f"{no1['player_name']} took over No. 1 (was {was} last week)")

    known = b[b["prev_rank"].notna()].copy()
    known["delta"] = known["prev_rank"] - known["rank"]     # + = climbed

    # 2. Biggest climber who landed in the top 5
    climbers = known[(known["rank"] <= REEL_TOP_SPOTS) & (known["delta"] >= REEL_MIN_CLIMB)]
    if not climbers.empty:
        r = climbers.sort_values(["delta", "rank"], ascending=[False, True]).iloc[0]
        return _story("top5_climber", r, prev_of(r),
                      f"{r['player_name']} climbed {int(r['delta'])} spots into the top {REEL_TOP_SPOTS}")

    # 3. Bubble breakthrough: 11+ last week, top 10 now
    risers = known[(known["prev_rank"] > TOP_N) & (known["rank"] <= TOP_N)]
    if not risers.empty:
        r = risers.sort_values(["delta", "rank"], ascending=[False, True]).iloc[0]
        return _story("bubble_breakthrough", r, prev_of(r),
                      f"{r['player_name']} moved from No. {prev_of(r)} into the top {TOP_N}")

    # 4. Biggest drop by a former top-5 player (still on the board)
    drops = known[(known["prev_rank"] <= REEL_TOP_SPOTS) & (known["delta"] <= -REEL_MIN_DROP)]
    if not drops.empty:
        r = drops.sort_values(["delta", "rank"], ascending=[True, True]).iloc[0]
        return _story("top5_drop", r, prev_of(r),
                      f"{r['player_name']} fell {-int(r['delta'])} spots from the top {REEL_TOP_SPOTS}")

    # 5. Fallback
    return _story("profile_no1", no1, prev_of(no1), "quiet week at the top: no rule above matched")
