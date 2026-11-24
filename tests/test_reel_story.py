"""Story picker rules. Run from the repo root: python -m pytest tests/"""

import pandas as pd

from publish.reel_story import pick_story, season_lows


def board(order):
    """Board in rank order from a list of player ids (rank = position + 1)."""
    return pd.DataFrame({"player_key": order, "rank": range(1, len(order) + 1),
                         "player_name": [f"Player {p}" for p in order]})


def ranks(order):
    return {p: i + 1 for i, p in enumerate(order)}


BASE = [f"p{i}" for i in range(1, 16)]          # p1..p15, last week's order


def test_first_board_is_fallback():
    s = pick_story(board(BASE), {})
    assert s.story_type == "profile_no1" and s.prev_rank is None and s.cur_rank == 1


def test_everyone_new_is_fallback():
    s = pick_story(board(BASE), {"x1": 1, "x2": 2})
    assert s.story_type == "profile_no1"


def test_new_no1():
    now = ["p3", "p1", "p2"] + BASE[3:]
    s = pick_story(board(now), ranks(BASE))
    assert (s.story_type, s.player["player_key"], s.prev_rank, s.cur_rank) == ("new_no1", "p3", 3, 1)


def test_back_to_no1():
    now = ["p3", "p1", "p2"] + BASE[3:]
    s = pick_story(board(now), ranks(BASE), past_no1=frozenset({"p3"}))
    assert (s.story_type, s.hook, s.prev_rank) == ("back_to_no1", "Back on top", 3)


def test_rank_history():
    from publish.reel_story import rank_history
    h = pd.DataFrame({"week": [6, 6, 7], "player_key": ["p1", "p2", "p2"], "rank": [1, 2, 1]})
    assert rank_history(h, "p1") == "Week 6: No. 1, Week 7: not on the board"
    assert rank_history(pd.DataFrame(columns=["week", "player_key", "rank"]), "p1") == ""


def test_new_no1_from_off_the_board():
    now = ["new"] + BASE[:14]
    s = pick_story(board(now), ranks(BASE))
    assert s.story_type == "new_no1" and s.prev_rank is None


def test_top5_climber_beats_bubble_and_drop():
    # p9 jumps 9 -> 3 (climb 6); p12 enters the top 10; p2 falls 2 -> 8
    now = ["p1", "p3", "p9", "p4", "p5", "p6", "p7", "p2", "p12", "p8", "p10", "p11", "p13", "p14", "p15"]
    s = pick_story(board(now), ranks(BASE))
    assert (s.story_type, s.player["player_key"], s.prev_rank, s.cur_rank) == ("top5_climber", "p9", 9, 3)


def test_climb_of_one_is_not_enough():
    now = ["p1", "p3", "p2"] + BASE[3:]          # p3 up 1 into the top 5
    s = pick_story(board(now), ranks(BASE))
    assert s.story_type == "profile_no1"


def test_bubble_breakthrough():
    now = BASE[:9] + ["p14", "p10", "p11", "p12", "p13", "p15"]   # p14: 14 -> 10
    s = pick_story(board(now), ranks(BASE))
    assert (s.story_type, s.prev_rank, s.cur_rank) == ("bubble_breakthrough", 14, 10)


def test_top5_drop():
    now = ["p1", "p2", "p3", "p5", "p6", "p7", "p8", "p4", "p9", "p10"] + BASE[10:]  # p4: 4 -> 8
    s = pick_story(board(now), ranks(BASE))
    assert (s.story_type, s.player["player_key"], s.prev_rank, s.cur_rank) == ("top5_drop", "p4", 4, 8)


def test_drop_off_the_board_is_skipped():
    now = [p for p in BASE if p != "p4"] + ["new"]    # p4 gone entirely (p11 slides 11 -> 10)
    s = pick_story(board(now), ranks(BASE))
    assert s.story_type != "top5_drop" and s.player["player_key"] != "p4"


def test_quiet_week_is_fallback():
    s = pick_story(board(BASE), ranks(BASE))
    assert s.story_type == "profile_no1" and s.prev_rank == 1


# ── Hooks (Nov 2026 rework): short, number first; a climb tells his season path when it's bigger ──
def test_climber_hook_without_history():
    now = ["p1", "p3", "p9", "p4", "p5", "p6", "p7", "p2", "p8", "p10"] + BASE[10:]
    s = pick_story(board(now), ranks(BASE))
    assert s.hook == "Up 6 spots"


def test_climber_hook_uses_season_path():
    now = ["p1", "p3", "p9", "p4", "p5", "p6", "p7", "p2", "p8", "p10"] + BASE[10:]
    s = pick_story(board(now), ranks(BASE), lows={"p9": (30, 3)})
    assert s.hook == "No. 30 → No. 3 in 3 weeks"


def test_path_hook_needs_a_bigger_path_and_two_weeks():
    now = ["p1", "p3", "p9", "p4", "p5", "p6", "p7", "p2", "p8", "p10"] + BASE[10:]
    assert pick_story(board(now), ranks(BASE), lows={"p9": (9, 1)}).hook == "Up 6 spots"    # same as last week
    assert pick_story(board(now), ranks(BASE), lows={"p9": (20, 1)}).hook == "Up 6 spots"   # only 1 week ago


def test_drop_hook_never_uses_the_path():
    now = ["p1", "p2", "p3", "p5", "p6", "p7", "p8", "p4", "p9", "p10"] + BASE[10:]
    s = pick_story(board(now), ranks(BASE), lows={"p4": (25, 3)})
    assert s.hook == "Down 4. What happened?"


def test_season_lows():
    h = pd.DataFrame({"week": [2, 3, 4, 2, 3], "player_key": ["a", "a", "a", "b", "b"], "rank": [30, 11, 7, 5, 5]})
    assert season_lows(h, 5) == {"a": (30, 3), "b": (5, 2)}      # a tie keeps the latest week
    assert season_lows(pd.DataFrame(columns=["week", "player_key", "rank"]), 5) == {}


# ── Then vs now (the comparison board that explains the move) ──
def test_comparison_row():
    from publish.reel_story import comparison_row
    h = pd.DataFrame({"week": [1, 2, 3, 4, 2, 4], "player_key": ["a", "a", "a", "a", "b", "b"],
                      "rank": [40, 30, 11, 7, 9, 8]})
    assert int(comparison_row(h, "a", 5)["week"]) == 2                       # earliest within 3 weeks back
    assert int(comparison_row(h, "a", 5, prefer_week=1)["week"]) == 1         # the path hook's week wins
    assert int(comparison_row(h, "a", 5, prefer_week=9)["week"]) == 2         # ...if he's on that board
    assert int(comparison_row(h, "b", 5)["week"]) == 2
    assert comparison_row(h, "c", 5) is None
    assert comparison_row(pd.DataFrame(columns=["week", "player_key", "rank"]), "a", 5) is None


def test_comparison_row_covers_the_hooks_move():
    from publish.reel_story import comparison_row
    # 2027 NHL Week 8: Kaprizov No. 7, 5, 15, then 3. "No. 15 -> No. 3" is compared with Week 7, not Week 5.
    h = pd.DataFrame({"week": [5, 6, 7], "player_key": ["k", "k", "k"], "rank": [7, 5, 15]})
    assert int(comparison_row(h, "k", 8, prev_rank=15, cur_rank=3)["week"]) == 7
    # a drop (No. 2 -> No. 9): the earliest board where he was No. 2 or better
    h = pd.DataFrame({"week": [5, 6, 7], "player_key": ["k", "k", "k"], "rank": [4, 1, 2]})
    assert int(comparison_row(h, "k", 8, prev_rank=2, cur_rank=9)["week"]) == 6
