"""Story picker rules. Run from the repo root: python -m pytest tests/"""

import pandas as pd

from publish.reel_story import pick_story


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
    assert (s.story_type, s.hook, s.prev_rank) == ("back_to_no1", "Back at No. 1", 3)


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
