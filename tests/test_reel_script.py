"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from publish.reel_script import (
    caption_with_cta, context_problems, driver, fallback_question, finish, milestone, probability_language, problems, opener_problems,
    proof_opener, story_frame, unanchored_time,
)
from publish.reel_story import Story

NAMES = ["Connor McDavid", "Evan Bouchard", "Kirill Kaprizov", "Cutter Gauthier", "Sebastian Aho",
         "Nathan MacKinnon", "Leon Draisaitl", "Jack Eichel", "Nikita Kucherov", "Cale Makar", "David Pastrnak",
         "Auston Matthews"]
TEAMS = ["EDM", "EDM", "MIN", "ANA", "CAR", "COL", "EDM", "VEG", "TBL", "COL", "BOS", "TOR"]
BOARD = pd.DataFrame({"rank": range(1, 13), "player_name": NAMES, "team": TEAMS, "player_key": [f"k{i}" for i in range(1, 13)]})


def row(name):
    return BOARD[BOARD["player_name"] == name].iloc[0]


def story(kind="top5_climber", name="Kirill Kaprizov", prev=15):
    r = row(name)
    return Story(kind, r, prev, int(r["rank"]), "Up 12 spots", "")


GOOD = {
    "beats": [
        {"scene": "hook", "text": "Kirill Kaprizov: No. 15 to No. 3 in one week."},
        {"scene": "proof", "text": "So what changed? His playmaking. He had eight assists in his last three games, "
                                   "and Minnesota won all three."},
        {"scene": "context", "text": "Two spots from No. 1, and McDavid is the one to catch."},
        {"scene": "ask", "text": "Kaprizov or McDavid: who finishes higher?"},
    ],
    "question": "Kaprizov or McDavid: who finishes higher?",
    "instagram_caption": "Kaprizov or McDavid: who finishes higher? Full board at stingercollectibles.com",
    "hashtags": ["#StingerNHL", "#HartTrophy", "#KirillKaprizov", "#mnwild", "#NHL"],
}


def script(**changes):
    d = copy.deepcopy(GOOD)
    for scene, text in changes.items():
        next(b for b in d["beats"] if b["scene"] == scene)["text"] = text
    return d


def check(d, s=None, ranks=None):
    return problems(d, s or story(), BOARD, {"ranks": ranks or {15, 3}})


def test_good_script_passes():
    assert check(GOOD) == []


def test_v2_style_fails_the_new_checks():
    v2 = script(hook="No. 15 to No. 3 in one week.",
                proof="Eight points in three games, all assists. Assists a game .75 then. One now. Minnesota went "
                      "3-0-0 since.",
                context="He's No. 3 now, behind two Oilers.")
    p = " || ".join(check(v2, story("bubble_breakthrough", "Sebastian Aho", prev=21), {21, 5}))
    assert "full name" in p and "hook must name him" in p
    assert "Minnesota went 3-0-0 since." in p


def test_context_rules():
    s = story("bubble_breakthrough", "Nikita Kucherov", prev=14)                 # No. 9: milestone the top 5
    assert any("Connor McDavid" in x for x in context_problems("McDavid still leads.", s, BOARD, {14, 9}))
    assert any("No. 9" in x for x in context_problems("He's No. 9, four spots out.", s, BOARD, {14, 9}))
    assert context_problems("Four spots from our top five, and Aho is next.", s, BOARD, {14, 9}) == []
    assert context_problems("Only McDavid is ahead.", story("top5_climber", "Evan Bouchard", prev=4), BOARD, {4, 2}) == []


def test_time_anchor():
    assert unanchored_time("Minnesota went 3-0-0 in those three games.") == []
    assert unanchored_time("Minnesota went 3-0-0 since.") == ["Minnesota went 3-0-0 since."]


def test_word_range_and_question():
    assert any("must total" in x for x in check(script(proof="So what changed? His playmaking.")))
    assert any("question mark" in x for x in check(script(ask="That's the board.")))


def test_fallback_questions():
    assert fallback_question(story("new_no1", "Connor McDavid")) == "Is McDavid your MVP right now?"
    assert fallback_question(story("top5_drop")) == "Is Kaprizov out of the Hart race?"
    assert fallback_question(story(), BOARD) == "Can Kaprizov catch McDavid?"
    assert fallback_question(story("bubble_breakthrough", "Nikita Kucherov", prev=14), BOARD) == "Can Kucherov crack the top 5?"
    d = finish(script(ask="That's the board."), story(), BOARD)
    assert d["question"] == "Can Kaprizov catch McDavid?"


def test_milestone():
    m = milestone(story("bubble_breakthrough", "Nikita Kucherov", prev=14), BOARD)
    assert (m["target"], m["gap"], m["other"]["player_name"]) == (5, 4, "Sebastian Aho")


def _row(**kw):
    base = {"player_name": "Evan Bouchard", "player_key": "e", "team": "EDM", "position": "D", "rank": 2,
            "games": 23, "team_games": 23, "goals": 12, "assists": 25, "points": 37, "ppg": 1.61,
            "team_record": "17-3-3", "goalie_wins": None, "save_pct": None, "gaa": None}
    base.update(kw)
    return pd.Series(base)


def _kaprizov():
    # 2027 Week 7 -> 8: No. 15 -> No. 3; 8 assists, no goals in 3 games; Minnesota 13-3-4 -> 16-3-4
    then = _row(player_name="Kirill Kaprizov", team="MIN", position="LW", rank=15, week=7, games=20, team_games=20,
                goals=17, assists=15, points=32, ppg=1.6, team_record="13-3-4")
    now = _row(player_name="Kirill Kaprizov", team="MIN", position="LW", rank=3, games=23, goals=17, assists=23,
               points=40, ppg=1.74, team_record="16-3-4")
    return now, then


def test_driver_kaprizov_is_his_production():
    d = driver(*_kaprizov(), 15)
    assert d["driver"] == "player" and d["biggest"]["col"] == "apg"
    assert d["team"]["window"] == "3-0-0"


def test_driver_team_and_goalie():
    now, then = _kaprizov()
    flat = now.copy()
    flat["assists"], flat["points"], flat["goals"], flat["ppg"] = 17, 37, 20, 1.61        # rates within thresholds
    then2 = then.copy()
    then2["team_record"] = "8-8-4"                                                         # .500, then 8-0-0
    flat["team_record"] = "16-8-4"
    assert driver(flat, then2, 15)["driver"] == "team"
    g_then = _row(position="G", rank=12, week=7, games=16, goalie_wins=9, save_pct=0.905, gaa=2.80, ppg=None)
    g_now = _row(position="G", rank=4, games=19, goalie_wins=12, save_pct=0.906, gaa=2.40, ppg=None)
    d = driver(g_now, g_then, 12)
    assert d["driver"] == "player" and d["biggest"]["col"] == "gaa"      # GAA down .40: lower is better


def test_then_now_lines():
    from publish.reel_script import then_now_lines
    then = _row(week=7, rank=9, games=16, team_games=16, goals=6, assists=17, points=23, ppg=1.44, team_record="11-3-2")
    text = "\n".join(then_now_lines(_row(), then, 8))
    assert "Week 7, 1 week ago" in text and "No. 9 -> No. 2" in text
    assert "11-3-2 -> 17-3-3 (6-0-1 in the window)" in text
    assert "Points / game 1.44 -> 1.61" in text
    assert "In the window: 14 points (6 goals) in 7 games" in text


def test_then_now_goalie_and_trade():
    from publish.reel_script import then_now_lines
    g = dict(position="G", goals=0, assists=0, points=0, ppg=None)
    now = _row(**g, goalie_wins=13, save_pct=0.921, gaa=2.31)
    then = _row(**g, week=7, rank=12, games=16, goalie_wins=9, save_pct=0.905, gaa=2.80, team="CAR")
    text = "\n".join(then_now_lines(now, then, 8))
    assert "traded" in text and "Team record" not in text
    assert "Save % .905 -> .921" in text and "4 wins in 7 games" in text


def test_then_now_cards():
    from publish.render_reel import ARROW_WORD, then_now_cards
    then = _row(week=7, games=16, goals=6, assists=17, points=23, ppg=1.44, team_record="11-3-2")
    cards = then_now_cards(_row(), then)
    assert cards[0] == ("EDM record", f"11-3-2 {ARROW_WORD} 17-3-3")
    assert [c[0] for c in cards[1:]] == ["Goals / game", "Points / game"]     # biggest relative change first


def test_then_now_cards_prefer_rates_that_match_the_move():
    from publish.render_reel import then_now_cards
    now, then = _kaprizov()
    assert [c[0] for c in then_now_cards(now, then)[1:]] == ["Assists / game", "Points / game"]


def test_save_pct_allowed_win_pct_not():
    for t in ("a .915 save percentage", "A .921 save percentage.", "a 92% save percentage", "92 percent save rate"):
        assert probability_language(t) == [], t
    for t in ("An .850 win percentage.", "A 30% chance he wins it.", "he's likely the Hart winner", "the odds"):
        assert probability_language(t), t


def test_proof_opener_rotates():
    four = [proof_opener(w) for w in range(1, 5)]
    assert len(set(four)) == 4                                   # a different lead-in each week, all four in turn
    assert all(proof_opener(w) != proof_opener(w + 1) for w in range(1, 20))


def test_proof_opener_check():
    proof = next(b["text"] for b in GOOD["beats"] if b["scene"] == "proof").replace("So what changed? ", "{}")
    assert opener_problems(proof.format("Here's why. "), "Here's why.") == []
    assert opener_problems(proof.format(""), "Here's why.") == []                 # no lead-in is fine
    p = opener_problems(proof.format("So what changed? "), "Here's why.")
    assert p and "\"So what changed?\"" in p[0]
    assert opener_problems(proof.format("The reason? "), "What happened?")
    assert opener_problems("Here's why. Here's why.", "Here's why.")


def test_story_frame_carries_the_opener():
    assert story_frame(story(), BOARD, None, 11)["opener"] == proof_opener(11)


def test_caption_site_line_is_its_own_sentence():
    cta = "Full board at stingercollectibles.com."
    assert caption_with_cta("Can Dak crack the top 5? He climbed in three weeks, and Mahomes holds No. 5 — "
                            "Full board at stingercollectibles.com") == (
        "Can Dak crack the top 5? He climbed in three weeks, and Mahomes holds No. 5. " + cta)
    assert caption_with_cta("Can Aho get back into the top 3? He slid in one week, so see where he stands: Full "
                            "board at stingercollectibles.com").endswith("so see where he stands. " + cta)
    assert caption_with_cta("Can he hold No. 1? Three straight wins.") == "Can he hold No. 1? Three straight wins. " + cta


def test_finish_rewrites_the_caption_ending():
    d = copy.deepcopy(GOOD)
    d["instagram_caption"] = d["question"] + " Up seven spots: Full board at stingercollectibles.com"
    out = finish(d, story(), BOARD, {"ranks": set()})
    assert out["instagram_caption"] == d["question"] + " Up seven spots. Full board at stingercollectibles.com."
