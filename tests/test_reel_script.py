"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from publish.reel_script import fallback_question, finish, problems
from publish.reel_story import Story

GOOD = {
    "beats": [
        {"scene": "hook", "text": "No. 9 to No. 2 in one week."},
        {"scene": "proof", "text": "His pace held. About a point and a half a game. So what changed? Edmonton. "
                                   "11-3-2 then. 6-0-1 since."},
        {"scene": "context", "text": "Only McDavid sits ahead of him now."},
        {"scene": "ask", "text": "McDavid or Bouchard: who finishes higher?"},
    ],
    "question": "McDavid or Bouchard: who finishes higher?",
    "instagram_caption": "McDavid or Bouchard: who finishes higher? Full board at stingercollectibles.com",
    "hashtags": ["#StingerNHL", "#HartTrophy", "#EvanBouchard", "#LetsGoOilers", "#NHLStats"],
}


def script(**changes):
    d = copy.deepcopy(GOOD)
    for scene, text in changes.items():
        next(b for b in d["beats"] if b["scene"] == scene)["text"] = text
    return d


def story(kind="top5_climber"):
    return Story(kind, pd.Series({"player_name": "Evan Bouchard", "player_key": "e"}), 9, 2, "Up 7 spots", "")


def test_good_script_passes():
    assert problems(GOOD) == []


def test_long_hook():
    assert any("hook beat" in p for p in problems(script(hook="So our model now has Evan Bouchard all the way up at No. 2.")))


def test_our_model_in_hook():
    assert any("our model" in p for p in problems(script(hook="Our model loves Bouchard now.")))


def test_ending_must_be_a_question():
    assert any("question mark" in p for p in problems(script(ask="Full board at stingercollectibles.com.")))


def test_filler_and_spoken_link():
    p = problems(script(proof="Here's what jumped out to me. I'll be honest, 37 points in 23 games.",
                        ask="Check out the full board. Who finishes higher?"))
    assert any("filler" in x and "i'll be honest" in x and "check out" in x and "full board" in x for x in p)


def test_wordy_proof():
    p = problems(script(proof="His pace held steady at about a point and a half a game all month. So what changed? "
                              "Edmonton. They were 11-3-2 then and 6-0-1 since. That's our model reacting to winning."))
    assert any("proof beat" in x for x in p)


def test_save_pct_allowed_win_pct_not():
    from publish.reel_script import probability_language
    assert probability_language("A .921 save percentage. A 92% save percentage. 92 percent save rate.") == []
    assert probability_language("An .850 win percentage.") and probability_language("A 30% chance he wins it.")


def test_finish_adds_fallback_question():
    d = finish(script(ask="That's the board this week."), story())
    assert d["beats"][-1]["text"].endswith("Is Bouchard a real Hart contender?")
    assert d["question"] == "Is Bouchard a real Hart contender?"
    assert (d["beats"][0]["start"], d["beats"][-1]["end"]) == (0, 20)


def test_fallback_questions_by_story():
    assert fallback_question(story("new_no1")) == "Is Bouchard your MVP right now?"
    assert fallback_question(story("top5_drop")) == "Is Bouchard out of the Hart race?"


def _row(**kw):
    base = {"player_name": "Evan Bouchard", "player_key": "e", "team": "EDM", "position": "D", "rank": 2,
            "games": 23, "team_games": 23, "goals": 12, "assists": 25, "points": 37, "ppg": 1.61,
            "team_record": "17-3-3", "goalie_wins": None, "save_pct": None, "gaa": None}
    base.update(kw)
    return pd.Series(base)


def test_then_now_lines():
    from publish.reel_script import then_now_lines
    then = _row(week=7, rank=9, games=16, team_games=16, goals=6, assists=17, points=23, ppg=1.44, team_record="11-3-2")
    text = "\n".join(then_now_lines(_row(), then, 8))
    assert "Week 7, 1 week ago" in text and "No. 9 -> No. 2" in text
    assert "11-3-2 -> 17-3-3 (6-0-1 since then)" in text
    assert "Points / game 1.44 -> 1.61" in text
    assert "14 points (6 goals) in 7 games" in text


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
    # Kaprizov, 2027 Week 7 -> 8 (No. 15 -> No. 3): assists a game up, goals a game down; the climb shows assists first
    then = _row(rank=15, week=7, games=20, goals=17, assists=15, points=32, ppg=1.6, team_record="13-3-4")
    now = _row(rank=3, games=23, goals=17, assists=23, points=40, ppg=1.74, team_record="16-3-4")
    assert [c[0] for c in then_now_cards(now, then)[1:]] == ["Assists / game", "Points / game"]
