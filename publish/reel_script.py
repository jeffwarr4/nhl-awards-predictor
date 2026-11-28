"""
Narration script for the weekly Hart Trophy reel, written for Jeff to voice over in Instagram (first person, his
voice), split into the four scene beats of config.REEL_SCENES, plus the on-screen comment question, an Instagram
caption and hashtags.

Reworked Nov 2026 for retention (v2: hook first, ends on a comment question) and tightened (v3) so it flows: the hook
names the player, the whole script uses ONE time window, the proof gives ONE reason (after a lead-in that rotates week to week), and the
context is the gap to the milestone the question is about. story_frame() works out the window, what drove the move
(his own numbers, his team, or unclear) and the milestone, so the script doesn't have to guess.

Uses REEL_MODEL (Opus 5.5) rather than the chips' Haiku: one call a week, and the voice matters.
Skipped with a warning when ANTHROPIC_API_KEY is not set (the reel still renders, with a fallback question).

Hard rules, checked in code by problems() (one rewrite request; if the ending still isn't a question, a
fallback question about the milestone is used):
  - REEL_WORDS words across the four beats; hook <= REEL_HOOK_MAX_WORDS, proof <= REEL_PROOF_MAX_WORDS,
    ask <= REEL_ASK_MAX_WORDS; the last beat ends with "?"; the on-screen question is 10 words or fewer
  - his full name (or REEL_NICKNAMES short name) in the beats, and his name in the hook
  - no "then" / "since" / "now" without a time anchor in the same sentence
  - the context doesn't repeat the hook's ranks, or name the No. 1 unless the story is about No. 1
  - no filler or stock phrases, no "my model", no "our model" in the hook, no spoken URL
  - no Hart probability / % / odds / betting language (a save percentage is fine)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import List, Optional

import pandas as pd

from config import (
    REEL_ASK_MAX_WORDS, REEL_FLAT_STATS, REEL_HOOK_MAX_WORDS, REEL_MODEL, REEL_NICKNAMES,
    REEL_PROOF_MAX_WORDS, REEL_SCENES, REEL_TEAM_SWING, REEL_WORDS, REEL_WPS,
)
from publish.generate_captions import TEAM_HASHTAGS
from publish.reel_story import Story
from publish.stats import REEL_LABELS as STAT_LABELS, format_stat, is_goalie, per_game, rate_cols, stat_line

CTA_LINE = "Full board at stingercollectibles.com"     # the caption's last sentence, added in code; never spoken
# The lead-in to the proof's reason rotates by week (one per week, offset per sport) so no phrase goes stale.
PROOF_OPENERS = ["So what changed?", "Here's why.", "The reason?", "What happened?"]
_OPENER_RE = {o: re.compile(p, re.IGNORECASE) for o, p in zip(PROOF_OPENERS, [
    r"\bso,? what changed\b", r"\bhere'?s why\b", r"\bthe reason\s*\?", r"\bwhat happened\s*\?"])}
QUESTION_MAX_WORDS = 10
NO1_STORIES = ("new_no1", "back_to_no1", "profile_no1")

# ── Sport-specific: records, the rates that say "his own numbers moved", names, facts ────────────
SPORT = "NHL"
MVP = "Hart Trophy"
LOWER_IS_BETTER = {"gaa"}


def key_rates(p: pd.Series) -> List[str]:
    """The per-game rates whose change says "his own numbers moved": points/goals/assists a game, or a goalie's
    save % and GAA (stats.rate_cols)."""
    return [c for c in rate_cols(p) if c in REEL_FLAT_STATS]


def prepare(row: pd.Series) -> pd.Series:
    """A board row ready for the then-vs-now comparison: plus goals and assists a game (stats.per_game)."""
    return per_game(row)


def team_record(p: pd.Series) -> str:
    return str(p["team_record"])


def team_rate(record: str) -> Optional[tuple]:
    """(points, points possible) from a W-L-OTL record, so the team's "rate" is its points percentage."""
    try:
        w, l, o = (int(x) for x in str(record).split("-")[:3])
        return 2 * w + o, 2 * (w + l + o)
    except ValueError:
        return None


def window_record(then_rec: str, now_rec: str) -> Optional[str]:
    """The record over the window ("6-0-1"), from the two season records."""
    try:
        a = [int(x) for x in str(then_rec).split("-")]
        b = [int(x) for x in str(now_rec).split("-")]
        d = [y - x for x, y in zip(a, b)]
        return "-".join(map(str, d)) if all(x >= 0 for x in d) else None
    except ValueError:
        return None


def spoken_name(p: pd.Series) -> str:
    """What the fallback question calls him: his short name, else the last name ("McDavid", "MacKinnon")."""
    parts = _name_parts(p)
    return nickname(p) or (parts[-1] if parts else str(p["player_name"]))


SPORT_FACTS = ("Team records are W-L-OTL (e.g. 17-3-3 is 17 wins, 3 regulation losses, 3 overtime losses); say "
               "them as given (\"17-3-3\"), and don't call a record perfect or undefeated unless it has no losses. "
               "Don't claim division, conference or playoff positions. Save percentage is given as a decimal (.921): "
               "say it the way fans do (\"a .921 save percentage\"). A save percentage is fine; a win percentage is "
               "not (say the record), and no other percent of anything. Write ranks as \"No. 4\".")

# What the model weighs, in words the script can say (Models/hart_common.hart_scores: half a vote-share model, 0.4 a
# top-5 model, 0.1 a win softmax; both models are trained on past Hart voting with scoring rates (points, goals,
# assists, power-play goals and shots per game), team points percentage, age and position; counting stats are
# projected to an 82-game pace).
MODEL_NOTE = ("How our model ranks Hart Trophy candidates: it learned from past Hart voting. Mostly the player's "
              "scoring pace (points, goals and assists per game, power-play goals, shots), weighed with his team's "
              "points percentage. Counting stats are projected to a full 82-game season, so a missed game doesn't "
              "sink a player. Goalies rarely rank high.")

APPROVED_TAGS = ("#NHL #NHLTwitter #HockeyTwitter #Hockey #NHL{season_tag} #NHLSeason "
                 "#HartTrophy #HartWatch #NHLMVP #MVPRace #NHLAwards #NHLStats #StingerNHL #StingerSports")
OPENER_OFFSET = 2                                   # PROOF_OPENERS rotation, per sport
STINGER_TAG = "#StingerNHL"
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# A percentage is fine when it's a shooting/completion/strikeout/save stat ("60% from the field", "a 68%
# completion rate", "K%"); those spans are removed before FORBIDDEN runs. Anything else with % or "percent" is
# treated as a probability, and win % is always out (say the record instead).
STAT_PCT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent)\s+(?:from (?:the field|the floor|three|deep|downtown|the line|beyond the arc)"
    r"|on (?:threes|3s|free throws|his shots|his passes)|shooting|of his (?:passes|shots|throws|attempts)"
    r"|(?:completion|save)s?(?:\s+(?:percentage|rate))?|strikeouts?|K\b)"
    r"|\b(?:shooting|shoots|shot|hitting|hits|making|completing|completes|completed|converting)\s+(?:a |at |just |over |under |nearly |about )?"
    r"\d+(?:\.\d+)?\s*(?:%|percent)"
    r"|\b(?:field[- ]goal|three[- ]point|3-point|free[- ]throw|completion|shooting|save|strikeout|FG|3P|3PT|FT|Comp|K|SV)\s*"
    r"(?:%|percentage|percent)",
    re.IGNORECASE,
)

# Probability, odds, betting and win-% words. Word-bounded so "better" or "between" don't trip it.
FORBIDDEN = re.compile(
    r"%|\bpercent|\bprobab|\bodds\b|\bchances?\b|\blikel(y|ihood)\b|\bfavou?rites?\b|\bbet(s|ting)?\b"
    r"|\bwager|\bparlay|\bsportsbook|\bline(s)? (move|movement)|\blongshot|\block\b|\bpayout",
    re.IGNORECASE,
)
WIN_PCT = re.compile(r"\bwin(?:ning)?\s*(?:%|percent|percentage|pct)", re.IGNORECASE)


def probability_language(text: str) -> List[str]:
    """Probability / odds / betting / win-% words in text, after allowing stat percentages."""
    hits = [m.group(0) for m in WIN_PCT.finditer(text)]
    hits += [m.group(0) for m in FORBIDDEN.finditer(STAT_PCT.sub(" ", WIN_PCT.sub(" ", text)))]
    return hits


# Stock phrases and warm-ups that cost seconds and don't say anything. Checked in the spoken beats only.
FILLER = re.compile(
    r"here'?s what (jumped|stood) out|\bi'?ll be honest\b|\blet'?s\b|\bcheck (it |this )?out\b|\bmake sure to\b"
    r"|\byou won'?t believe\b|\bmy model\b|\bstingercollectibles\b|\blink in bio\b|\bfull board\b",
    re.IGNORECASE,
)

# Time anchors. A sentence that says "then", "since" or "now" must also say when (the heuristic in problems()).
_NUMW = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|a few)"
TIME_WORD = re.compile(r"\b(then|since|now)\b", re.IGNORECASE)
ANCHOR = re.compile(
    rf"\bthis (?:week|season)\b|\blast week\b|\bright now\b|\bweek \d+\b|\ba week ago\b"
    rf"|\b{_NUMW} (?:weeks?|games?) ago\b"
    rf"|\b(?:in|over|across) (?:the )?(?:last |past )?{_NUMW} (?:weeks?|games?)\b"
    rf"|\b(?:his|the|those|these|that) (?:last |past )?(?:{_NUMW} )?(?:games?|weeks?|stretch)\b",
    re.IGNORECASE,
)
# A hook that reads like a label: "Jack Eichel: No. 15 to No. 5" (a colon, or "No." before a rank)
ROBOT_HOOK = re.compile(r":|\bNo\.\s*\d", re.IGNORECASE)
_NUM_WORDS = "one two three four five six seven eight nine ten eleven twelve".split()

SCENE_NOTES = {
    "hook":    "Hook. On screen from the very first frame: his headshot, the hook text and his rank flipping. One natural "
               "spoken sentence: his name, a verb and the move (\"Jack Eichel jumped from 15 to 5 in one week.\").",
    "proof":   "Proof. On screen: the cards listed below punch in one at a time. This week's lead-in (below), then ONE "
               "reason (the driver below), with at most two numbers in one spoken sentence.",
    "context": "Context. On screen: the top 5 this week, featured player highlighted. The gap to the milestone below.",
    "ask":     "Ask. On screen: your comment question in big type. One yes/no or two-way question about the milestone. "
               "Nothing after it.",
}

SYSTEM_PROMPT = """You write the weekly voiceover script for STINGER's {sport} {mvp} reel: a 20-second vertical video that tells ONE story from our model's {sport} {mvp} board. Jeff, who runs STINGER, records it in Instagram over the video, sometimes ad-libbing.

Write for speaking aloud: full sentences that flow into each other, never fragments or a list. Contractions. Say "our model", never "my model". Round numbers the way a person says them ("twenty-eight points", "nearly thirteen boards" for 12.6). No greeting, no hype words, no puns, no exclamation marks. Never use stock phrases: "here's what jumped out", "I'll be honest", "let's", "you won't believe", "check out", "make sure to".

Four beats, one per scene, {lo}-{hi} words in total:

1. hook ({hook_max} words or fewer): one natural sentence, the way a fan would say it out loud: his name, a verb and the move, e.g. "Jack Eichel jumped from 15 to 5 in one week." or "Brock Purdy is back on top after three weeks." Bare numbers (never "No." in the hook), no colon, no team name before his name. Use his full name, or the short name the user message gives if fans know him by it (Giannis, Wemby). That name must be in the voiceover. No "our model" in the hook.

2. proof ({proof_max} words or fewer): ONE reason. Lead into the reason with this week's lead-in from the user message, or skip the lead-in and go straight to the reason when the hook already makes it obvious. Never use a different lead-in. In a team story, one short line about his own numbers may come first ("His numbers barely moved."). At most two supporting numbers, in one connected spoken sentence. The user message says what drove the move:
   - player: his own production. Give the one number that changed most, before and after.
   - team: his numbers were roughly flat and his team's record changed. Say so plainly ("His numbers barely moved; Miami's record did.") and give the record.
   - board: he didn't play in the window (a bye or a rest), so his numbers didn't change; the players ahead of him dropped. Say so and name who dropped.
   - unclear: give the single biggest change in the input, and say what changed without claiming it's why he moved.
   Only say something caused the move when the user message says so.

3. context (about {ctx} words): the gap to the milestone in the user message (the top 10, the top 5, No. 1, or the player directly ahead), e.g. "Five spots from our top five, and Mahomes is the one to catch." Never repeat the rank from the hook. Don't name the board leader unless the user message says the milestone is No. 1.

4. ask ({ask_max} words or fewer): one yes/no or two-way question about that milestone, ending with "?" ("Can Dak crack the top 5?", "Giannis or Wemby: who finishes higher?"). Nothing after it. No website, link or "full board".

One time period for the whole script. The user message gives it ("in three weeks", "his last three games", "since Week 8"). Use it in the hook, and anchor every number to it. Never write "then", "since" or "now" without saying when in the same sentence ("since Week 8", "three weeks ago").

Facts: use ONLY the facts in the user message; never invent a stat, game, injury, streak or standing. {sport_facts} Never show or imply a probability: no win %, "percent chance", "odds", "chance", "likely", "favorite", "lock", or betting language.

Season context: if the user message gives his rank by week, use it only as given. A "back_to_no1" story means he held No. 1 earlier this season and has taken it back; never call him new to No. 1.

Style examples. THE FACTS IN THEM ARE EXAMPLES ONLY: use only the facts in the user message.
Example 1, a team-driven climb (NBA):
  hook:    "Giannis jumped from 30 to 3 in three weeks."
  proof:   "His numbers barely moved: twenty-eight points and nearly thirteen boards a night. The reason? Miami, 1-4 three weeks ago, has gone 8-2 since."
  context: "Only Luka and Wemby are ahead of him on our board."
  ask:     "Giannis or Wemby: who finishes higher?"
Example 2, a production-driven climb (NFL):
  hook:    "Dak Prescott climbed from 17 to 10 in three weeks."
  proof:   "Here's why. His last three games: twelve touchdowns, two Dallas wins, and his season rating up from 93 to 101."
  context: "Five spots from our top five, and Mahomes is the one to catch."
  ask:     "Can Dak crack the top 5?"

Also write:
- question: the comment question as on-screen text, {q_max} words or fewer, ending with "?".
- instagram_caption: 2 sentences, under 230 characters, no hashtags, ranks as "No. 4". The first sentence repeats the question; the second gives the hook fact. No website, link or "full board": the code adds "{cta}." as its own last sentence.
- hashtags: exactly 5, most important first, from the approved tags and the player/team tags only. Always include {stinger_tag}.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"scene": {"type": "string"}, "text": {"type": "string"}},
                "required": ["scene", "text"],
                "additionalProperties": False,
            },
        },
        "question": {"type": "string"},
        "instagram_caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["beats", "question", "instagram_caption", "hashtags"],
    "additionalProperties": False,
}

SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


# ── Names ─────────────────────────────────────────────────────────────────────────────────────
def fold(text: str) -> str:
    """Lowercase, accents removed ("Dončić" -> "doncic"), for name matching."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)).lower()


def _name_parts(p: pd.Series) -> List[str]:
    return [w for w in str(p["player_name"]).split() if w.lower() not in SUFFIXES]


def nickname(p: pd.Series) -> Optional[str]:
    nick = {fold(k): v for k, v in REEL_NICKNAMES.items()}
    return nick.get(fold(" ".join(_name_parts(p)))) or nick.get(fold(p["player_name"]))


def _has(text: str, phrase: str) -> bool:
    return bool(phrase) and re.search(rf"(?<![\w-]){re.escape(fold(phrase))}(?![\w-])", fold(text)) is not None


def short_name(p: pd.Series) -> str:
    """How the context/ask can refer to another player: his short name, else his last name."""
    parts = _name_parts(p)
    return nickname(p) or (parts[-1] if parts else str(p["player_name"]))


# ── The story frame: one time window, what drove the move, the milestone ─────────────────────
def _num_word(n: int) -> str:
    return _NUM_WORDS[n - 1] if 1 <= n <= len(_NUM_WORDS) else str(n)


def _rate(p: pd.Series, c: str) -> Optional[float]:
    try:
        v = float(p[c])
        return None if pd.isna(v) else v
    except (KeyError, TypeError, ValueError):
        return None


def milestone(story: Story, board: pd.DataFrame) -> dict:
    """The next milestone for the context and the ask, by his rank: hold No. 1, catch No. 1 (from No. 2-3), the top
    3 (from 4-5), the top 5 (from 6-10), the top 10 (from 11+)."""
    cur = int(story.cur_rank)
    ranked = board.sort_values("rank").reset_index(drop=True)

    def at(r):
        row = ranked[ranked["rank"] == r]
        return row.iloc[0] if len(row) else None

    if cur == 1:
        return {"kind": "hold", "target": 1, "gap": 0, "other": at(2)}
    target = 1 if cur <= 3 else 3 if cur <= 5 else 5 if cur <= 10 else 10
    return {"kind": "no1" if target == 1 else "top", "target": target, "gap": cur - target, "other": at(target)}


def leader_allowed(story: Story) -> bool:
    """The No. 1's name may be in the context when the story is about No. 1 or he's No. 2-3 (No. 1 is the milestone)."""
    return story.story_type in NO1_STORIES or int(story.cur_rank) <= 3


def driver(p: pd.Series, then: Optional[pd.Series], start_rank: Optional[int]) -> dict:
    """What drove the move over the window: 'player' (a key per-game stat moved at least its REEL_FLAT_STATS amount
    with his rank), 'team' (else his team's win rate in the window is at least REEL_TEAM_SWING above or below its rate
    before the window, over 3+ team games, the same way as his rank) or 'unclear'."""
    out = {"driver": "unclear", "rates": [], "team": None}
    if then is None:
        return out
    p, then = prepare(p), prepare(then)
    cur = int(p["rank"])
    way = 0 if start_rank is None or start_rank == cur else (1 if cur < start_rank else -1)   # 1 = a climb
    rates = []
    for c in key_rates(p):
        a, b = _rate(then, c), _rate(p, c)
        if a is None or b is None:
            continue
        rel = (b - a) / max(abs(a), 1e-9)
        good = -rel if c in LOWER_IS_BETTER else rel                  # > 0: better
        rates.append({"col": c, "then": a, "now": b, "rel": rel, "good": good})
    out["rates"] = rates
    a, b = team_rate(team_record(then)), team_rate(team_record(p))
    if a and b and str(then.get("team")) == str(p.get("team")) and b[1] > a[1]:
        before = a[0] / a[1] if a[1] else None
        in_win = (b[0] - a[0]) / (b[1] - a[1])
        swing = None if before is None else in_win - before
        out["team"] = {"then": team_record(then), "now": team_record(p), "window": window_record(team_record(then), team_record(p)),
                       "games": b[1] - a[1], "swing": swing}
    moved = [r for r in rates if abs(r["now"] - r["then"]) >= REEL_FLAT_STATS[r["col"]]
             and (way == 0 or (r["good"] > 0) == (way > 0))]
    t = out["team"]
    if moved:
        out["driver"] = "player"
        out["biggest"] = max(moved, key=lambda r: abs(r["now"] - r["then"]) / REEL_FLAT_STATS[r["col"]])   # vs its threshold
    elif t and t["swing"] is not None and t["games"] >= 3 and abs(t["swing"]) >= REEL_TEAM_SWING \
            and (way == 0 or (t["swing"] > 0) == (way > 0)):
        out["driver"] = "team"
    return out


def crossed(story: Story, board: pd.DataFrame, earlier: Optional[pd.DataFrame], start_rank: Optional[int]) -> List[dict]:
    """For a climb, the players ahead of him on the window-start board who are behind him now (or off the board); for a
    drop, the players behind him then who are ahead of him now. Up to 3, nearest first."""
    if earlier is None or earlier.empty or not start_rank or start_rank == story.cur_rank:
        return []
    cur, climb = int(story.cur_rank), int(story.cur_rank) < start_rank
    now = {fold(n): int(k) for n, k in zip(board["player_name"], board["rank"])}
    me = fold(story.player["player_name"])
    out = []
    for _, e in earlier.sort_values("rank").iterrows():
        n, was = fold(e["player_name"]), int(e["rank"])
        if n == me:
            continue
        is_now = now.get(n)
        if climb and was < start_rank and (is_now is None or is_now > cur):
            out.append({"name": str(e["player_name"]), "then": was, "now": is_now})
        if not climb and was > start_rank and is_now is not None and is_now < cur:
            out.append({"name": str(e["player_name"]), "then": was, "now": is_now})
    return sorted(out, key=lambda x: abs(x["then"] - start_rank))[:3]


def story_frame(story: Story, board: pd.DataFrame, then: Optional[pd.Series], week: int,
                earlier: Optional[pd.DataFrame] = None) -> dict:
    """The facts the script is built around: the window (one time period), the rank at its start, the driver, the
    players he passed (or who passed him) and the milestone. earlier: the window-start board. Also used by problems()
    (the hook's ranks) and by the fallback question."""
    p = story.player
    cur = int(story.cur_rank)
    if then is not None:
        k = week - int(then["week"])
        start_rank = int(then["rank"])
        games = None
        try:
            games = int(p["games"]) - int(then["games"])
        except (KeyError, TypeError, ValueError):
            pass
        weeks = f"{_num_word(k)} week{'s' if k != 1 else ''}"
        window = (f"the last {weeks if k != 1 else 'week'} (Week {int(then['week'])} to Week {week})"
                  + (f", his last {_num_word(games)} game{'s' if games != 1 else ''}" if games else ""))
        say = f'"in {weeks}"' + (f', "his last {_num_word(games)} games"' if games and games > 1 else "") + \
              f', "since Week {int(then["week"])}"'
        rank_line = f"Rank: No. {start_rank} in Week {int(then['week'])} -> No. {cur} now."
    else:
        start_rank = story.prev_rank
        window = "this week (last week's board to this one)"
        say = '"this week"'
        rank_line = (f"Rank: No. {start_rank} last week -> No. {cur} this week." if start_rank
                     else f"Rank: No. {cur} this week.")
    d = driver(p, then, start_rank)
    x = crossed(story, board, earlier, start_rank)
    try:
        idle = then is not None and int(p["games"]) == int(then["games"])
    except (KeyError, TypeError, ValueError):
        idle = False
    if idle and x:
        d["driver"] = "board"            # he didn't play in the window: the players around him moved, not him
        try:
            d["bye"] = int(p["team_games"]) == int(then["team_games"])     # his team didn't play either
        except (KeyError, TypeError, ValueError):
            d["bye"] = False
    return {"window": window, "say": say, "rank_line": rank_line, "start_rank": start_rank, "driver": d,
            "opener": proof_opener(week),
            "crossed": x, "milestone": milestone(story, board), "ranks": {r for r in (start_rank, cur) if r}}


def proof_opener(week: int) -> str:
    """This week's lead-in to the proof's reason: the list rotates by week, offset per sport."""
    return PROOF_OPENERS[(int(week) + OPENER_OFFSET) % len(PROOF_OPENERS)]


def opener_problems(proof: str, opener: Optional[str]) -> List[str]:
    """The proof may use this week's lead-in once, or none; never another one from the list."""
    used = [o for o, rx in _OPENER_RE.items() if rx.search(proof)]
    other = [o for o in used if o != opener]
    if other:
        return [f'the proof\'s lead-in this week is "{opener}" (or none, if the hook makes the reason obvious); '
                f'drop {", ".join(f"{chr(34)}{o}{chr(34)}" for o in other)}']
    if opener and len(_OPENER_RE[opener].findall(proof)) > 1:
        return [f'say "{opener}" once at most']
    return []


def caption_with_cta(caption: str) -> str:
    """The model's caption with any site line it wrote cut off, then CTA_LINE as its own sentence."""
    cap = re.split(r"(?i)full board|stingercollectibles|link in bio", caption or "")[0]
    cap = re.sub(r"[\s,:;\u2013\u2014-]+$", "", cap)
    if cap and cap[-1] not in ".?!":
        cap += "."
    return f"{cap} {CTA_LINE}.".strip()


def frame_lines(story: Story, f: dict) -> List[str]:
    p = story.player
    lines = [f"Time period for the whole script: {f['window']}. Say it as {f['say']}; anchor every number to it.",
             f["rank_line"]]
    if f.get("opener"):
        lines.append(f'Lead-in to the proof\'s reason this week: "{f["opener"]}". Use it, or none at all if the hook '
                     f'already makes the reason obvious. Not any other.')
    d = f["driver"]
    rate_txt = ", ".join(f"{STAT_LABELS.get(r['col'], r['col'])} {format_stat(r['col'], r['then'])} -> "
                         f"{format_stat(r['col'], r['now'])}" for r in d["rates"])
    t = d["team"]
    team_txt = (f"{p['team']} was {t['then']} before the window and went {t['window']} in it ({t['now']} now)"
                if t and t["window"] else None)
    x = f.get("crossed") or []
    climb = f["start_rank"] and int(story.cur_rank) < f["start_rank"]
    x_txt = "; ".join(f"{c['name']} No. {c['then']} -> " + (f"No. {c['now']}" if c["now"] else "off our top 30")
                      for c in x)
    if d["driver"] == "board":
        why = f"{p['team']} didn't play in the window (a bye)" if d.get("bye") else "He didn't play in the window"
        lines.append(f"What drove the move: the players around him. {why}, so his numbers didn't change; he moved "
                     f"because players ahead of him dropped ({x_txt}). Say that, and name the player who dropped.")
    elif d["driver"] == "player":
        b = d["biggest"]
        lines.append(f"What drove the move: player (his own production). The number that changed most: "
                     f"{STAT_LABELS.get(b['col'], b['col'])} {format_stat(b['col'], b['then'])} -> "
                     f"{format_stat(b['col'], b['now'])}. Build the proof on it, or on what he did in the window (below) if that "
                     f"says it better (\"five points in his last two games\"). His per-game numbers: {rate_txt}."
                     + (f" Team: {team_txt}." if team_txt else ""))
    elif d["driver"] == "team":
        lines.append(f"What drove the move: team. His per-game numbers barely moved ({rate_txt}); {team_txt}. Say his numbers barely moved and the record changed; you may say "
                     f"our model weighs winning.")
    else:
        lines.append("What drove the move: unclear. Give the single biggest change below and say what changed, "
                     "without claiming it's why he moved." + (f" His per-game numbers: {rate_txt}." if rate_txt else "")
                     + (f" Team: {team_txt}." if team_txt else ""))
    if x and d["driver"] != "board":
        lines.append(("Ahead of him at the start of the window, behind him now: " if climb else
                      "Behind him at the start of the window, ahead of him now: ") + x_txt
                     + ". A fact you may use; don't claim it's why he moved.")
    m = f["milestone"]
    o = m["other"]
    who = f"{o['player_name']} ({o['team']})" if o is not None else "not on the board"
    if m["kind"] == "hold":
        lines.append(f"Milestone: holding No. 1. No. 2 is {who}. The context and the ask are about staying on top.")
    elif m["kind"] == "no1":
        lines.append(f"Milestone: No. 1. He's {m['gap']} spot{'s' if m['gap'] != 1 else ''} behind {who}, the board "
                     f"leader; you may name him.")
    else:
        lines.append(f"Milestone: the top {m['target']}. He's {m['gap']} spot{'s' if m['gap'] != 1 else ''} away; "
                     f"No. {m['target']} is {who}. Don't name the board leader.")
    nick = nickname(p)
    lines.append(f"His name: {p['player_name']}" + (f' (fans know him as "{nick}"; saying "{nick}" is enough)' if nick else "")
                 + ". The hook says it.")
    return lines


def fallback_question(story: Story, board: Optional[pd.DataFrame] = None) -> str:
    """Used when there's no script, or the model's ending still isn't a question after the rewrite: about the
    milestone ("Can Dak crack the top 5?", "Can Giannis catch Luka?")."""
    n = spoken_name(story.player)
    if story.story_type in NO1_STORIES:
        return f"Is {n} your MVP right now?"
    if story.story_type == "top5_drop":
        return f"Is {n} out of the Hart race?"
    if board is None:
        return f"Is {n} a real Hart contender?"
    m = milestone(story, board)
    if m["kind"] == "no1" and m["other"] is not None:
        return f"Can {n} catch {short_name(m['other'])}?"
    if m["kind"] == "top":
        return f"Can {n} crack the top {m['target']}?"
    return f"Is {n} a real Hart contender?"


# ── The user message ─────────────────────────────────────────────────────────────────────────
def _clock(s: float) -> str:
    return f"0:{int(s):02d}"


def _player_line(p: pd.Series) -> str:
    return (f"No. {int(p['rank'])} {p['player_name']} ({p['team']} {p['position']}, team record {team_record(p)}, "
            f"{p.get('movement', '')} vs last week): {stat_line(p)}")


def then_now_lines(p: pd.Series, then: pd.Series, week: int) -> List[str]:
    """The featured player then (an earlier weekly board) vs now, for the script to explain the move. Rates, not
    season totals (totals only go up), plus what he did in the window."""
    p, then = per_game(p), per_game(then)
    ago = week - int(then["week"])
    lines = [f"Then vs now (Week {int(then['week'])}, {ago} week{'s' if ago != 1 else ''} ago -> Week {week}):",
             f"- Rank: No. {int(then['rank'])} -> No. {int(p['rank'])}"]
    if str(then.get("team")) != str(p.get("team")):
        lines.append(f"- Team: {then.get('team')} then, {p.get('team')} now (traded; records not comparable)")
    else:
        rec = f"- Team record (W-L-OTL): {team_record(then)} -> {team_record(p)}"
        w = window_record(team_record(then), team_record(p))
        if w:
            rec += f" ({w} in the window)"
        lines.append(rec)
    stats = [f"{STAT_LABELS.get(c, c)} {format_stat(c, then[c])} -> {format_stat(c, p[c])}"
             for c in rate_cols(p) if c in then and c in p and pd.notna(then[c]) and pd.notna(p[c])]
    if stats:
        lines.append("- His rates (season to date): " + ", ".join(stats))
    try:
        games = int(p["games"]) - int(then["games"])
        if games > 0 and is_goalie(p):
            wins = int(p["goalie_wins"]) - int(then["goalie_wins"])
            lines.append(f"- In the window: {wins} wins in {games} game{'s' if games != 1 else ''}")
        elif games > 0:
            pts, g = int(p["points"]) - int(then["points"]), int(p["goals"]) - int(then["goals"])
            lines.append(f"- In the window: {pts} points ({g} goals) in {games} game{'s' if games != 1 else ''}")
    except (KeyError, TypeError, ValueError):
        pass
    return lines


def build_message(story: Story, board: pd.DataFrame, cards: List[tuple], season: int, week: int,
                  history: str = "", then: Optional[pd.Series] = None, frame: Optional[dict] = None) -> str:
    p = story.player
    frame = frame or story_frame(story, board, then, week)
    prev = (f"No. {story.prev_rank} last week" if story.prev_rank
            else "not ranked in our top 30 last week" if story.story_type != "profile_no1" or week > 1
            else "first board of the season")
    lines = [
        f"NHL Hart Trophy board, {season - 1}-{str(season)[-2:]} season, week {week} (stats through {p['stats_through']}).",
        f"Story type: {story.story_type}. On-screen hook: \"{story.hook}\"",
        f"Why this story: {story.reason}.",
        "",
        f"Featured player: {p['player_name']} ({p['team']}, {p['position']})",
        f"Rank: No. {story.cur_rank} this week, {prev}.",
        *([f"His rank on our earlier boards this season: {history}."] if history else []),
        f"Team record: {team_record(p)} (W-L-OTL). Games played: {int(p['games'])} of {int(p['team_games'])} team games." + (" Goalie." if is_goalie(p) else ""),
        f"Stats: {stat_line(p)}",
    ]
    if isinstance(p.get("chips"), str) and p["chips"].strip():
        lines.append(f"Strength tags: {p['chips']}")
    if isinstance(p.get("why_blurb"), str) and p["why_blurb"].strip():
        lines.append(f"One-line why: {p['why_blurb']}")
    lines += ["", "THE STORY FRAME (build the script on this):"] + frame_lines(story, frame)
    if then is not None:
        lines += [""] + then_now_lines(p, then, week)
    lines += ["", MODEL_NOTE]
    lines += ["", "Top 5 this week:"] + [_player_line(r) for _, r in board.sort_values("rank").head(5).iterrows()]

    lines += ["", "Scenes (write one beat for each, in order):"]
    for name, start, end in REEL_SCENES:
        note = SCENE_NOTES[name]
        if name == "proof":
            note += " Cards: " + "; ".join(f"{label} {value}" for label, value in cards)
        words = (f"{REEL_HOOK_MAX_WORDS} words or fewer" if name == "hook"
                 else f"{REEL_PROOF_MAX_WORDS} words or fewer" if name == "proof"
                 else f"{REEL_ASK_MAX_WORDS} words or fewer" if name == "ask"
                 else f"about {round((end - start) * REEL_WPS)} words")
        lines.append(f"- {name} {_clock(start)}-{_clock(end)}: {words}. {note}")

    player_tag = "#" + re.sub(r"[^A-Za-z0-9]", "", str(p["player_name"]))
    team_tag = TEAM_HASHTAGS.get(str(p["team"]), "")
    lines += ["", f"Approved tags: {APPROVED_TAGS.format(season_tag=f'{season - 1}{str(season)[-2:]}')}",
              f"Player/team tags: {player_tag} {team_tag}".rstrip()]
    return "\n".join(lines)


# ── Checks ────────────────────────────────────────────────────────────────────────────────────
def word_count(data: dict) -> int:
    return sum(len(b["text"].split()) for b in data["beats"])


def sentences(text: str) -> List[str]:
    """Split on . ? ! ("No. 17" isn't a sentence end)."""
    text = re.sub(r"\bNo\.\s*(?=\d)", "No ", text)
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def unanchored_time(text: str) -> List[str]:
    """Sentences that say then/since/now without a time anchor (this week, Week 8, three weeks ago, his last three
    games, ...)."""
    return [s for s in sentences(text) if TIME_WORD.search(s) and not ANCHOR.search(s)]


def name_problems(beats: dict, p: pd.Series) -> List[str]:
    out = []
    parts = _name_parts(p)
    full = " ".join(parts)
    nick = nickname(p)
    spoken = " ".join(beats.values())
    if not (_has(spoken, full) or _has(spoken, str(p["player_name"])) or (nick and _has(spoken, nick))):
        out.append(f'say his full name ("{full}"' + (f' or "{nick}"' if nick else "") + ") in the voiceover")
    hook = beats.get("hook", "")
    if not any(_has(hook, n) for n in [parts[0] if parts else "", parts[-1] if parts else "", nick or ""] if n):
        out.append(f"the hook must name him (e.g. \"{nick or full}: ...\")")
    return out


def context_problems(context: str, story: Story, board: pd.DataFrame, ranks: set) -> List[str]:
    out = []
    repeats = [r for r in sorted(ranks)
               if re.search(rf"\b(?:No\.?|number)\s*{r}\b", context, re.IGNORECASE)]
    if repeats:
        out.append("the context repeats the hook's rank (" + ", ".join(f"No. {r}" for r in repeats)
                   + "); give the gap to the milestone instead")
    if not leader_allowed(story):
        leader = board.sort_values("rank").iloc[0]
        if str(leader.get("player_name")) != str(story.player["player_name"]):
            lp = _name_parts(leader)
            if any(_has(context, n) for n in [" ".join(lp), lp[-1] if lp else "", nickname(leader) or ""] if n):
                out.append(f"don't name the board leader ({leader['player_name']}) in the context; the milestone "
                           "isn't No. 1")
    return out


def problems(data: dict, story: Optional[Story] = None, board: Optional[pd.DataFrame] = None,
             frame: Optional[dict] = None) -> List[str]:
    """Everything wrong with a script, as instructions the model can act on. Empty when it's fine. The name and
    context checks need the story and board."""
    out = []
    names = [s[0] for s in REEL_SCENES]
    beats = data.get("beats") or []
    if [b.get("scene") for b in beats] != names:
        out.append(f"beats must be exactly {names}, in that order")
    n = word_count(data)
    if not REEL_WORDS[0] <= n <= REEL_WORDS[1]:
        out.append(f"the four beats total {n} words; they must total {REEL_WORDS[0]}-{REEL_WORDS[1]}")
    by_scene = {b.get("scene"): b.get("text", "") for b in beats}
    hook = by_scene.get("hook", "")
    if len(hook.split()) > REEL_HOOK_MAX_WORDS:
        out.append(f"the hook beat is {len(hook.split())} words; keep it to {REEL_HOOK_MAX_WORDS} or fewer")
    proof = by_scene.get("proof", "")
    if len(proof.split()) > REEL_PROOF_MAX_WORDS:
        out.append(f"the proof beat is {len(proof.split())} words; cut it to {REEL_PROOF_MAX_WORDS} or fewer "
                   "(one reason, at most two numbers)")
    if ROBOT_HOOK.search(hook):
        out.append('say the hook the way a fan would, as one sentence with a verb and bare numbers: "Jack Eichel '
                   'jumped from 15 to 5 in one week." (no "No.", no colon)')
    if re.search(r"\bour model\b", hook, re.IGNORECASE):
        out.append('don\'t say "our model" in the hook beat; open with his name and the fact')
    if beats:
        last = beats[-1].get("text", "").rstrip()
        if not last.endswith("?"):
            out.append("the last beat must end with the comment question (a question mark)")
        if len(last.split()) > REEL_ASK_MAX_WORDS:
            out.append(f"the last beat is {len(last.split())} words; keep it to {REEL_ASK_MAX_WORDS} or fewer")
    q = data.get("question", "").strip()
    if not q.endswith("?") or len(q.split()) > QUESTION_MAX_WORDS:
        out.append(f"question: one question, {QUESTION_MAX_WORDS} words or fewer, ending with \"?\"")
    loose = [s for b in beats for s in unanchored_time(b.get("text", ""))]
    if loose:
        out.append("say when in every sentence with then/since/now (e.g. \"since Week 8\", \"three weeks ago\"): "
                   + " | ".join(f'"{s}"' for s in loose))
    filler = sorted({m.group(0).lower() for b in beats for m in FILLER.finditer(b.get("text", ""))})
    if filler:
        out.append(f"cut the filler / stock phrases / spoken link: {', '.join(filler)}")
    texts = [b.get("text", "") for b in beats] + [q, data.get("instagram_caption", "")]
    hits = sorted({h for t in texts for h in probability_language(t)})
    if hits:
        out.append(f"remove probability/betting language: {', '.join(hits)}")
    if frame is not None and frame.get("opener"):
        out += opener_problems(by_scene.get("proof", ""), frame.get("opener"))
    if len(data.get("hashtags") or []) != 5:
        out.append("exactly 5 hashtags")
    if story is not None:
        out += name_problems(by_scene, story.player)
        if board is not None:
            ranks = (frame or {}).get("ranks") or {r for r in (story.prev_rank, story.cur_rank) if r}
            out += context_problems(by_scene.get("context", ""), story, board, ranks)
    return out


# ── The API call ──────────────────────────────────────────────────────────────────────────────
def system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        sport=SPORT, mvp=MVP, lo=REEL_WORDS[0], hi=REEL_WORDS[1], cta=CTA_LINE, hook_max=REEL_HOOK_MAX_WORDS,
        proof_max=REEL_PROOF_MAX_WORDS, ask_max=REEL_ASK_MAX_WORDS, q_max=QUESTION_MAX_WORDS,
        ctx=round((REEL_SCENES[2][2] - REEL_SCENES[2][1]) * REEL_WPS), sport_facts=SPORT_FACTS,
        stinger_tag=STINGER_TAG)


def _ask(client, message: str) -> dict:
    resp = client.beta.messages.create(
        model=REEL_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                      # if Opus declines, the API reruns it on a fallback model
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=system_prompt(),
        messages=[{"role": "user", "content": message}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("the model declined the request")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ValueError(f"no text in the response (stop_reason {resp.stop_reason})")
    return json.loads(text)


def finish(data: dict, story: Story, board: Optional[pd.DataFrame] = None, frame: Optional[dict] = None) -> dict:
    """Last-resort fixes after the rewrite, then timings, word count and any remaining warnings."""
    beats = data["beats"]
    if beats and not beats[-1]["text"].rstrip().endswith("?"):
        q = fallback_question(story, board)
        beats[-1]["text"] = f"{beats[-1]['text'].rstrip()} {q}".strip()
        data["question"] = q
    q = data.get("question", "").strip()
    if not q.endswith("?") or len(q.split()) > QUESTION_MAX_WORDS:
        data["question"] = fallback_question(story, board)
    data["instagram_caption"] = caption_with_cta(data.get("instagram_caption", ""))
    data["warnings"] = problems(data, story, board, frame)
    for w in data["warnings"]:
        print(f"  [WARN] Reel script: {w}")
    windows = {name: (start, end) for name, start, end in REEL_SCENES}
    for b in beats:
        b["start"], b["end"] = windows.get(b["scene"], (None, None))
    data["words"] = word_count(data)
    return data


def frame_summary(frame: dict) -> dict:
    """The story frame as plain JSON, for reel_meta.json and the script file."""
    m, d = frame["milestone"], frame["driver"]
    return {"window": frame["window"], "start_rank": frame["start_rank"], "driver": d["driver"],
            "opener": frame.get("opener"),
            "crossed": [c["name"] for c in frame.get("crossed") or []],
            "milestone": ("hold No. 1" if m["kind"] == "hold" else "No. 1" if m["kind"] == "no1" else f"top {m['target']}"),
            "milestone_player": None if m["other"] is None else str(m["other"]["player_name"])}


def generate_reel_script(story: Story, board: pd.DataFrame, cards: List[tuple],
                         season: int, week: int, history: str = "", then: Optional[pd.Series] = None,
                         earlier: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """{beats: [{scene, start, end, text}], question, instagram_caption, hashtags, words, warnings,
    first_draft_problems, rewritten, frame}, or None.
    cards: the (label, value) cards on screen in the Proof scene. history: the featured player's rank on earlier
    weekly boards (reel_story.rank_history). then: his row on the comparison board (reel_story.comparison_row).
    earlier: that whole board (who passed whom over the window)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — reel script skipped")
        return None
    import anthropic

    client = anthropic.Anthropic()
    frame = story_frame(story, board, then, week, earlier)
    message = build_message(story, board, cards, season, week, history, then, frame)
    print(f"  Writing reel script ({REEL_MODEL})...")
    try:
        data = _ask(client, message)
        issues = problems(data, story, board, frame)
        first, rewritten = list(issues), False
        if issues:
            print(f"  Script needs a rewrite: {'; '.join(issues)}")
            data2 = _ask(client, f"{message}\n\nYour last draft was:\n{json.dumps(data, ensure_ascii=False)}\n\n"
                                 f"Fix these problems and keep everything else: {'; '.join(issues)}.")
            if len(problems(data2, story, board, frame)) <= len(issues):
                data, rewritten = data2, True
    except (anthropic.APIError, RuntimeError, ValueError, json.JSONDecodeError) as e:
        print(f"  ! Reel script failed: {e}")
        return None
    data = finish(data, story, board, frame)
    data["first_draft_problems"], data["rewritten"] = first, rewritten
    data["frame"] = frame_summary(frame)
    return data
