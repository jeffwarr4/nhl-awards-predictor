"""
Narration script for the weekly Hart Trophy reel, written for Jeff to voice over in Instagram (first person, his
voice), split into the four scene beats of config.REEL_SCENES, plus the on-screen comment question, an Instagram
caption and hashtags.

Reworked Nov 2026 for retention: the first beat is the surprising fact (no warm-up), there's no filler, and the
reel ends on one easy comment question instead of a spoken URL (the URL is in the caption and small on screen).

Uses REEL_MODEL (Opus 5.5) rather than the chips' Haiku: one call a week, and the voice matters.
Skipped with a warning when ANTHROPIC_API_KEY is not set (the reel still renders, with a fallback question).

Hard rules, checked in code by problems() (one rewrite request; if the ending still isn't a question, a
fallback question for the story type is used):
  - REEL_WORDS words across the four beats; hook <= REEL_HOOK_MAX_WORDS, ask <= REEL_ASK_MAX_WORDS
  - the last beat ends with "?"; the on-screen question is 10 words or fewer
  - no filler phrases, no "our model" in the hook, no spoken URL
  - no MVP probability / % / odds / betting language (the board ranks players; it shows no probability)
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import pandas as pd

from config import (
    REEL_ASK_MAX_WORDS, REEL_HOOK_MAX_WORDS, REEL_MODEL, REEL_PROOF_MAX_WORDS, REEL_SCENES, REEL_WORDS, REEL_WPS,
)
from publish.generate_captions import TEAM_HASHTAGS
from publish.reel_story import Story
from publish.stats import REEL_LABELS, format_stat, is_goalie, per_game, rate_cols, stat_line

CTA_LINE = "Full board at stingercollectibles.com"     # ends the Instagram caption; not spoken any more
QUESTION_MAX_WORDS = 10

# Probability, odds and betting words. Word-bounded so "better" or "between" don't trip it.
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


# Warm-up phrases that cost seconds and don't say anything. Checked in the spoken beats only.
FILLER = re.compile(
    r"here'?s what (jumped|stood) out|\bi'?ll be honest\b|\blet'?s\b|\bcheck (it |this )?out\b|\bmake sure to\b"
    r"|\bstingercollectibles\b|\blink in bio\b|\bfull board\b",
    re.IGNORECASE,
)

# What the model weighs, in words the script can say (Models/hart_common.hart_scores: half a vote-share model, 0.4 a
# top-5 model, 0.1 a win softmax; both models are trained on past Hart voting with scoring rates (points, goals,
# assists, power-play goals and shots per game), team points percentage, age and position; counting stats are
# projected to an 82-game pace).
MODEL_NOTE = ("How our model ranks Hart Trophy candidates: it learned from past Hart voting. Mostly the player's "
              "scoring pace (points, goals and assists per game, power-play goals, shots), weighed with his team's "
              "points percentage. Counting stats are projected to a full 82-game season, so a missed game doesn't "
              "sink a player. Goalies rarely rank high.")

SCENE_NOTES = {
    "hook":    "Hook. On screen from the very first frame: his headshot, the hook text and his rank flipping from last "
               "week's to this week's. Say the surprising fact, nothing else. No greeting, no \"our model\", no name-drop "
               "of STINGER.",
    "proof":   "Proof. On screen: the cards listed below punch in one at a time. Explain WHY the model moved "
               "him: what changed since then (his numbers, his team, or both), tied to what the model weighs. "
               "Two or three numbers at most, each as a before-and-after or with what it means.",
    "context": "Context. On screen: the top 5 this week, featured player highlighted. Where he sits and who's ahead.",
    "ask":     "Ask. On screen: your comment question in big type. One question viewers can answer in a word "
               "(yes/no, or pick one of two names). Nothing after it.",
}

APPROVED_TAGS = ("#NHL #NHLTwitter #HockeyTwitter #Hockey #NHL{season_tag} #NHLSeason "
                 "#HartTrophy #HartWatch #NHLMVP #MVPRace #NHLAwards #NHLStats #StingerNHL #StingerSports")

SYSTEM_PROMPT = """You write the weekly voiceover script for STINGER's NHL Hart Trophy reel: a 20-second vertical video that tells ONE story from our model's Hart Trophy (NHL MVP) board. Jeff, who runs STINGER, records it in Instagram over the video on the first take.

The first two seconds decide whether anyone keeps watching. Open with the most surprising fact in the input (a rank jump, a big number), in {hook_max} words or fewer. No greeting, no "our model" in the first beat, no warm-up.

Voice: Jeff, in the first person. A data guy telling a friend what his model found. Short sentences. Plain words that are easy to say out loud. Contractions. Every sentence carries a fact. No filler: never "here's what jumped out", "I'll be honest", "let's", "check out", "make sure to". No hype words, no puns, no exclamation marks.

Facts: use ONLY the facts in the user message. Never invent a stat, game, injury, streak or standing. Team records are W-L-OTL (e.g. 17-3-3 is 17 wins, 3 regulation losses, 3 overtime losses); say them as given ("17-3-3"), and don't call a record perfect or undefeated unless it has no losses. Don't claim division, conference or playoff positions. Say numbers the way a fan would out loud: round them ("fifty-five points in twenty-three games", "more than two points a game" for 2.39), never past what the number supports. Save percentage is given as a decimal (.921): say it the way fans do ("a .921 save percentage", "nine-twenty-one"). A save percentage is fine; a win percentage is not (say the record), and no other percent of anything. Write ranks as "No. 4".

Explain, don't recite. The viewer should come away knowing WHY the model moved him, not a list of his stats. When the user message has a "then vs now" comparison, build the proof beat on it: say what changed since then (his own scoring pace, his team's record, or both) and connect it to what the model weighs (given in the user message; say it in plain words). If his numbers barely moved and his team's record did, say that plainly, e.g. "So what changed? Edmonton." and "That's our model reacting to winning." If his pace jumped, say which number and by how much, or what he did since then ("eleven points in four games"). The proof beat is {proof_max} words or fewer: short sentences, one fact each, no "and" chaining two facts. His own numbers get one quick, rounded line; the weight goes on the ONE change that moved him, as a before-and-after ("5-4-1 then. 6-0-1 since."). Never a list of stats. A modest number isn't a selling point. Don't use games played or availability ("he's played every game") as a selling point; it isn't an MVP argument.

Season context: if the user message gives his rank by week, use it only as given. A "back_to_no1" story means he held No. 1 earlier this season and has taken it back; never call him new to No. 1.

Never show or imply a probability: no win %, no "percent chance", "odds", "chance", "likely", "favorite", "lock", and no betting language at all. The model ranks players; it does not give anyone a percent chance. The only percentage allowed is a save percentage.

End on ONE comment question that a viewer can answer in a word: yes/no, or pick one of two players named in the input (e.g. "McDavid or MacKinnon: who finishes higher?"). The last beat is {ask_max} words or fewer and ends with that question mark. Don't mention the website, a link or "full board" anywhere in the beats.

Two examples of the voice and shape (made-up facts; match the style, never reuse the facts or names):
  When the team drove the move:
    hook: "No. 22 to No. 6 in three weeks."
    proof: "His scoring barely moved. Still about a point a game. So what changed? Winnipeg. 5-5-1 then. 7-0-1 since. That's our model reacting to winning."
    context: "Only three forwards sit ahead of him now."
    ask: "Is Connor a real Hart contender?"
  When his own play drove it:
    proof: "Same team, same pace. The difference is him. Eleven points in four games. Goals a game went from .3 to .6."

Structure: exactly four beats, one per scene, in order, each sized to its time window (about {wps:g} words per second; the user message gives a target for each). The whole script is {lo}-{hi} words.

Also write:
- question: the comment question as on-screen text, {q_max} words or fewer, ending with "?".
- instagram_caption: 2 sentences, under 280 characters, no hashtags, ranks as "No. 4". The first sentence repeats the question; the second ends with exactly "{cta}".
- hashtags: exactly 5, most important first, chosen only from the approved tags and the player/team tags in the user message. Always include #StingerNHL.
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


def _clock(s: float) -> str:
    return f"0:{int(s):02d}"


def _player_line(p: pd.Series) -> str:
    return (f"No. {int(p['rank'])} {p['player_name']} ({p['team']} {p['position']}, team record {p['team_record']}, "
            f"{p.get('movement', '')} vs last week): {stat_line(p)}")


def spoken_name(p: pd.Series) -> str:
    """What the fallback question calls him: the last name ("McDavid", "MacKinnon")."""
    return str(p["player_name"]).split()[-1]


def fallback_question(story: Story) -> str:
    """Used when there's no script, or the model's ending still isn't a question after the rewrite."""
    n = spoken_name(story.player)
    if story.story_type in ("new_no1", "back_to_no1", "profile_no1"):
        return f"Is {n} your MVP right now?"
    if story.story_type == "top5_drop":
        return f"Is {n} out of the Hart race?"
    return f"Is {n} a real Hart contender?"


def _wlo(record) -> Optional[tuple]:
    """(W, L, OTL) from "17-3-3"."""
    try:
        w, l, o = (int(x) for x in str(record).split("-")[:3])
        return w, l, o
    except ValueError:
        return None


def then_now_lines(p: pd.Series, then: pd.Series, week: int) -> List[str]:
    """The featured player then (an earlier weekly board) vs now, for the script to explain the move. Rates, not
    season totals (totals only go up), plus what he did in the games since then."""
    p, then = per_game(p), per_game(then)
    ago = week - int(then["week"])
    lines = [f"Then vs now (Week {int(then['week'])}, {ago} week{'s' if ago != 1 else ''} ago -> Week {week}):",
             f"- Rank: No. {int(then['rank'])} -> No. {int(p['rank'])}"]
    if str(then.get("team")) != str(p.get("team")):
        lines.append(f"- Team: {then.get('team')} then, {p.get('team')} now (traded; records not comparable)")
    else:
        rec = f"- Team record (W-L-OTL): {then['team_record']} -> {p['team_record']}"
        a, b = _wlo(then["team_record"]), _wlo(p["team_record"])
        if a and b and all(y >= x for x, y in zip(a, b)):
            rec += f" ({'-'.join(str(y - x) for x, y in zip(a, b))} since then)"
        lines.append(rec)
    stats = [f"{REEL_LABELS.get(c, c)} {format_stat(c, then[c])} -> {format_stat(c, p[c])}"
             for c in rate_cols(p) if c in then and c in p and pd.notna(then[c]) and pd.notna(p[c])]
    if stats:
        lines.append("- His rates: " + ", ".join(stats))
    try:
        games = int(p["games"]) - int(then["games"])
        if games > 0 and is_goalie(p):
            wins = int(p["goalie_wins"]) - int(then["goalie_wins"])
            lines.append(f"- Since then: {wins} wins in {games} game{'s' if games != 1 else ''}")
        elif games > 0:
            pts, g = int(p["points"]) - int(then["points"]), int(p["goals"]) - int(then["goals"])
            lines.append(f"- Since then: {pts} points ({g} goals) in {games} game{'s' if games != 1 else ''}")
    except (KeyError, TypeError, ValueError):
        pass
    return lines


def build_message(story: Story, board: pd.DataFrame, cards: List[tuple], season: int, week: int,
                  history: str = "", then: Optional[pd.Series] = None) -> str:
    p = story.player
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
        f"Team record: {p['team_record']} (W-L-OTL). Games played: {int(p['games'])} of {int(p['team_games'])} team games." + (" Goalie." if is_goalie(p) else ""),
        f"Stats: {stat_line(p)}",
    ]
    if isinstance(p.get("chips"), str) and p["chips"].strip():
        lines.append(f"Strength tags: {p['chips']}")
    if isinstance(p.get("why_blurb"), str) and p["why_blurb"].strip():
        lines.append(f"One-line why: {p['why_blurb']}")
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


def word_count(data: dict) -> int:
    return sum(len(b["text"].split()) for b in data["beats"])


def problems(data: dict) -> List[str]:
    """Everything wrong with a script, as instructions the model can act on. Empty when it's fine."""
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
                   "(short sentences, one fact each, one change)")
    if re.search(r"\bour model\b", hook, re.IGNORECASE):
        out.append('don\'t say "our model" in the hook beat; open with the fact')
    if beats:
        last = beats[-1].get("text", "").rstrip()
        if not last.endswith("?"):
            out.append("the last beat must end with the comment question (a question mark)")
        if len(last.split()) > REEL_ASK_MAX_WORDS:
            out.append(f"the last beat is {len(last.split())} words; keep it to {REEL_ASK_MAX_WORDS} or fewer")
    q = data.get("question", "").strip()
    if not q.endswith("?") or len(q.split()) > QUESTION_MAX_WORDS:
        out.append(f"question: one question, {QUESTION_MAX_WORDS} words or fewer, ending with \"?\"")
    filler = sorted({m.group(0).lower() for b in beats for m in FILLER.finditer(b.get("text", ""))})
    if filler:
        out.append(f"cut the filler / spoken link: {', '.join(filler)}")
    texts = [b.get("text", "") for b in beats] + [q, data.get("instagram_caption", "")]
    hits = sorted({h for t in texts for h in probability_language(t)})
    if hits:
        out.append(f"remove probability/betting language: {', '.join(hits)}")
    if not data.get("instagram_caption", "").rstrip().rstrip(".").endswith(CTA_LINE):
        out.append(f'the Instagram caption must end with "{CTA_LINE}"')
    if len(data.get("hashtags") or []) != 5:
        out.append("exactly 5 hashtags")
    return out


def _ask(client, message: str) -> dict:
    resp = client.beta.messages.create(
        model=REEL_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                      # if Opus declines, the API reruns it on a fallback model
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM_PROMPT.format(lo=REEL_WORDS[0], hi=REEL_WORDS[1], wps=REEL_WPS, cta=CTA_LINE, hook_max=REEL_HOOK_MAX_WORDS, proof_max=REEL_PROOF_MAX_WORDS,
                                    ask_max=REEL_ASK_MAX_WORDS, q_max=QUESTION_MAX_WORDS),
        messages=[{"role": "user", "content": message}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("the model declined the request")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ValueError(f"no text in the response (stop_reason {resp.stop_reason})")
    return json.loads(text)


def finish(data: dict, story: Story) -> dict:
    """Last-resort fixes after the rewrite, then timings, word count and any remaining warnings."""
    beats = data["beats"]
    if beats and not beats[-1]["text"].rstrip().endswith("?"):
        q = fallback_question(story)
        beats[-1]["text"] = f"{beats[-1]['text'].rstrip()} {q}".strip()
        data["question"] = q
    q = data.get("question", "").strip()
    if not q.endswith("?") or len(q.split()) > QUESTION_MAX_WORDS:
        data["question"] = fallback_question(story)
    cap = data.get("instagram_caption", "").rstrip()
    if not cap.rstrip(".").endswith(CTA_LINE):
        data["instagram_caption"] = f"{cap} {CTA_LINE}".strip()
    data["warnings"] = problems(data)
    for w in data["warnings"]:
        print(f"  [WARN] Reel script: {w}")
    windows = {name: (start, end) for name, start, end in REEL_SCENES}
    for b in beats:
        b["start"], b["end"] = windows.get(b["scene"], (None, None))
    data["words"] = word_count(data)
    return data


def generate_reel_script(story: Story, board: pd.DataFrame, cards: List[tuple],
                         season: int, week: int, history: str = "", then: Optional[pd.Series] = None) -> Optional[dict]:
    """{beats: [{scene, start, end, text}], question, instagram_caption, hashtags, words, warnings}, or None.
    cards: the (label, value) cards on screen in the Proof scene. history: the featured player's rank on earlier
    weekly boards (reel_story.rank_history). then: his row on the comparison board (reel_story.comparison_row)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — reel script skipped")
        return None
    import anthropic

    client = anthropic.Anthropic()
    message = build_message(story, board, cards, season, week, history, then)
    print(f"  Writing reel script ({REEL_MODEL})...")
    try:
        data = _ask(client, message)
        issues = problems(data)
        if issues:
            print(f"  Script needs a rewrite: {'; '.join(issues)}")
            data2 = _ask(client, f"{message}\n\nYour last draft was:\n{json.dumps(data, ensure_ascii=False)}\n\n"
                                 f"Fix these problems and keep everything else: {'; '.join(issues)}.")
            if len(problems(data2)) <= len(issues):
                data = data2
    except (anthropic.APIError, RuntimeError, ValueError, json.JSONDecodeError) as e:
        print(f"  ! Reel script failed: {e}")
        return None
    return finish(data, story)
