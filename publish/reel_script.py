"""
Narration script for the weekly Hart Trophy reel (ported from nfl-awards-predictor), written for Jeff to read aloud (first person, his voice),
split into the five scene beats of config.REEL_SCENES, plus on-screen caption chunks, an Instagram
caption and hashtags.

Uses REEL_MODEL (Opus 5.5) rather than the chips' Haiku: one call a week, and the voice matters.
Skipped with a warning when ANTHROPIC_API_KEY is not set (the reel still renders, without captions).

Hard rules, checked in code (one rewrite request, then the ending line is forced):
  - REEL_WORDS words across the five beats
  - no win probability / % / odds / betting language (CLAUDE.md: no win probability on the board)
  - the last beat ends with CTA_LINE
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import pandas as pd

from config import REEL_MODEL, REEL_SCENES, REEL_WORDS
from publish.generate_captions import TEAM_HASHTAGS
from publish.reel_story import Story
from publish.stats import STAT_LABELS, format_stat, is_goalie, stat_line

CTA_LINE = "Full board at stingercollectibles.com"

# Probability, odds and betting words. Word-bounded so "better" or "between" don't trip it.
FORBIDDEN = re.compile(
    r"%|\bpercent|\bprobab|\bodds\b|\bchances?\b|\blikel(y|ihood)\b|\bfavou?rites?\b|\bbet(s|ting)?\b"
    r"|\bwager|\bparlay|\bsportsbook|\bline(s)? (move|movement)|\blongshot|\block\b|\bpayout",
    re.IGNORECASE,
)

SCENE_NOTES = {
    "hook":    "Hook. On screen: the hook text in big type, with the week label.",
    "player":  "Player. On screen: headshot and team logo; the rank animates from last week's to this week's.",
    "why":     "Why. On screen: the stat chips listed below build in one at a time, with your caption chunks under them.",
    "context": "Context. On screen: the top 5 this week, featured player highlighted.",
    "cta":     "Call to action. On screen: stingercollectibles.com and the Stinger logo.",
}

APPROVED_TAGS = ("#NHL #NHLTwitter #HockeyTwitter #Hockey #NHL{season_tag} #NHLSeason "
                 "#HartTrophy #HartWatch #NHLMVP #MVPRace #NHLAwards #NHLStats #StingerNHL #StingerSports")

SYSTEM_PROMPT = """You write the weekly voiceover script for STINGER's NHL Hart Trophy reel: a 25-second vertical video that tells ONE story from our model's Hart Trophy (NHL MVP) board. Jeff, who runs STINGER, records it himself on his phone, reading your script on the first take.

Voice: Jeff, in the first person. A data guy talking to a friend about what his model found, not an announcer reading copy. Say "our model", "here's what jumped out to me", "I'll be honest". Short sentences. Plain words that are easy to say out loud. Contractions. No hype words, no puns, no exclamation marks unless one really earns it.

Facts: use ONLY the facts in the user message. Never invent a stat, game, injury, streak or standing. Team records are W-L-OTL (e.g. 17-3-3 is 17 wins, 3 regulation losses, 3 overtime losses); say them as given ("seventeen and three and three" or "17-3-3"), and don't claim division, conference or playoff positions. Write ranks as "No. 4". Numbers should be easy to say ("55 points in 23 games").

Make the case with how he's playing: points, goals, assists, points per game, plus-minus, and the team's record (for a goalie: save percentage, goals-against average, wins). Don't use games played or availability ("he's played every game") as a selling point; it isn't an MVP argument.

Season context: if the user message gives his rank by week, use it only as given. A "back_to_no1" story means he held No. 1 earlier this season and has taken it back; never call him new to No. 1 or say it's his first time there.

Never show or imply a probability: no %, no "percent", "odds", "chance", "likely", "favorite", "lock", and no betting language at all. The model ranks players; it does not give anyone a percent chance.

Structure: exactly five beats, one per scene, in order, each sized to its time window (about 2.7 words per second; the user message gives a target for each). The whole script is {lo}-{hi} words. The last beat ends with exactly: "{cta}"

Also write:
- captions: 3 to 5 short on-screen text chunks for the Why scene (each 6 words or fewer), facts from the input, in the order you mention them.
- instagram_caption: 2 sentences, under 280 characters, no hashtags, ranks as "No. 4", ending with "{cta}".
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
        "captions": {"type": "array", "items": {"type": "string"}},
        "instagram_caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["beats", "captions", "instagram_caption", "hashtags"],
    "additionalProperties": False,
}


def _clock(s: float) -> str:
    return f"0:{int(s):02d}"


def _player_line(p: pd.Series) -> str:
    return (f"No. {int(p['rank'])} {p['player_name']} ({p['team']} {p['position']}, team record {p['team_record']}, "
            f"{p.get('movement', '')} vs last week): {stat_line(p)}")


def build_message(story: Story, board: pd.DataFrame, chip_stats: List[str], season: int, week: int,
                  history: str = "") -> str:
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
    lines += ["", "Top 5 this week:"] + [_player_line(r) for _, r in board.sort_values("rank").head(5).iterrows()]

    lines += ["", "Scenes (write one beat for each, in order):"]
    for name, start, end in REEL_SCENES:
        note = SCENE_NOTES[name]
        if name == "why":
            note += " Chips: " + ", ".join(f"{STAT_LABELS.get(c, c)} {format_stat(c, p.get(c))}" for c in chip_stats)
        lines.append(f"- {name} {_clock(start)}-{_clock(end)}: about {round((end - start) * 2.7)} words. {note}")

    name_tag = "#" + re.sub(r"[^A-Za-z0-9]", "", str(p["player_name"]))
    team_tag = TEAM_HASHTAGS.get(str(p["team"]), "")
    lines += ["", f"Approved tags: {APPROVED_TAGS.format(season_tag=f'{season - 1}{str(season)[-2:]}')}",
              f"Player/team tags: {name_tag} {team_tag}".rstrip()]
    return "\n".join(lines)


def word_count(data: dict) -> int:
    return sum(len(b["text"].split()) for b in data["beats"])


def problems(data: dict) -> List[str]:
    out = []
    names = [s[0] for s in REEL_SCENES]
    if [b.get("scene") for b in data["beats"]] != names:
        out.append(f"beats must be exactly {names}, in that order")
    n = word_count(data)
    if not REEL_WORDS[0] <= n <= REEL_WORDS[1]:
        out.append(f"the five beats total {n} words; they must total {REEL_WORDS[0]}-{REEL_WORDS[1]}")
    texts = [b["text"] for b in data["beats"]] + data["captions"] + [data["instagram_caption"]]
    hits = sorted({m.group(0) for t in texts for m in FORBIDDEN.finditer(t)})
    if hits:
        out.append(f"remove probability/betting language: {', '.join(hits)}")
    if data["beats"] and not data["beats"][-1]["text"].rstrip().rstrip(".").endswith(CTA_LINE):
        out.append(f'the last beat must end with "{CTA_LINE}"')
    if not 3 <= len(data["captions"]) <= 5 or any(len(c.split()) > 6 for c in data["captions"]):
        out.append("captions: 3 to 5 chunks, each 6 words or fewer")
    if len(data["hashtags"]) != 5:
        out.append("exactly 5 hashtags")
    return out


def _ask(client, message: str) -> dict:
    resp = client.beta.messages.create(
        model=REEL_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                      # if Opus declines, the API reruns it on a fallback model
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM_PROMPT.format(lo=REEL_WORDS[0], hi=REEL_WORDS[1], cta=CTA_LINE),
        messages=[{"role": "user", "content": message}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("the model declined the request")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ValueError(f"no text in the response (stop_reason {resp.stop_reason})")
    return json.loads(text)


def _force_ending(data: dict) -> None:
    last = data["beats"][-1]
    if not last["text"].rstrip().rstrip(".").endswith(CTA_LINE):
        last["text"] = f"{last['text'].rstrip()} {CTA_LINE}."


def generate_reel_script(story: Story, board: pd.DataFrame, chip_stats: List[str],
                         season: int, week: int, history: str = "") -> Optional[dict]:
    """{beats: [{scene, start, end, text}], captions, instagram_caption, hashtags, words, warnings}, or None.
    history: the featured player's rank on earlier weekly boards (reel_story.rank_history)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — reel script skipped")
        return None
    import anthropic

    client = anthropic.Anthropic()
    message = build_message(story, board, chip_stats, season, week, history)
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

    _force_ending(data)
    data["warnings"] = problems(data)
    for w in data["warnings"]:
        print(f"  [WARN] Reel script: {w}")
    windows = {name: (start, end) for name, start, end in REEL_SCENES}
    for b in data["beats"]:
        b["start"], b["end"] = windows.get(b["scene"], (None, None))
    data["words"] = word_count(data)
    return data
