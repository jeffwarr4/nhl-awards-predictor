"""
Generate a ready-to-paste Instagram/Facebook caption with hashtags for the weekly
Hart Trophy carousel: slide 1 the top-10 board, slide 2 "On the Bubble" (ranks 11–15).

Port of nfl-awards-predictor/src/publish/generate_captions.py.

Output:
  predictions/{season}/graphics/caption_hart.txt

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest board + chips):
    python -m publish.generate_captions
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import BUBBLE_RANKS, CURRENT_PRED_DIR
from publish.generate_chips import CLAUDE_MODEL, EFFORT
from publish.stats import stat_line

TEAM_HASHTAGS = {
    "ANA": "#FlyTogether",     "BOS": "#NHLBruins",      "BUF": "#LetsGoBuffalo",  "CAR": "#LetsGoCanes",
    "CBJ": "#CBJ",             "CGY": "#Flames",         "CHI": "#Blackhawks",     "COL": "#GoAvsGo",
    "DAL": "#TexasHockey",     "DET": "#LGRW",           "EDM": "#LetsGoOilers",   "FLA": "#TimeToHunt",
    "LAK": "#GoKingsGo",       "MIN": "#mnwild",         "MTL": "#GoHabsGo",       "NJD": "#NJDevils",
    "NSH": "#Smashville",      "NYI": "#Isles",          "NYR": "#NYR",            "OTT": "#GoSensGo",
    "PHI": "#LetsGoFlyers",    "PIT": "#LetsGoPens",     "SEA": "#SeaKraken",      "SJS": "#TheFutureIsTeal",
    "STL": "#stlblues",        "TBL": "#GoBolts",        "TOR": "#LeafsForever",   "UTA": "#TusksUp",
    "VAN": "#Canucks",         "VEG": "#VegasBorn",      "WPG": "#GoJetsGo",       "WSH": "#ALLCAPS",
}

SYSTEM_PROMPT = """You are a sports social media manager for STINGER, an NHL awards prediction tracker.

Write the weekly update post for our Hart Trophy carousel: slide 1 is the top-10 board, slide 2 is "On the Bubble" (ranks 11-15, the players knocking on the door). Produce:
1. A punchy caption body — 3 sentences max, under 320 characters total. The first 1-2 sentences cover the top of the board (slide 1). The last sentence names the top bubble player and why they're close to the top 10 (slide 2). Data-driven and slightly hype. Reference actual stats from the input. You may use the history note for context, but don't force it. No filler ("Let's dive in!", "Check this out!"). 1-2 emoji max. Do NOT include hashtags in the body. Write ranks as "No. 11", never "#11" (Instagram reads "#11" as a hashtag).
   Describe team records only with the numbers given: never call a record perfect or undefeated unless it has no losses, and don't claim standings (division or conference leads, playoff spots) the input doesn't state.
2. 5 hashtags for Instagram, most important first (Instagram now limits posts to 5). Choose ONLY from the approved base tags below plus the player-name and team tags given in the user message. Do NOT invent tags that aren't in the approved list or given in the message.
3. 5 hashtags for Facebook, most important first (favour generic NHL + award tags over player/team specifics).

Approved base hashtags (pick the most relevant, always include #StingerNHL):
#NHL #NHLTwitter #HockeyTwitter #Hockey #NHL{season_tag} #NHLSeason
#HartTrophy #HartWatch #NHLMVP #MVPRace #NHLAwards #NHLStats
#StingerNHL #StingerSports

Output ONLY valid JSON in exactly this shape, no markdown fences, no preamble:
{{"caption": "...", "hashtags_instagram": ["#tag1", ...], "hashtags_facebook": ["#tag1", ...]}}"""


# Context for the bubble sentence (from the 2025-26 weekly snapshots vs the actual vote).
BUBBLE_HISTORY = ("2025-26: Nikita Kucherov sat No. 25 on our mid-December board and won the Hart.")


def _build_prompt(board: pd.DataFrame, chip_df: pd.DataFrame, week: int) -> str:
    whys = dict(zip(chip_df["player_key"], chip_df["Why"])) if not chip_df.empty else {}
    board = board.sort_values("rank")

    def line(p: pd.Series) -> str:
        out = (f"No. {int(p['rank'])} {p['player_name']} ({p['team']} {p['position']}, "
               f"team {p['team_record']}): {stat_line(p)}")
        if isinstance(whys.get(p["player_key"]), str) and whys[p["player_key"]]:
            out += f"  | {whys[p['player_key']]}"
        return out

    lines = [f"Hart Trophy board — week {week}", "", "Slide 1 — top 3 contenders:"]
    lines += [line(p) for _, p in board.head(3).iterrows()]
    bubble = board[board["rank"].between(*BUBBLE_RANKS)]
    if not bubble.empty:
        lines += ["", f"Slide 2 — On the Bubble (ranks {BUBBLE_RANKS[0]}-{BUBBLE_RANKS[1]}):"]
        lines += [line(p) for _, p in bubble.iterrows()]
        lines += ["", f"History note: {BUBBLE_HISTORY}"]
    top3 = board.head(3)
    name_tags = " ".join("#" + re.sub(r"[^A-Za-z0-9]", "", n) for n in top3["player_name"])
    team_tags = " ".join(dict.fromkeys(TEAM_HASHTAGS[t] for t in top3["team"] if t in TEAM_HASHTAGS))
    lines += ["", f"Player tags: {name_tags}", f"Team tags: {team_tags}"]
    return "\n".join(lines)


def generate_caption(client, board: pd.DataFrame, chip_df: pd.DataFrame,
                     season: int, week: int, retries: int = 2) -> Optional[dict]:
    prompt = _build_prompt(board, chip_df, week)
    system = SYSTEM_PROMPT.format(season_tag=f"{season - 1}{str(season)[-2:]}")
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=3000,
                output_config={"effort": EFFORT},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), None)
            if not text:
                raise ValueError(f"No text in response (stop_reason={resp.stop_reason})")
            text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries:
                print(f"  ! Caption generation failed: {e}")
                return None
            time.sleep(1.5)


def generate_caption_file(board: pd.DataFrame, chip_df: pd.DataFrame, gfx_dir: Path,
                          season: int, week: int) -> Optional[Path]:
    """board: the 30-deep board (a top-10-only board still works; slide 2 is then omitted)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — caption generation skipped")
        return None

    from anthropic import Anthropic
    print("  Generating caption: Hart Trophy")
    data = generate_caption(Anthropic(), board, chip_df, season, week)
    if data is None:
        return None

    gfx_dir.mkdir(parents=True, exist_ok=True)
    path = gfx_dir / "caption_hart.txt"
    path.write_text(
        f"=== HART TROPHY WATCH — WEEK {week}, {season - 1}–{str(season)[-2:]} ===\n\n"
        f"CAPTION:\n{data['caption']}\n\n"
        f"INSTAGRAM HASHTAGS:\n{' '.join(data['hashtags_instagram'])}\n\n"
        f"FACEBOOK HASHTAGS:\n{' '.join(data['hashtags_facebook'])}\n",
        encoding="utf-8",
    )
    print(f"Caption -> {path}")
    return path


if __name__ == "__main__":
    top = pd.read_csv(CURRENT_PRED_DIR / "board_hart_latest.csv")
    chip_path = CURRENT_PRED_DIR / "chip_data" / "chip_data_latest.csv"
    chips = pd.read_csv(chip_path).fillna("") if chip_path.exists() else pd.DataFrame(columns=["player_key", "Why"])
    generate_caption_file(top, chips, CURRENT_PRED_DIR / "graphics",
                          int(top["season"].iloc[0]), int(top["week"].iloc[0]))
