"""
Generate the "Chip" (3 pipe-separated tags) and "Why" (one-line rationale) for the
top 3 on the Hart board, plus a short Why (no chip) for the "On the Bubble" ranks
(config.BUBBLE_RANKS), using the Claude API.

Port of nfl-awards-predictor/src/publish/generate_chips.py.

Output:
  predictions/{season}/chip_data/chip_data_latest.csv   (player_key, Chip, Why, week; Chip blank for bubble rows)
  predictions/{season}/chip_data/chip_data_{ts}.csv     (+ rank, for reference)

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest board):
    python -m publish.generate_chips
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import BUBBLE_RANKS, CURRENT_PRED_DIR, TOP_N
from publish.stats import position_group, stat_line

# Haiku: fast and cheap, and reliable at strict JSON for 3-word tags. Thinking is on by
# default on Haiku 5.5; low effort keeps it short, and max_tokens leaves room for it.
CLAUDE_MODEL = "claude-haiku-5-5"
EFFORT = "low"

CHIP_SYSTEM_PROMPT = """You generate two short labels for NHL Hart Trophy prediction graphics: a "Chip" and a "Why".

Rules:
- "chip": exactly 3 short tags (1-3 words each), separated by " | ". Each tag names a distinct statistical strength (e.g. "Point machine", "Power-play threat", "Shot volume", "Two-way D", "Winning team", "Elite save %").
- "why": one sentence, under 100 characters, giving the stat-driven reason for the player's rank. No fluff, no adjectives that aren't backed by a stat in the input.
- Base every word ONLY on the stats given in the user message. Never invent, infer, or assume a stat that wasn't provided.
- Tags describe skill/performance, not identity — don't put the player's name, team, or rank number in the chip itself.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{"chip": "Tag1 | Tag2 | Tag3", "why": "One sentence explanation."}"""


BUBBLE_WHY_MAX = 75   # two wrapped lines on the narrowest bubble label plate (330 px)

BUBBLE_SYSTEM_PROMPT = f"""You write a one-line "Why" for players ranked just outside the top 10 of an NHL Hart Trophy prediction board ("On the Bubble").

Rules:
- At most {BUBBLE_WHY_MAX} characters, including spaces. Shorter is better. No ending period.
- Stats only: use ONLY the numbers given in the user message. Never invent, infer, or assume a stat.
- When the stats show it, say what is keeping the player out of the top 10 (e.g. team record, point total, games missed). You may use the comparison player (the lowest-ranked top-10 player in the same role: forward, defenseman or goalie), but only when it makes the point clearer; if the player's own numbers tell the story better (e.g. a strong record or a standout stat), use those instead.
- Never use this player's own name, first or last; the graphic already shows it. Never mention a rank number (no "#10"). Refer to the comparison player by last name only.
- No hype adjectives.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{{"why": "Short line"}}"""

ROLE = {"F": "forward", "D": "defenseman", "G": "goalie"}


def build_user_message(player: pd.Series) -> str:
    return (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
            f"Rank: #{int(player['rank'])} in our model's Hart Trophy prediction\n"
            f"Team record: {player['team_record']} (W-L-OTL)\n"
            f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
            f"Stats: {stat_line(player)}")


def _parse_json(text: str) -> dict:
    """First JSON object in the reply; ignores code fences and any trailing commentary."""
    text = re.sub(r"^```json|```$", "", text.strip()).strip()
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def _ask(client, system: str, message: str) -> dict:
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        output_config={"effort": EFFORT},
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise ValueError(f"No text in response (stop_reason={resp.stop_reason})")
    return _parse_json(text)


def generate_chip_for_player(client, player: pd.Series, retries: int = 2) -> dict:
    message = build_user_message(player)
    for attempt in range(retries + 1):
        try:
            data = _ask(client, CHIP_SYSTEM_PROMPT, message)
            return {"chip": data["chip"], "why": data["why"]}
        except Exception as e:
            if attempt == retries:
                print(f"  ! Chip generation failed for {player.get('player_name', '?')}: {e}")
                return {"chip": "", "why": ""}
            time.sleep(1.5)


def build_bubble_message(player: pd.Series, comp: pd.Series | None) -> str:
    msg = (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
           f"Rank: #{int(player['rank'])}, just outside the top 10\n"
           f"Team record: {player['team_record']} (W-L-OTL)\n"
           f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
           f"Stats: {stat_line(player)}")
    if comp is not None:
        role = ROLE[position_group(comp["position"])]
        msg += (f"\n\nFor comparison, the lowest-ranked {role} inside the top 10: "
                f"{comp['player_name']} (team record {comp['team_record']}): {stat_line(comp)}")
    return msg


def generate_bubble_why(client, player: pd.Series, comp: pd.Series | None, retries: int = 2) -> str:
    """≤ BUBBLE_WHY_MAX chars. Errors retry like chips; an over-length answer gets one
    rewrite request, then is hard-truncated (a failed rewrite keeps the first answer)."""
    message = build_bubble_message(player, comp)
    for attempt in range(retries + 1):
        try:
            why = str(_ask(client, BUBBLE_SYSTEM_PROMPT, message)["why"]).strip()
            break
        except Exception as e:
            if attempt == retries:
                print(f"  ! Bubble Why failed for {player.get('player_name', '?')}: {e}")
                return ""
            time.sleep(1.5)
    if len(why) > BUBBLE_WHY_MAX:
        try:
            why = str(_ask(client, BUBBLE_SYSTEM_PROMPT,
                           f"{message}\n\nYour last answer was {len(why)} characters: "
                           f'"{why}". Rewrite it in {BUBBLE_WHY_MAX} characters or fewer.')["why"]).strip()
        except Exception as e:
            print(f"  ! Bubble Why rewrite failed for {player.get('player_name', '?')}: {e}")
    if len(why) > BUBBLE_WHY_MAX:
        why = why[:BUBBLE_WHY_MAX - 1].rstrip(" ,;·-") + "…"
    return why


def comparison_player(board: pd.DataFrame, player: pd.Series) -> pd.Series | None:
    """Lowest-ranked top-10 player in the same role (forward / defenseman / goalie);
    falls back to #TOP_N when no top-10 player shares the role."""
    top = board[board["rank"] <= TOP_N].sort_values("rank")
    same = top[top["position"].map(position_group) == position_group(player["position"])]
    pick = same if not same.empty else top
    return pick.iloc[-1] if not pick.empty else None


def load_chips(outdir: Path, week: int) -> pd.DataFrame:
    """This week's saved chips (same-week rerun without a key), else empty."""
    path = Path(outdir) / "chip_data" / "chip_data_latest.csv"
    if path.exists():
        df = pd.read_csv(path).fillna("")
        if "week" in df.columns and len(df) and int(df["week"].iloc[0]) == week:
            print(f"  Reusing this week's chips from {path.name}")
            return df[["player_key", "Chip", "Why"]]
    return pd.DataFrame(columns=["player_key", "Chip", "Why"])


def generate_chips(board: pd.DataFrame, outdir: Path, top_n: int = 3) -> pd.DataFrame:
    """Chips + Why for the top_n players, Why only for ranks BUBBLE_RANKS.
    Returns (player_key, Chip, Why); empty if no API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — chip generation skipped")
        return pd.DataFrame(columns=["player_key", "Chip", "Why"])

    from anthropic import Anthropic
    client = Anthropic()

    board = board.sort_values("rank")
    rows = []
    for _, player in board.head(top_n).iterrows():
        print(f"  Generating chip: {player['player_name']}")
        result = generate_chip_for_player(client, player)
        rows.append({"player_key": player["player_key"], "Chip": result["chip"],
                     "Why": result["why"], "rank": int(player["rank"])})

    for _, player in board[board["rank"].between(*BUBBLE_RANKS)].iterrows():
        print(f"  Generating bubble Why: {player['player_name']}")
        rows.append({"player_key": player["player_key"], "Chip": "",
                     "Why": generate_bubble_why(client, player, comparison_player(board, player)),
                     "rank": int(player["rank"])})
    chip_df = pd.DataFrame(rows)
    if not (chip_df["Why"] != "").any():
        print("[WARN] every chip/Why request failed — nothing saved")
        return pd.DataFrame(columns=["player_key", "Chip", "Why"])

    chip_dir = Path(outdir) / "chip_data"
    chip_dir.mkdir(parents=True, exist_ok=True)
    week = int(board["week"].iloc[0])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chip_df.assign(week=week)[["player_key", "Chip", "Why", "week"]].to_csv(
        chip_dir / "chip_data_latest.csv", index=False)
    chip_df.assign(week=week).to_csv(chip_dir / f"chip_data_{ts}.csv", index=False)
    print(f"Chip data -> {chip_dir / 'chip_data_latest.csv'}")
    return chip_df[["player_key", "Chip", "Why"]]


if __name__ == "__main__":
    generate_chips(pd.read_csv(CURRENT_PRED_DIR / "board_hart_latest.csv"), CURRENT_PRED_DIR)
