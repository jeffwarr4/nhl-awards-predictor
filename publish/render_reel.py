"""
Weekly Hart Trophy reel (ported from nfl-awards-predictor, reworked with the NBA reel): one story from the board as a silent 1080×1920 MP4 (20 s + a 1 s hold) and a narration
script Jeff voices over in Instagram. Posted first, the afternoon the board completes; the static board
posts the next day (order reversed 2026-11-24).

Reworked Nov 2026 for retention: four scenes (hook, proof, context, ask). Frame 0 already shows the hook, the
player and his rank (no fade-in), and the reel ends on a comment question instead of a spoken URL.

    python -m publish.render_reel                       # predictions/{CURRENT_YEAR}/
    python -m publish.render_reel --outdir <replay dir> # a replay folder
    python -m publish.render_reel --frames-only         # one PNG per scene (+ safe-zone guides), no video
    python -m publish.render_reel --no-script           # skip the API call (no narration; fallback question)

Reads board_hart_latest.csv and .prev_board_hart.csv (movement baseline) from the folder, picks the story
(reel_story.pick_story), writes the script (reel_script, REEL_MODEL), renders frames with Pillow and pipes
them to ffmpeg (imageio-ffmpeg's bundled binary): H.264, 30 fps, yuv420p, +faststart, no audio.

Outputs ({outdir}/reel/):
  reel_latest.mp4    not committed (.gitignore); the workflow uploads it to the "reel-latest" release
                     as nhl_reel_{season}_wk{NN}.mp4 (reel_meta.json: release_asset)
  reel_script.txt    scene-by-scene script, readable on a phone
  reel_meta.json     story, player, ranks, question, caption, hashtags, week
  frames/            --frames-only: frame 0, the last frame of each scene, and *_guides.png with the platform
                     UI zones shaded

Layout keeps everything important inside x 120–960, y 250–1520 (config REEL_SAFE_*).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from config import (
    BOARD_BACKGROUND, CURRENT_PRED_DIR, REEL_FPS, REEL_HOLD_S, REEL_MODEL, REEL_SAFE_BOTTOM,
    REEL_SAFE_LEFT, REEL_SAFE_RIGHT, REEL_SAFE_TOP, REEL_SCENES, REEL_SIZE, TOP_N,
)
from publish.reel_script import fallback_question, finish, generate_reel_script
from publish.reel_story import (
    Story, comparison_row, load_history, load_prev_ranks, pick_story, rank_history, season_lows,
)
from publish.render_graphics import (
    BLACK, DOWN, GOLD, GREY, HEADSHOTS_DIR, NAVY, UP, WHITE, _font, circle_headshot, draw_movement, fit_text,
    load_headshot, load_headshot_placeholder, load_logo, placeholder_background, player_stats, season_label,
    text_w, wrap_text,
)
from publish.stats import REEL_LABELS, format_stat, per_game, rate_cols, stat_cols

W, H = REEL_SIZE
CX = W // 2
SIDE = max(REEL_SAFE_LEFT, REEL_SAFE_RIGHT)       # symmetric margins keep things centred
X0, X1 = SIDE, W - SIDE                           # 120 … 960
Y0, Y1 = REEL_SAFE_TOP, H - REEL_SAFE_BOTTOM      # 250 … 1520
MAX_W = X1 - X0
XFADE = 0.15                                      # crossfade between scenes, seconds (none into the first)
DURATION = REEL_SCENES[-1][2] + REEL_HOLD_S

F = lru_cache(maxsize=None)(_font)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def prog(t: float, start: float, dur: float) -> float:
    return ease((t - start) / dur) if dur > 0 else float(t >= start)


def with_alpha(img: Image.Image, a: float) -> Image.Image:
    if a >= 1:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: int(v * max(0.0, a))))
    return out


def put(canvas: Image.Image, img: Image.Image, xy, a: float = 1.0):
    if a <= 0:
        return
    x, y = int(xy[0]), int(xy[1])
    canvas.alpha_composite(with_alpha(img, a), (x, y)) if 0 <= x and 0 <= y and \
        x + img.width <= canvas.width and y + img.height <= canvas.height else \
        _put_clipped(canvas, with_alpha(img, a), x, y)


def _put_clipped(canvas, img, x, y):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(img, (x, y))
    canvas.alpha_composite(layer)


@lru_cache(maxsize=None)
def text_img(text: str, font_key: str, size: int, fill=WHITE, stroke: int = 0, stroke_fill=BLACK) -> Image.Image:
    """Tightly cropped RGBA image of one line of text (cached)."""
    font = F(font_key, size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (max(1, r - l), max(1, b - t)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return img


def put_text_c(canvas, cx, y, text, font_key, size, fill=WHITE, stroke=0, a=1.0, scale=1.0):
    img = text_img(text, font_key, size, fill, stroke)
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    put(canvas, img, (cx - img.width // 2, y - img.height // 2), a)
    return img.height


def fit_size(text: str, font_key: str, start: int, max_w: int, min_size: int = 20) -> int:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    size = start
    while size > min_size and text_w(probe, text, F(font_key, size)) > max_w:
        size -= 2
    return size


def wrap_fit(text: str, font_key: str, start: int, max_w: int, max_lines: int, min_size: int = 40):
    """Largest size (from start down) at which text wraps into max_lines lines without ellipsis."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    size = start
    while True:
        lines = wrap_text(probe, text, F(font_key, size), max_w, max_lines)
        if (" ".join(lines) == " ".join(text.split()) and all(text_w(probe, l, F(font_key, size)) <= max_w for l in lines)) \
                or size <= min_size:
            return size, lines
        size -= 4


def panel(size, radius, fill, outline=None, width=0) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=fill,
                                          outline=outline, width=width)
    return img


@lru_cache(maxsize=None)
def headshot(player_key: str, width: int) -> Image.Image:
    return load_headshot(player_key, width) or load_headshot_placeholder(width)


@lru_cache(maxsize=None)
def headshot_circle(player_key: str, d: int) -> Image.Image:
    try:
        img = Image.open(HEADSHOTS_DIR / f"{player_key}.png").convert("RGBA")
    except Exception:
        img = load_headshot_placeholder(600)
    ring = Image.new("RGBA", (d + 8, d + 8), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, d + 7, d + 7), fill=(20, 20, 20, 255), outline=(*GOLD, 255), width=4)
    ring.alpha_composite(circle_headshot(img, d), (4, 4))
    return ring


@lru_cache(maxsize=None)
def logo(team: str, size: int) -> Image.Image:
    return load_logo(team, size)


def arrow(up: bool, size: int, color) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pts = [(0, size), (size, size), (size // 2, 0)] if up else [(0, 0), (size, 0), (size // 2, size)]
    ImageDraw.Draw(img).polygon(pts, fill=(*color, 255))
    return img


ARROW_WORD = "→"     # in hook text; the reel fonts have no arrow glyph, so it's drawn


def _arrow_right(size: int, color=GOLD) -> Image.Image:
    """A right arrow sized to a line of text at font size `size` (shaft + head), with a black outline."""
    w, h = int(size * 0.72), int(size * 0.5)
    img = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    sh = max(6, h // 4)
    pts = [(4, 4 + h // 2 - sh // 2), (4 + int(w * 0.55), 4 + h // 2 - sh // 2), (4 + int(w * 0.55), 4),
           (4 + w, 4 + h // 2), (4 + int(w * 0.55), 4 + h), (4 + int(w * 0.55), 4 + h // 2 + sh // 2),
           (4, 4 + h // 2 + sh // 2)]
    d.polygon(pts, fill=(*color, 255), outline=(*BLACK, 255), width=4)
    return img


def _word_w(word: str, size: int) -> int:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    return _arrow_right(size).width if word == ARROW_WORD else text_w(probe, word, F("agency_bold", size))


def hook_layout(text: str, max_w: int, max_lines: int, start: int, min_size: int):
    """Largest size (from start down) at which the hook wraps into max_lines lines; "→" counts as a word."""
    words = text.split()
    size = start
    while True:
        space = _word_w(" ", size) or size // 4
        lines, cur, cur_w = [], [], 0
        for w in words:
            ww = _word_w(w, size)
            if cur and cur_w + space + ww > max_w:
                lines.append(" ".join(cur))
                cur, cur_w = [w], ww
            else:
                cur_w = cur_w + (space if cur else 0) + ww
                cur.append(w)
        lines.append(" ".join(cur))
        if (len(lines) <= max_lines and all(_word_w(w, size) <= max_w for w in words)) or size <= min_size:
            return size, lines[:max_lines]
        size -= 4


def value_img(value: str, size: int, max_w: int) -> Image.Image:
    """A card value: plain ("28.2") or then → now ("1-4 → 9-6"), the "now" in white, the "then" in grey,
    shrunk to fit max_w."""
    while True:
        if ARROW_WORD not in value:
            img = text_img(value, "agency_bold", size, WHITE)
        else:
            then, now = (x.strip() for x in value.split(ARROW_WORD, 1))
            parts = [text_img(then, "agency_bold", int(size * 0.8), GREY), _arrow_right(int(size * 0.8)),
                     text_img(now, "agency_bold", size, WHITE)]
            gap = max(10, size // 6)
            h = max(i.height for i in parts)
            img = Image.new("RGBA", (sum(i.width for i in parts) + gap * 2, h), (0, 0, 0, 0))
            x = 0
            for i in parts:
                img.alpha_composite(i, (x, h - i.height if i is not parts[1] else (h - i.height) // 2))
                x += i.width + gap
        if img.width <= max_w or size <= 48:
            return img
        size -= 6


def hook_line_img(line: str, size: int) -> Image.Image:
    """One line of hook text: white with a black stroke, numbers ("No. 30") in gold, "→" drawn."""
    parts = []
    for w in line.split():
        if w == ARROW_WORD:
            parts.append(_arrow_right(size))
        else:
            gold = any(ch.isdigit() for ch in w)
            parts.append(text_img(w, "agency_bold", size, GOLD if gold else WHITE, 5))
    space = max(12, size // 4)
    h = max(i.height for i in parts)
    img = Image.new("RGBA", (sum(i.width for i in parts) + space * (len(parts) - 1), h), (0, 0, 0, 0))
    x = 0
    for i in parts:
        img.alpha_composite(i, (x, h - i.height if i.height > size * 0.6 else (h - i.height) // 2))
        x += i.width + space
    return img


# ─────────────────────────────────────────────────────────────────
# Static layers: rink background and the STINGER / HART TROPHY WATCH banner
# ─────────────────────────────────────────────────────────────────
def _source_board() -> Image.Image:
    if BOARD_BACKGROUND.exists():
        return Image.open(BOARD_BACKGROUND).convert("RGB").resize((1080, 1350), Image.LANCZOS)
    return placeholder_background().convert("RGB")


@lru_cache(maxsize=1)
def background() -> Image.Image:
    """The board's rink (below its header band) scaled to 1080×1920 and tinted navy, so white text reads on
    the light ice; darker top/bottom bands where the platform UI sits."""
    turf = _source_board().crop((0, 240, 1080, 1350))
    scale = H / turf.height
    turf = turf.resize((round(turf.width * scale), H), Image.LANCZOS)
    left = (turf.width - W) // 2
    turf = turf.crop((left, 0, left + W, H))
    arr = np.asarray(turf).astype(np.float32) * 0.38 + np.array(NAVY, dtype=np.float32) * 0.62
    y = np.arange(H, dtype=np.float32)[:, None]
    shade = 1 - 0.45 * np.clip((Y0 - y) / Y0, 0, 1) - 0.45 * np.clip((y - Y1) / (H - Y1), 0, 1)
    arr = (arr * shade[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")


@lru_cache(maxsize=None)
def banner(width: int) -> Image.Image:
    """The board graphic's header (bees, STINGER, HART TROPHY WATCH) as a badge: rounded corners and a gold
    border. The light ice behind it can't be keyed out the way the NFL reel keys its turf."""
    hdr = _source_board().convert("RGBA").crop((0, 0, 1080, 228))
    hdr = hdr.resize((width, round(hdr.height * width / 1080)), Image.LANCZOS)
    mask = Image.new("L", hdr.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, hdr.width - 1, hdr.height - 1), radius=28, fill=255)
    hdr.putalpha(mask)
    ImageDraw.Draw(hdr).rounded_rectangle((0, 0, hdr.width - 1, hdr.height - 1), radius=28,
                                          outline=(*GOLD, 255), width=5)
    return hdr


# ─────────────────────────────────────────────────────────────────
# Scenes. Each draws onto a fresh copy of the background at local time t (seconds into the scene).
# ─────────────────────────────────────────────────────────────────
class Reel:
    def __init__(self, story: Story, board: pd.DataFrame, cards: List[tuple], script: Optional[dict],
                 season: int, week: int, featured_movement: bool = True):
        """cards: [(label, value)] for the Proof scene. featured_movement=False hides the featured player's
        week-over-week arrow in the top-5 list (a one-off story tells its own before/after)."""
        self.s, self.board, self.cards, self.script = story, board.sort_values("rank"), cards, script
        self.featured_movement = featured_movement
        self.season, self.week = season, week
        self.p = story.player
        self.key = str(self.p["player_key"])
        self.question = ((script or {}).get("question") or fallback_question(story)).strip()
        self.then_week = None                # set by make_reel when the cards are then → now
        self.scenes = [(name, start, end, getattr(self, f"scene_{name}")) for name, start, end in REEL_SCENES]

    def record_line(self) -> str:
        """Under the stat cards: the team's record (W-L-OTL) and points, as on the board."""
        line = f"{self.p['team']} {self.p['team_record']}"
        try:
            line += f" · {int(self.p['team_points'])} PTS"
        except (KeyError, TypeError, ValueError):
            pass
        return line.upper()

    # 1. Hook: everything is on screen from frame 0 (no fade-ins). The rank flips by 0.8 s, then punches.
    def scene_hook(self, c: Image.Image, t: float):
        hook = self.s.hook.upper().replace("NO. ", "No. ")      # "No. 3", as on the rank badge
        size, lines = hook_layout(hook, MAX_W, 2, 124, 76)
        lh = int(size * 1.08)
        y = Y0 + 20
        for line in lines:
            img = hook_line_img(line, size)
            put(c, img, (CX - img.width // 2, y))
            y += lh
        top = max(y + 30, Y0 + 300)

        hs = headshot(self.key, 600)
        put(c, hs, (CX - hs.width // 2, top))
        lg = logo(str(self.p["team"]), 210)
        put(c, lg, (X1 - lg.width + 10, top - 10), 0.95)
        bar_y = top + hs.height
        name = str(self.p["player_name"]).upper()
        put(c, panel((MAX_W, 104), 14, (0, 0, 0, 230)), (X0, bar_y))
        put_text_c(c, CX, bar_y + 52, name, "agency_bold", fit_size(name, "agency_bold", 78, MAX_W - 40), WHITE)

        prev, cur = self.s.prev_rank, self.s.cur_rank
        moving = prev is not None and prev != cur
        y_rank = bar_y + 104 + 150
        if moving:
            shown = prev if t < 0.45 else round(prev + (cur - prev) * prog(t, 0.45, 0.35))
            caption = self.s.labels[0] if t < 0.8 else self.s.labels[1]
            scale = 1.0 + 0.22 * (1 - prog(t, 0.8, 0.35)) if t >= 0.8 else 1.0
        else:
            shown, scale = cur, 1.0
            caption = ("HOLDS No. 1" if prev == cur == 1 else "SAME SPOT AS LAST WEEK" if prev == cur
                       else "FIRST BOARD OF THE SEASON" if self.s.reason.startswith("no previous")
                       else "NEW TO OUR TOP 30")
        put_text_c(c, CX, y_rank - 112, caption, "fg_demi", 42, GOLD)
        put_text_c(c, CX, y_rank, f"No. {shown}", "agency_bold", 180, GOLD, 6, 1.0, scale)
        if moving and t >= 0.8:
            up = cur < prev
            n = abs(prev - cur)
            txt = text_img(f"{'UP' if up else 'DOWN'} {n} SPOT{'S' if n != 1 else ''}", "agency_bold", 60,
                           UP if up else DOWN, 3)
            ar = arrow(up, 50, UP if up else DOWN)
            x = CX - (ar.width + 16 + txt.width) // 2
            a = prog(t, 0.8, 0.15)
            put(c, ar, (x, y_rank + 118), a)
            put(c, txt, (x + ar.width + 16, y_rank + 118 + (ar.height - txt.height) // 2), a)
        mark = banner(220)
        put(c, mark, (CX - mark.width // 2, Y1 - mark.height), 0.9)

    # 2. Proof: the stat cards punch in, then the team's record and the board's one-line why
    def scene_proof(self, c: Image.Image, t: float):
        ring = headshot_circle(self.key, 150)
        put(c, ring, (X0, Y0 + 20))
        nx = X0 + ring.width + 30
        name = str(self.p["player_name"]).upper()
        put(c, text_img(name, "agency_bold", fit_size(name, "agency_bold", 64, X1 - nx)), (nx, Y0 + 40))
        meta = f"No. {self.s.cur_rank} · {self.p['team']} · {self.p['position']}"
        put(c, text_img(meta, "fg_demi", 40, GOLD), (nx, Y0 + 120))

        step, y = 0.45, 470
        if self.then_week is not None:
            lab = hook_line_img(f"WEEK {self.then_week} {ARROW_WORD} WEEK {self.week}", 46)
            put(c, lab, (CX - lab.width // 2, y - 14), prog(t, 0, 0.2))
            y += 50
        for i, (label, value) in enumerate(self.cards):
            ai = prog(t, 0.15 + step * i, 0.25)
            if ai <= 0:
                continue
            card = panel((MAX_W, 150), 18, (10, 10, 10, 220))
            ImageDraw.Draw(card).rounded_rectangle((0, 0, 14, 149), radius=7, fill=(*GOLD, 255))
            val = value_img(str(value), 104, MAX_W - 300)
            lab_w = MAX_W - val.width - 124
            lab = text_img(str(label).upper(), "fg_demi", fit_size(str(label).upper(), "fg_demi", 44, lab_w), GOLD)
            card.alpha_composite(lab, (44, (150 - lab.height) // 2))
            card.alpha_composite(val, (MAX_W - val.width - 40, (150 - val.height) // 2))
            k = 1.06 - 0.06 * ai                                   # a small punch as it lands
            card = card.resize((int(card.width * k), int(card.height * k)), Image.LANCZOS)
            put(c, card, (CX - card.width // 2, y + i * 170 + (150 - card.height) // 2), ai)
        y_rec = y + len(self.cards) * 170 + 40
        t_rec = 0.15 + step * len(self.cards)
        if self.then_week is None:           # then → now cards already carry the record
            rec = self.record_line()
            put_text_c(c, CX, y_rec, rec, "agency_bold", fit_size(rec, "agency_bold", 60, MAX_W), WHITE, 4, prog(t, t_rec, 0.25))
        else:
            y_rec -= 60
        # the board's one-line why lists stats; with then → now cards the narration explains the move instead
        if self.then_week is None and isinstance(self.p.get("why_blurb"), str) and self.p["why_blurb"].strip():
            size, lines = wrap_fit(self.p["why_blurb"], "fg_demi", 46, MAX_W, 3, 34)
            for j, line in enumerate(lines):
                put_text_c(c, CX, y_rec + 80 + j * int(size * 1.2), line, "fg_demi", size, GREY, 3,
                           prog(t, t_rec + 0.5, 0.3))

    # 3. Context: compact top 5, featured player highlighted (shown as a 6th row if outside the top 5)
    def scene_context(self, c: Image.Image, t: float):
        put_text_c(c, CX, Y0 + 40, "TOP 5 THIS WEEK", "agency_bold", 76, GOLD, 4)
        rows = list(self.board.head(5).iterrows())
        extra = int(self.p["rank"]) > 5
        if extra:
            rows.append((None, self.p))
        row_h, y = 168, Y0 + 110
        for i, (_, r) in enumerate(rows):
            if extra and i == 5:
                put_text_c(c, CX, y + 14, "· · ·", "agency_bold", 48, GREY, 0, prog(t, 0.45, 0.2))
                y += 40
            ai = prog(t, 0.05 + 0.08 * i, 0.25)
            put(c, self._row(r, str(r["player_key"]) == self.key), (X0 + int(50 * (1 - ai)), y), ai)
            y += row_h

    def _row(self, r: pd.Series, featured: bool) -> Image.Image:
        h = 156
        fg = BLACK if featured else WHITE
        row = panel((MAX_W, h), 18, (*GOLD, 240) if featured else (10, 10, 10, 205))
        d = ImageDraw.Draw(row)
        rk = text_img(str(int(r["rank"])), "agency_bold", 84, fg)
        row.alpha_composite(rk, (60 - rk.width // 2, (h - rk.height) // 2))
        ring = headshot_circle(str(r["player_key"]), 124)
        row.alpha_composite(ring, (112, (h - ring.height) // 2))
        tx = 112 + ring.width + 24
        name = str(r["player_name"]).upper()
        nm = text_img(name, "agency_bold", fit_size(name, "agency_bold", 54, MAX_W - tx - 130), fg)
        row.alpha_composite(nm, (tx, 30))
        meta = text_img(f"{r['team']} · {r['position']}", "fg_demi", 36, (60, 60, 60) if featured else GREY)
        row.alpha_composite(meta, (tx, 98))
        mv = str(r.get("movement", ""))
        if featured and not self.featured_movement:
            return row
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        mw = draw_movement(probe, 0, 0, mv, 44)
        mx = MAX_W - mw - 34
        if featured:     # green/red arrows don't read on gold: give them a dark pill
            d.rounded_rectangle((mx - 16, (h - 64) // 2, mx + mw + 16, (h + 64) // 2), radius=14, fill=(15, 15, 15, 235))
        draw_movement(d, mx, (h - 52) // 2, mv, 44)
        return row

    # 4. Ask: the comment question, then the brand and URL small (held for REEL_HOLD_S at the end).
    #    Every animation is done by 1.0 s, when encode() starts holding the frame.
    def scene_ask(self, c: Image.Image, t: float):
        ring = headshot_circle(self.key, 220)
        put(c, ring, (CX - ring.width // 2, Y0 + 20))
        q = self.question.upper().replace("NO. ", "No. ")             # "No. 1", as everywhere else
        size, lines = wrap_fit(q, "agency_bold", 120, MAX_W, 4, 70)
        lh = int(size * 1.1)
        y = Y0 + 20 + ring.height + 60
        k = 1.08 - 0.08 * prog(t, 0, 0.3)
        for i, line in enumerate(lines):
            put_text_c(c, CX, y + i * lh + lh // 2, line, "agency_bold", size, WHITE, 5, 1.0, k)
        y = y + len(lines) * lh + 50
        a = prog(t, 0.35, 0.25)
        lab = text_img("COMMENT BELOW", "agency_bold", 64, GOLD, 3)
        ar = arrow(False, 48, GOLD)
        bob = int(12 * np.sin(np.pi * min(1.0, max(0.0, (t - 0.35) / 0.6))))
        x = CX - (lab.width + 18 + ar.width) // 2
        put(c, lab, (x, y), a)
        put(c, ar, (x + lab.width + 18, y + (lab.height - ar.height) // 2 + bob), a)
        mark = banner(520)
        yb = Y1 - mark.height - 70
        put(c, mark, (CX - mark.width // 2, yb), prog(t, 0.5, 0.3))
        put_text_c(c, CX, yb + mark.height + 40, "stingercollectibles.com", "fg_demi", 44, WHITE, 0, prog(t, 0.6, 0.3))

    # Frame at absolute time t; a short crossfade into each scene after the first
    def frame(self, t: float) -> Image.Image:
        idx = next((i for i, (_, s, e, _) in enumerate(self.scenes) if t < e), len(self.scenes) - 1)
        name, start, end, draw = self.scenes[idx]
        c = background().copy()
        draw(c, t - start)
        if idx > 0 and t - start < XFADE:
            _, ps, pe, pdraw = self.scenes[idx - 1]
            prev = background().copy()
            pdraw(prev, pe - ps)
            c = Image.blend(prev, c, (t - start) / XFADE)
        return c


# ─────────────────────────────────────────────────────────────────
# Video, frames and text outputs
# ─────────────────────────────────────────────────────────────────
def encode(reel: Reel, out: Path) -> Path:
    import imageio_ffmpeg
    n = round(DURATION * REEL_FPS)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(REEL_FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_scene_static = REEL_SCENES[-1][1] + 1.0      # the ask's animations are done by then
    held = None
    try:
        for i in range(n):
            t = i / REEL_FPS
            if t >= last_scene_static:
                held = held or reel.frame(t).convert("RGB").tobytes()
                proc.stdin.write(held)
            else:
                proc.stdin.write(reel.frame(t).convert("RGB").tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return out


def guides(img: Image.Image) -> Image.Image:
    g = img.copy()
    layer = Image.new("RGBA", g.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    red = (255, 0, 0, 90)
    d.rectangle((0, 0, W, Y0), fill=red)
    d.rectangle((0, Y1, W, H), fill=red)
    d.rectangle((W - REEL_SAFE_RIGHT, Y0, W, Y1), fill=red)
    d.rectangle((0, Y0, REEL_SAFE_LEFT, Y1), fill=red)
    d.rectangle((X0, Y0, X1, Y1), outline=(0, 255, 255, 200), width=3)
    g.alpha_composite(layer)
    return g


def save_frames(reel: Reel, frames_dir: Path) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    first = reel.frame(0)
    first.convert("RGB").save(frames_dir / "0_first_frame.png")
    guides(first).convert("RGB").save(frames_dir / "0_first_frame_guides.png")
    out = [frames_dir / "0_first_frame.png"]
    for i, (name, start, end, _) in enumerate(reel.scenes, 1):
        img = reel.frame(end - 0.05)
        p = frames_dir / f"{i}_{name}.png"
        img.convert("RGB").save(p)
        guides(img).convert("RGB").save(frames_dir / f"{i}_{name}_guides.png")
        out.append(p)
    return out


def script_text(story: Story, script: Optional[dict], season: int, week: int, cards: List[tuple]) -> str:
    p = story.player
    prev = f"No. {story.prev_rank}" if story.prev_rank else "NEW"
    lines = [f"NHL HART REEL — WEEK {week}, {season_label(season)}" + ("" if story.labels[0] == "LAST WEEK" else f" ({story.story_type})"),
             f"Story: {story.story_type} — {story.hook}",
             f"{p['player_name']} ({p['team']} {p['position']}): {prev} -> No. {story.cur_rank}",
             f"Why this story: {story.reason}",
             f"Video: {DURATION} s ({REEL_SCENES[-1][2]} s + {REEL_HOLD_S} s hold). Post this first; the board posts the next day.",
             "Record over the video in Instagram; add the Captions sticker (most people watch muted).",
             ""]
    if not script:
        lines += ["(No narration: the script step was skipped or failed.)",
                  f"Stat cards on screen: {', '.join(f'{l} {v}' for l, v in cards)}",
                  f"Question on screen: {fallback_question(story)}"]
        return "\n".join(lines) + "\n"
    lines.append(f"SCRIPT ({script['words']} words)")
    for i, b in enumerate(script["beats"], 1):
        lines += ["", f"SCENE {i} · {b['scene'].upper()} · 0:{b['start']:02d}–0:{b['end']:02d}", b["text"]]
    lines += ["", "QUESTION ON SCREEN", script["question"],
              "", "INSTAGRAM CAPTION", script["instagram_caption"],
              "", "HASHTAGS", " ".join(script["hashtags"])]
    if script.get("warnings"):
        lines += ["", "CHECK BEFORE RECORDING", *[f"- {w}" for w in script["warnings"]]]
    return "\n".join(lines) + "\n"


def _story_from_file(spec: dict, board: pd.DataFrame) -> Story:
    """A one-off story (a correction, a feature) instead of the weekly picker. See make_reel."""
    rows = board[board["player_name"].str.lower() == spec["player_name"].lower()]
    if rows.empty:
        raise SystemExit(f"{spec['player_name']} isn't on this board")
    row = rows.iloc[0]
    return Story(spec.get("story_type", "feature"), row, spec.get("prev_rank"), int(spec.get("cur_rank", row["rank"])),
                 spec["hook"], spec.get("note", ""), tuple(spec.get("labels", ("LAST WEEK", "THIS WEEK"))))


def _prepared_script(spec: dict, story: Story) -> dict:
    """A hand-written script from the story file (beats for REEL_SCENES, question, instagram_caption,
    hashtags), checked and finished like a generated one."""
    data = {k: spec["script"].get(k, "" if k != "hashtags" else []) for k in ("beats", "question", "instagram_caption", "hashtags")}
    return finish(data, story)


def then_now_cards(p: pd.Series, then: pd.Series) -> List[tuple]:
    """Proof cards as then → now: the team's record (skipped for a player traded since then), then the rates for
    his position that moved most since then (relative change, the ones that
    went the same way as his rank first): points/goals/assists a game, or a goalie's save %
    and GAA. Rates, not season totals, which only go up."""
    p, then = per_game(p), per_game(then)
    cards = []
    if str(then.get("team")) == str(p.get("team")):
        cards.append((f"{p['team']} record", f"{then['team_record']} {ARROW_WORD} {p['team_record']}"))
    cols = [c for c in rate_cols(p) if c in then and c in p and pd.notna(then[c]) and pd.notna(p[c])]

    up = int(p["rank"]) <= int(then["rank"])           # a climb (or a hold): show the rates that improved first

    def moved(c):
        a, b = float(then[c]), float(p[c])
        better = (b < a) if c == "gaa" else (b > a)
        return (better == up, abs(b - a) / max(abs(a), 1e-9))

    for c in sorted(cols, key=moved, reverse=True)[: 3 - len(cards)]:
        cards.append((REEL_LABELS.get(c, c), f"{format_stat(c, then[c])} {ARROW_WORD} {format_stat(c, p[c])}"))
    return cards


def make_reel(outdir: Path = CURRENT_PRED_DIR, frames_only: bool = False, write_script: bool = True,
              story_file: Optional[Path] = None) -> Path:
    """The weekly reel, or with story_file a one-off. The story file (JSON) names the board player and the hook,
    and may set story_type, prev_rank/cur_rank, labels (over the rank, before/after), note (facts for the
    script), cards ([[label, value]] for the Proof scene, instead of stat cells), script (hand-written beats for
    REEL_SCENES, question, instagram_caption, hashtags; else one is generated) and name. A one-off writes
    reel/reel_{name}.mp4, _script.txt and _meta.json, leaving the weekly files alone."""
    outdir = Path(outdir)
    board = pd.read_csv(outdir / "board_hart_latest.csv")
    season, week = int(board["season"].iloc[0]), int(board["week"].iloc[0])
    spec = json.loads(Path(story_file).read_text(encoding="utf-8")) if story_file else None
    # Earlier weeks' boards (board_hart_wkNN.csv): who has held No. 1, and the featured player's path this season
    history = load_history(outdir, week)
    past_no1 = frozenset(history.loc[history["rank"] == 1, "player_key"].astype(str))
    story = _story_from_file(spec, board) if spec else \
        pick_story(board, load_prev_ranks(outdir / ".prev_board_hart.csv"), past_no1, season_lows(history, week))
    path = rank_history(history, str(story.player["player_key"]))
    print(f"Reel story: {story.story_type} — {story.player['player_name']} "
          f"({'NEW' if story.prev_rank is None else f'No. {story.prev_rank}'} -> No. {story.cur_rank}). {story.reason}")

    pool = board[board["rank"] <= TOP_N]
    if int(story.player["rank"]) > TOP_N:
        pool = pd.concat([pool, story.player.to_frame().T])
    # the board card's four stat cells, so the two agree; win % is left out (the record shows it)
    # the board card's stat cells (3; a goalie gets his own set), win % left out (the record line shows it)
    chips = [c for c in player_stats(story.player, pool, 4) if c != "team_win_pct"][:3]
    order = stat_cols(story.player)
    chips = sorted(chips, key=lambda c: order.index(c) if c in order else 99)   # board order (points first)
    # Then vs now: his row on an earlier weekly board (the path hook's week, else up to REEL_LOOKBACK_WEEKS back)
    low = season_lows(history, week).get(str(story.player["player_key"]))
    prefer = week - low[1] if low and ARROW_WORD in story.hook else None
    then = None if spec else comparison_row(history, str(story.player["player_key"]), week, prefer_week=prefer,
                                                     prev_rank=story.prev_rank, cur_rank=story.cur_rank)
    if spec and spec.get("cards"):
        cards = [tuple(c) for c in spec["cards"]]
    elif then is not None:
        cards = then_now_cards(story.player, then)
    else:
        cards = [(REEL_LABELS.get(c, c), format_stat(c, story.player.get(c))) for c in chips]

    if spec and spec.get("script"):
        script = _prepared_script(spec, story)
    else:
        script = generate_reel_script(story, board, cards, season, week, path, then) if write_script else None
    reel = Reel(story, board, cards, script, season, week, featured_movement=spec is None)
    reel.then_week = None if then is None else int(then["week"])
    reel.through = pd.to_datetime(board["stats_through"].iloc[0]).strftime("%b %-d" if sys.platform != "win32" else "%b %#d")
    reel_dir = outdir / "reel"
    reel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reel_{spec.get('name', story.story_type)}" if spec else "reel"
    video_name = f"{stem}.mp4" if spec else "reel_latest.mp4"

    (reel_dir / f"{stem}_script.txt").write_text(script_text(story, script, season, week, cards), encoding="utf-8")
    p = story.player
    meta = {
        "sport": "nhl", "season": season, "through_week": week, "stats_through": str(board["stats_through"].iloc[0]),
        "story_type": story.story_type, "hook": story.hook, "reason": story.reason,
        "player": {k: str(p[k]) for k in ("player_name", "team", "position", "player_key")},
        "prev_rank": story.prev_rank, "cur_rank": story.cur_rank, "season_ranks": path,
        "compare_week": None if then is None else int(then["week"]),
        "stat_chips": [{"label": l, "value": v} for l, v in cards],
        "question": (script or {}).get("question") or fallback_question(story),
        "captions": [],                       # retired with the Nov 2026 rework (no on-screen caption chunks)
        "instagram_caption": (script or {}).get("instagram_caption", ""),
        "hashtags": (script or {}).get("hashtags", []),
        "script_model": ("hand-written" if spec and spec.get("script") else REEL_MODEL) if script else None,
        "script_words": (script or {}).get("words"),
        "warnings": (script or {}).get("warnings", []),
        "video": video_name, "duration_s": DURATION,
        # The workflow uploads the MP4 under this name to the repo's "reel-latest" release; the website's
        # social-drafts job downloads it by this name, so a failed upload can't send last week's video.
        "release_tag": "reel-latest",
        "release_asset": f"nhl_{stem}_{season}_wk{week:02d}.mp4" if spec else f"nhl_reel_{season}_wk{week:02d}.mp4",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }
    (reel_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Reel script -> {reel_dir / f'{stem}_script.txt'}")

    if frames_only:
        paths = save_frames(reel, reel_dir / ("frames" if not spec else f"frames_{stem}"))
        print(f"Reel frames -> {paths[0].parent}")
        return paths[0].parent
    out = encode(reel, reel_dir / video_name)
    print(f"Reel -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=CURRENT_PRED_DIR)
    ap.add_argument("--frames-only", action="store_true", help="save one PNG per scene instead of the video")
    ap.add_argument("--no-script", action="store_true", help="skip the narration API call")
    ap.add_argument("--story", type=Path, default=None,
                    help="one-off reel from a story file (JSON; see make_reel) instead of the weekly picker")
    a = ap.parse_args()
    make_reel(a.outdir, a.frames_only, not a.no_script, a.story)


if __name__ == "__main__":
    main()
