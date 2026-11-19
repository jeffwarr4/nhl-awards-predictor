"""
Weekly Hart Trophy reel (ported from nfl-awards-predictor/src/publish/render_reel.py): one story from the board as a silent 1080×1920 MP4 (25 s + a 2 s hold) and a narration
script Jeff records over it in CapCut. Posted the day after the static board.

    python -m publish.render_reel                       # predictions/{CURRENT_YEAR}/
    python -m publish.render_reel --outdir <replay dir> # a replay folder
    python -m publish.render_reel --frames-only         # one PNG per scene (+ safe-zone guides), no video
    python -m publish.render_reel --no-script           # skip the API call (no narration, no captions)

Reads board_hart_latest.csv and .prev_board_hart.csv (movement baseline) from the folder, picks the story
(reel_story.pick_story), writes the script (reel_script, REEL_MODEL), renders frames with Pillow and pipes
them to ffmpeg (imageio-ffmpeg's bundled binary): H.264, 30 fps, yuv420p, +faststart, no audio.

Outputs ({outdir}/reel/):
  reel_latest.mp4    not committed (.gitignore); the workflow uploads it to the "reel-latest" release
                     as nhl_reel_{season}_wk{NN}.mp4 (reel_meta.json: release_asset)
  reel_script.txt    scene-by-scene script, readable on a phone
  reel_meta.json     story, player, ranks, caption, hashtags, week
  frames/            --frames-only: scene PNGs and *_guides.png with the platform UI zones shaded

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
from publish.reel_script import generate_reel_script
from publish.reel_story import Story, load_history, load_prev_ranks, pick_story, rank_history
from publish.render_graphics import (
    BLACK, DOWN, GOLD, GREY, HEADSHOTS_DIR, NAVY, UP, WHITE, _font, circle_headshot, draw_movement, fit_text,
    load_headshot, load_headshot_placeholder, load_logo, placeholder_background, player_stats, season_label,
    text_w, wrap_text,
)
from publish.stats import STAT_LABELS, format_stat

W, H = REEL_SIZE
CX = W // 2
SIDE = max(REEL_SAFE_LEFT, REEL_SAFE_RIGHT)       # symmetric margins keep things centred
X0, X1 = SIDE, W - SIDE                           # 120 … 960
Y0, Y1 = REEL_SAFE_TOP, H - REEL_SAFE_BOTTOM      # 250 … 1520
MAX_W = X1 - X0
XFADE = 0.25                                      # crossfade between scenes, seconds
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


# ─────────────────────────────────────────────────────────────────
# Static layers: turf background and the STINGER / MVP WATCH banner
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
        """cards: [(label, value)] for the Why scene. featured_movement=False hides the featured player's
        week-over-week arrow in the top-5 list (a one-off story tells its own before/after)."""
        self.s, self.board, self.cards, self.script = story, board.sort_values("rank"), cards, script
        self.featured_movement = featured_movement
        self.season, self.week = season, week
        self.p = story.player
        self.key = str(self.p["player_key"])
        self.captions = (script or {}).get("captions") or []
        self.scenes = [(name, start, end, getattr(self, f"scene_{name}")) for name, start, end in REEL_SCENES]

    # 1. Hook
    def scene_hook(self, c: Image.Image, t: float):
        bn = banner(880)
        put(c, bn, (CX - bn.width // 2, Y0 + 10), prog(t, 0, 0.4))
        label = f"{season_label(self.season)} HART WATCH · WEEK {self.week}"
        lab = text_img(label, "agency_bold", 46, BLACK)
        pill = panel((lab.width + 60, lab.height + 34), 16, (*GOLD, 245), (*BLACK, 255), 3)
        pill.alpha_composite(lab, (30, 17))
        put(c, pill, (CX - pill.width // 2, 520), prog(t, 0.2, 0.4))

        size, lines = wrap_fit(self.s.hook.upper(), "agency_bold", 132, MAX_W, 3, 80)
        a, k = prog(t, 0.5, 0.45), 0.88 + 0.12 * prog(t, 0.5, 0.45)
        line_h = int(size * 1.12)
        y = 960 - (len(lines) * line_h) // 2
        for i, line in enumerate(lines):
            put_text_c(c, CX, y + i * line_h + line_h // 2, line, "agency_bold", size, WHITE, 5, a, k)
        bar_w = int(420 * prog(t, 1.0, 0.6))
        if bar_w > 0:
            put(c, panel((bar_w, 12), 6, (*GOLD, 255)), (CX - bar_w // 2, y + len(lines) * line_h + 30))

    # 2. Player: headshot, logo, rank counting from last week's to this week's
    def scene_player(self, c: Image.Image, t: float):
        a, dy = prog(t, 0, 0.5), int(60 * (1 - prog(t, 0, 0.5)))
        lg = logo(str(self.p["team"]), 300)
        put(c, lg, (X1 - lg.width + 20, Y0 + 20 + dy), a * 0.95)
        hs = headshot(self.key, 740)
        put(c, hs, (CX - hs.width // 2, Y0 + 60 + dy), a)
        hs_bot = Y0 + 60 + hs.height

        a2 = prog(t, 0.3, 0.4)
        name = str(self.p["player_name"]).upper()
        bar = panel((MAX_W, 104), 14, (0, 0, 0, 225))
        put(c, bar, (X0, hs_bot), a2)
        put_text_c(c, CX, hs_bot + 52, name, "agency_bold", fit_size(name, "agency_bold", 78, MAX_W - 40), WHITE, 0, a2)
        put_text_c(c, CX, hs_bot + 140, f"{self.p['team']} · {self.p['position']}", "fg_demi", 44, GREY, 0, a2)

        prev, cur = self.s.prev_rank, self.s.cur_rank
        y_rank = hs_bot + 330
        moving = prev is not None and prev != cur
        if moving:
            shown = round(prev + (cur - prev) * prog(t, 1.0, 1.6))
            caption = self.s.labels[0] if t < 1.0 else self.s.labels[1]
        else:
            shown = cur
            caption = ("HOLDS No. 1" if prev == cur == 1 else "SAME SPOT AS LAST WEEK" if prev == cur
                       else "FIRST BOARD OF THE SEASON" if self.s.reason.startswith("no previous")
                       else "NEW TO OUR TOP 30")
        a3 = prog(t, 0.6, 0.4)
        put_text_c(c, CX, y_rank - 120, caption, "fg_demi", 42, GOLD, 0, a3)
        put_text_c(c, CX, y_rank, f"No. {shown}", "agency_bold", 190, GOLD, 6, a3)
        if moving:
            a4 = prog(t, 2.7, 0.35)
            up = cur < prev
            n = abs(prev - cur)
            txt = text_img(f"{'UP' if up else 'DOWN'} {n} SPOT{'S' if n != 1 else ''}", "agency_bold", 64,
                           UP if up else DOWN, 3)
            ar = arrow(up, 54, UP if up else DOWN)
            x = CX - (ar.width + 18 + txt.width) // 2
            put(c, ar, (x, y_rank + 130), a4)
            put(c, txt, (x + ar.width + 18, y_rank + 130 + (ar.height - txt.height) // 2), a4)

    # 3. Why: stat chips one at a time, caption chunks underneath
    def scene_why(self, c: Image.Image, t: float):
        a = prog(t, 0, 0.35)
        ring = headshot_circle(self.key, 150)
        put(c, ring, (X0, Y0 + 20), a)
        nx = X0 + ring.width + 30
        name = str(self.p["player_name"]).upper()
        put(c, text_img(name, "agency_bold", fit_size(name, "agency_bold", 64, X1 - nx)), (nx, Y0 + 40), a)
        meta = f"No. {self.s.cur_rank} · {self.p['team']} · {self.p['position']}"
        put(c, text_img(meta, "fg_demi", 40, GOLD), (nx, Y0 + 120), a)

        for i, (label, value) in enumerate(self.cards):
            ai = prog(t, 0.4 + 1.0 * i, 0.5)
            dx = int(-90 * (1 - ai))
            y = 470 + i * 170
            card = panel((MAX_W, 150), 18, (10, 10, 10, 215))
            ImageDraw.Draw(card).rounded_rectangle((0, 0, 14, 149), radius=7, fill=(*GOLD, 255))
            val = text_img(str(value), "agency_bold", 104, WHITE)
            lab_w = MAX_W - val.width - 124
            lab = text_img(str(label).upper(), "fg_demi", fit_size(str(label).upper(), "fg_demi", 44, lab_w), GOLD)
            card.alpha_composite(lab, (44, (150 - lab.height) // 2))
            card.alpha_composite(val, (MAX_W - val.width - 40, (150 - val.height) // 2))
            put(c, card, (X0 + dx, y), ai)

        y_cap = 470 + max(3, len(self.cards)) * 170 + 150
        if self.captions:
            seg = (REEL_SCENES[2][2] - REEL_SCENES[2][1] - 0.4) / len(self.captions)
            i = min(len(self.captions) - 1, max(0, int((t - 0.4) // seg)))
            ac = prog(t, 0.4 + i * seg, 0.25)
            # one line when it fits at a readable size; a two-line break mid-phrase reads badly
            size, lines = wrap_fit(self.captions[i], "agency_bold", 84, MAX_W, 1, 64)
            if fit_text(ImageDraw.Draw(Image.new("RGBA", (1, 1))), self.captions[i], F("agency_bold", size), MAX_W) != self.captions[i]:
                size, lines = wrap_fit(self.captions[i], "agency_bold", 84, MAX_W, 2, 56)
            lh = int(size * 1.15)
            for j, line in enumerate(lines):
                put_text_c(c, CX, y_cap + j * lh, line, "agency_bold", size, WHITE, 5, ac)
        elif isinstance(self.p.get("why_blurb"), str) and self.p["why_blurb"].strip():
            size, lines = wrap_fit(self.p["why_blurb"], "fg_demi", 52, MAX_W, 3, 36)
            for j, line in enumerate(lines):
                put_text_c(c, CX, y_cap + j * int(size * 1.2), line, "fg_demi", size, WHITE, 3, prog(t, 3.4, 0.4))

    # 4. Context: compact top 5, featured player highlighted (shown as a 6th row if outside the top 5)
    def scene_context(self, c: Image.Image, t: float):
        put_text_c(c, CX, Y0 + 40, "TOP 5 THIS WEEK", "agency_bold", 76, GOLD, 4, prog(t, 0, 0.3))
        rows = list(self.board.head(5).iterrows())
        extra = int(self.p["rank"]) > 5
        if extra:
            rows.append((None, self.p))
        row_h, y = 168, Y0 + 110
        for i, (_, r) in enumerate(rows):
            if extra and i == 5:
                put_text_c(c, CX, y + 14, "· · ·", "agency_bold", 48, GREY, 0, prog(t, 0.7, 0.3))
                y += 40
            ai = prog(t, 0.1 + 0.12 * i, 0.35)
            put(c, self._row(r, str(r["player_key"]) == self.key), (X0 + int(60 * (1 - ai)), y), ai)
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

    # 5. Call to action (held for REEL_HOLD_S at the end)
    def scene_cta(self, c: Image.Image, t: float):
        a = prog(t, 0, 0.5)
        bn = banner(880)
        put(c, bn, (CX - bn.width // 2, Y0 + 60), a)
        put_text_c(c, CX, 780, "FULL BOARD AT", "fg_demi", 58, WHITE, 0, a)
        url = "stingercollectibles.com"
        put_text_c(c, CX, 880, url, "agency_bold", fit_size(url, "agency_bold", 112, MAX_W), GOLD, 5, prog(t, 0.2, 0.5))
        put_text_c(c, CX, 1030, "Top 30 + On the Bubble, every week", "fg_demi", 44, WHITE, 0, prog(t, 0.5, 0.5))
        put_text_c(c, CX, 1100, f"Stats through {getattr(self, 'through', '')}", "fg_demi", 40, GREY, 0, prog(t, 0.5, 0.5))

    # Frame at absolute time t, with a short crossfade into each scene
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
    last_scene_static = REEL_SCENES[-1][1] + 1.0      # the CTA's animations are done by then
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
    out = []
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
             f"Video: {DURATION} s ({REEL_SCENES[-1][2]} s + {REEL_HOLD_S} s hold). Post the day after the board.",
             ""]
    if not script:
        lines += ["(No narration: the script step was skipped or failed.)",
                  f"Stat cards on screen: {', '.join(f'{l} {v}' for l, v in cards)}"]
        return "\n".join(lines) + "\n"
    lines.append(f"SCRIPT ({script['words']} words)")
    for i, b in enumerate(script["beats"], 1):
        lines += ["", f"SCENE {i} · {b['scene'].upper()} · 0:{b['start']:02d}–0:{b['end']:02d}", b["text"]]
    lines += ["", "ON-SCREEN CAPTIONS (scene 3)", *[f"- {c}" for c in script["captions"]],
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


def _prepared_script(spec: dict) -> dict:
    """A hand-written script from the story file, checked like a generated one."""
    from publish.reel_script import problems, word_count
    data = {k: spec["script"][k] for k in ("beats", "captions", "instagram_caption", "hashtags")}
    windows = {name: (start, end) for name, start, end in REEL_SCENES}
    for b in data["beats"]:
        b["start"], b["end"] = windows[b["scene"]]
    data["words"] = word_count(data)
    data["warnings"] = problems(data)
    for w in data["warnings"]:
        print(f"  [WARN] Reel script: {w}")
    return data


def make_reel(outdir: Path = CURRENT_PRED_DIR, frames_only: bool = False, write_script: bool = True,
              story_file: Optional[Path] = None) -> Path:
    """The weekly reel, or with story_file a one-off. The story file (JSON) names the board player and the hook,
    and may set story_type, prev_rank/cur_rank, labels (over the rank, before/after), note (facts for the
    script), cards ([[label, value]] for the Why scene, instead of stat cells), script (hand-written beats,
    captions, instagram_caption, hashtags; else one is generated) and name. A one-off writes
    reel/reel_{name}.mp4, _script.txt and _meta.json, leaving the weekly files alone."""
    outdir = Path(outdir)
    board = pd.read_csv(outdir / "board_hart_latest.csv")
    season, week = int(board["season"].iloc[0]), int(board["week"].iloc[0])
    spec = json.loads(Path(story_file).read_text(encoding="utf-8")) if story_file else None
    # Earlier weeks' boards (board_hart_wkNN.csv): who has held No. 1, and the featured player's path this season
    history = load_history(outdir, week)
    past_no1 = frozenset(history.loc[history["rank"] == 1, "player_key"].astype(str))
    story = _story_from_file(spec, board) if spec else \
        pick_story(board, load_prev_ranks(outdir / ".prev_board_hart.csv"), past_no1)
    path = rank_history(history, str(story.player["player_key"]))
    print(f"Reel story: {story.story_type} — {story.player['player_name']} "
          f"({'NEW' if story.prev_rank is None else f'No. {story.prev_rank}'} -> No. {story.cur_rank}). {story.reason}")

    pool = board[board["rank"] <= TOP_N]
    if int(story.player["rank"]) > TOP_N:
        pool = pd.concat([pool, story.player.to_frame().T])
    chips = player_stats(story.player, pool, 3)
    cards = [tuple(c) for c in spec["cards"]] if spec and spec.get("cards") else \
        [(STAT_LABELS.get(c, c), format_stat(c, story.player.get(c))) for c in chips]

    if spec and spec.get("script"):
        script = _prepared_script(spec)
    else:
        script = generate_reel_script(story, board, chips, season, week, path) if write_script else None
    reel = Reel(story, board, cards, script, season, week, featured_movement=spec is None)
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
        "stat_chips": [{"label": l, "value": v} for l, v in cards],
        "captions": (script or {}).get("captions", []),
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
