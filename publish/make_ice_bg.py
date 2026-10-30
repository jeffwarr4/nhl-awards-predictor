"""Generate an original 1080×1350 ice background: scratched ice, an embossed honeycomb,
and rink markings under the surface (red center line, blue lines, center face-off circle).

The header band (STINGER wordmark, bee mascots, HART TROPHY WATCH, rule line) is taken
from config.HEADER_SOURCE and feathered into the new ice below the rule. Deterministic
for a given seed. After regenerating, rerun publish.make_bubble_bg (it builds on this image).

Run:
    python -m publish.make_ice_bg                       # → config.BOARD_BACKGROUND (Hart_Ice_bg.png)
    python -m publish.make_ice_bg <out.png> [--seed 11] # elsewhere, e.g. to try a seed
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1350
HEADER_CUT = 234          # header band kept from the source image (rule line sits at ~228)
FEATHER = 70              # px blended below the cut

# Matched to Hart_Trophy_bg.png: body ≈ (198, 209, 226), edges ≈ (156, 174, 200)
ICE_BASE = np.array([204, 214, 230], dtype=float)
RED = (196, 32, 44)
BLUE = (26, 74, 162)

# Rink layout (portrait: lines run across the width)
CENTER_Y = 800            # red center line + center face-off circle
BLUE_LINES_Y = (548, 1052)
CIRCLE_R = 168


def smooth_noise(rng, scale: int, amp: float) -> np.ndarray:
    """Low-frequency cloudiness: coarse random grid upscaled and blurred."""
    small = Image.fromarray((rng.random((H // scale + 2, W // scale + 2)) * 255).astype(np.uint8))
    big = small.resize((W + 2 * scale, H + 2 * scale), Image.BICUBIC).filter(ImageFilter.GaussianBlur(scale / 2))
    a = np.asarray(big, dtype=float)[scale:scale + H, scale:scale + W] / 255.0 - 0.5
    return a * amp


def ice_base(rng) -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W]
    # cool vignette: brighter centre, bluer edges
    d = np.hypot((xx - W / 2) / (W * 0.62), (yy - H * 0.55) / (H * 0.72))
    light = 12 * (1 - np.clip(d, 0, 1.3)) - 34 * np.clip(d - 0.75, 0, 1) ** 1.4
    cloud = smooth_noise(rng, 120, 26) + smooth_noise(rng, 40, 12)
    grain = rng.normal(0, 3.0, (H, W))
    img = ICE_BASE[None, None, :] + (light + cloud + grain)[..., None]
    img[..., 0] -= 8 * np.clip(d - 0.6, 0, 1)   # edges lean blue
    img[..., 1] -= 4 * np.clip(d - 0.6, 0, 1)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def rink_markings() -> Image.Image:
    """Paint under the ice: drawn crisp, then softened and faded by the caller."""
    m = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(m)
    # blue lines (wider) and red center line
    for y in BLUE_LINES_Y:
        d.rectangle((0, y - 13, W, y + 13), fill=(*BLUE, 255))
    d.rectangle((0, CENTER_Y - 7, W, CENTER_Y + 7), fill=(*RED, 255))
    # center face-off circle, blue in the NHL; dot and hash marks
    cx, cy, r = W // 2, CENTER_Y, CIRCLE_R
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*BLUE, 255), width=6)
    d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=(*BLUE, 255))
    return m


def faceoff_spot(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int):
    """End-zone style face-off circle: red ring, red dot, hash marks on both sides."""
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*RED, 255), width=5)
    d.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(*RED, 255))
    for side in (-1, 1):
        x = cx + side * r
        for dy in (-22, 22):
            d.line((x, cy + dy, x + side * 26, cy + dy), fill=(*RED, 255), width=5)


def scratches(rng) -> Image.Image:
    """Skate marks: many short hairlines plus a few long carved arcs; white and grey."""
    s = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    for _ in range(1400):
        x, y = rng.uniform(-50, W + 50), rng.uniform(0, H)
        ang = rng.normal(rng.choice([0.35, -0.5, 1.2]), 0.35)
        length = rng.gamma(2.0, 28)
        x2, y2 = x + math.cos(ang) * length, y + math.sin(ang) * length
        shade = (255, 255, 255, int(rng.uniform(25, 80))) if rng.random() < 0.7 else (150, 165, 185, int(rng.uniform(20, 55)))
        d.line((x, y, x2, y2), fill=shade, width=1)
    for _ in range(26):                      # long carved arcs (stops and turns)
        cx, cy = rng.uniform(0, W), rng.uniform(HEADER_CUT, H + 200)
        rr = rng.uniform(250, 900)
        a0 = rng.uniform(0, 360); span = rng.uniform(12, 40)
        d.arc((cx - rr, cy - rr, cx + rr, cy + rr), a0, a0 + span,
              fill=(255, 255, 255, int(rng.uniform(40, 90))), width=int(rng.integers(1, 3)))
    return s.filter(ImageFilter.GaussianBlur(0.4))


def snow(rng) -> Image.Image:
    """Snowy scuffs where skates stopped: blurred speckle patches."""
    layer = np.zeros((H, W), dtype=float)
    for _ in range(14):
        cx, cy = rng.uniform(0, W), rng.uniform(HEADER_CUT, H)
        rx, ry = rng.uniform(60, 180), rng.uniform(20, 60)
        yy, xx = np.ogrid[0:H, 0:W]
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        layer += blob * rng.uniform(0.4, 1.0)
    speck = rng.random((H, W)) ** 6
    alpha = np.clip(layer * speck * 420, 0, 140).astype(np.uint8)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    img.putalpha(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.8)))
    return img


def honeycomb(rng, radius: int = 96) -> Image.Image:
    """Embossed hexagons (pointy-top), lit from the top-left; stronger toward the centre."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    hw = math.sqrt(3) * radius          # cell width
    vstep = 1.5 * radius
    light = np.array([-0.6, -0.8])       # from upper left
    rows = int(H / vstep) + 3
    cols = int(W / hw) + 3
    for row in range(-1, rows):
        for col in range(-1, cols):
            cx = col * hw + (hw / 2 if row % 2 else 0)
            cy = row * vstep + 20
            pts = [(cx + (radius - 5) * math.cos(math.radians(60 * k - 30)),
                    cy + (radius - 5) * math.sin(math.radians(60 * k - 30))) for k in range(6)]
            # cell face: slight random tone so the comb reads as tiles
            tone = int(rng.uniform(-14, 14))
            d.polygon(pts, fill=(255, 255, 255, 22 + tone) if tone > 0 else (120, 140, 170, 16 - tone))
            # bevel: edges facing the light are bright, the others dark
            for k in range(6):
                (x1, y1), (x2, y2) = pts[k], pts[(k + 1) % 6]
                nx, ny = (y2 - y1), -(x2 - x1)                # outward normal
                n = math.hypot(nx, ny); nx, ny = nx / n, ny / n
                f = -(nx * light[0] + ny * light[1])
                col_ = (255, 255, 255, int(110 * f)) if f > 0 else (60, 80, 110, int(-90 * f))
                d.line((x1, y1, x2, y2), fill=col_, width=5)
    layer = layer.filter(ImageFilter.GaussianBlur(2.2))
    # fade toward the edges and the bottom, like a spotlight on centre ice
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot((xx - W / 2) / (W * 0.62), (yy - H * 0.55) / (H * 0.62))
    fade = np.clip(1.15 - dist, 0.25, 1.0)
    a = np.asarray(layer.getchannel("A"), dtype=float) * fade
    layer.putalpha(Image.fromarray(a.astype(np.uint8)))
    return layer


def build(header_from: Path | None, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = ice_base(rng)

    # markings under the ice: soft, faded, scuffed through by the scratches above them
    marks = rink_markings()
    d = ImageDraw.Draw(marks)
    for cy in (372, H - 138):                 # end-zone circles beyond each blue line
        faceoff_spot(d, 185, cy, 112)
        faceoff_spot(d, W - 185, cy, 112)
    marks = marks.filter(ImageFilter.GaussianBlur(1.6))
    wear = np.clip(0.62 + smooth_noise(rng, 30, 0.5) + rng.normal(0, 0.05, (H, W)), 0.3, 0.8)
    marks.putalpha(Image.fromarray((np.asarray(marks.getchannel("A"), dtype=float) * wear).astype(np.uint8)))
    img = Image.alpha_composite(img, marks)

    img = Image.alpha_composite(img, honeycomb(rng))
    img = Image.alpha_composite(img, snow(rng))
    img = Image.alpha_composite(img, scratches(rng))

    if header_from and header_from.exists():
        src = Image.open(header_from).convert("RGBA").resize((W, H))
        mask = np.zeros((H, W), dtype=float)
        mask[:HEADER_CUT] = 1.0
        ramp = np.linspace(1, 0, FEATHER)
        mask[HEADER_CUT:HEADER_CUT + FEATHER] = ramp[:, None]
        img = Image.composite(src, img, Image.fromarray((mask * 255).astype(np.uint8)))
    return img.convert("RGB")


def main() -> None:
    from config import BOARD_BACKGROUND, HEADER_SOURCE
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path, nargs="?", default=BOARD_BACKGROUND)
    p.add_argument("--header-from", type=Path, default=HEADER_SOURCE)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    build(a.header_from, a.seed).save(a.out)
    print("ok ->", a.out)


if __name__ == "__main__":
    main()
