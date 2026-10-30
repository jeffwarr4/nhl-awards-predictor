"""Build the Hart 'On the Bubble' background for ranks 11-15 from the board background
(config.BOARD_BACKGROUND, Hart_Ice_bg.png).

Port of nfl-awards-predictor/src/publish/make_bubble_bg.py, restyled for the light
ice board: navy wells and plates instead of dark turf, gold rings, ice-blue micro-bubbles.

Outputs (config paths):
  Hart_Bubble_bg.png        background: header + 'ON THE BUBBLE' pill + 5 empty bubbles + plates
  Hart_Bubble_overlay.png   transparent layer: gold rings + gloss, composited OVER headshots
  assets/bubble_slots.json  centers, diameters, label boxes, rank/logo badge anchors, footer

Run:
    python -m publish.make_bubble_bg
    python -m publish.make_bubble_bg --preview <png>   # also render the latest board's ranks 11-15
"""
import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import BOARD_BACKGROUND, BUBBLE_BACKGROUND, BUBBLE_OVERLAY, BUBBLE_SLOTS

W, H = 1080, 1350
GOLD, BLACK = (255, 210, 0), (0, 0, 0)
NAVY = (14, 28, 48)
ICE = (205, 225, 245)
HEADER_BOTTOM = 312   # header band + rule + pill + subtitle

# ── Slots: #11 biggest and highest ("closest to breaking in"), shrinking and
#    sinking toward #15. Zig-zag so labels never sit under another bubble.
SLOTS = [
    {"rank": 11, "cx": 300, "cy": 495,  "d": 330},
    {"rank": 12, "cx": 790, "cy": 550,  "d": 290},
    {"rank": 13, "cx": 545, "cy": 952,  "d": 255},
    {"rank": 14, "cx": 215, "cy": 1095, "d": 225},
    {"rank": 15, "cx": 870, "cy": 1105, "d": 200},
]
LABEL_H = 112  # name + meta + up to two Why lines
for s in SLOTS:
    r = s["d"] // 2
    lw = max(s["d"] + 40, 330)
    s["label_box"] = [s["cx"] - lw // 2, s["cy"] + r + 8, s["cx"] + lw // 2, s["cy"] + r + 8 + LABEL_H]
    a = math.radians(135)  # rank badge at ~7:30 on the ring
    s["rank_badge"] = [int(s["cx"] + r * math.cos(a)), int(s["cy"] + r * math.sin(a))]
    a = math.radians(-45)  # team logo badge at ~1:30
    s["logo_badge"] = [int(s["cx"] + r * math.cos(a)), int(s["cy"] + r * math.sin(a))]
FOOTER = {"cx": 540, "cy": 1272}
FOOTER_BOX = [FOOTER["cx"] - 150, FOOTER["cy"] - 28, FOOTER["cx"] + 150, FOOTER["cy"] + 50]


def check_layout():
    """No bubble/label/footer overlaps; everything on canvas below the header."""
    def boxes(s):
        r = s["d"] // 2
        return [(s["cx"] - r, s["cy"] - r, s["cx"] + r, s["cy"] + r), tuple(s["label_box"])]
    allb = [(s["rank"], b) for s in SLOTS for b in boxes(s)] + [("footer", tuple(FOOTER_BOX))]
    for i, (ra, a) in enumerate(allb):
        assert a[0] >= 0 and a[1] >= HEADER_BOTTOM and a[2] <= W and a[3] <= H, (ra, a)
        for rb, b in allb[i + 1:]:
            if ra != rb and not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
                raise SystemExit(f"overlap #{ra} {a} vs #{rb} {b}")


def pill(img, text, cy, size=30):
    from publish.render_graphics import _font
    d = ImageDraw.Draw(img); f = _font("agency_bold", size)
    bb = d.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bw, bh = tw + 44, 42; bx, by = W // 2 - bw // 2, cy - bh // 2
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=10, fill=(*GOLD, 245), outline=NAVY, width=2)
    d.text((bx + (bw - tw) // 2 - bb[0], by + (bh - th) // 2 - bb[1]), text, font=f, fill=BLACK)


def build_background() -> Image.Image:
    from publish.render_graphics import _font
    bg = Image.open(BOARD_BACKGROUND).convert("RGBA")
    pill(bg, "ON THE BUBBLE", 262)
    d = ImageDraw.Draw(bg)
    sub = "RANKS 11–15 · KNOCKING ON THE DOOR"
    f = _font("fg_demi_cond", 24); bb = d.textbbox((0, 0), sub, font=f)
    d.text((W // 2 - (bb[2] - bb[0]) // 2, 288), sub, font=f, fill=NAVY)

    # decorative micro-bubbles, kept clear of slots, labels and the footer
    rng = np.random.default_rng(7)
    deco = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(deco)
    placed = 0
    while placed < 22:
        r = int(rng.integers(8, 26)); x = int(rng.integers(30, W - 30)); y = int(rng.integers(340, H - 30))
        clear = all(math.hypot(x - s["cx"], y - s["cy"]) > s["d"] / 2 + r + 18 for s in SLOTS)
        clear &= all(not (lb[0] - r < x < lb[2] + r and lb[1] - r < y < lb[3] + r)
                     for lb in [s["label_box"] for s in SLOTS] + [FOOTER_BOX])
        if clear:
            dd.ellipse((x - r, y - r, x + r, y + r), outline=(*NAVY, 110), width=2, fill=(255, 255, 255, 60))
            dd.arc((x - r * .6, y - r * .6, x + r * .2, y + r * .2), 200, 260, fill=(255, 255, 255, 200), width=2)
            placed += 1
    bg = Image.alpha_composite(bg, deco)

    # empty bubble wells: soft shadow + deep icy-navy interior (headshots paste here)
    for s in SLOTS:
        r = s["d"] // 2; cx, cy = s["cx"], s["cy"]
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse((cx - r + 6, cy - r + 14, cx + r + 6, cy + r + 14), fill=(10, 20, 40, 110))
        bg = Image.alpha_composite(bg, sh.filter(ImageFilter.GaussianBlur(14)))
        well = Image.new("RGBA", (W, H), (0, 0, 0, 0)); wd = ImageDraw.Draw(well)
        for i in range(r, 0, -2):  # radial: lighter centre, darker edge
            t = i / r
            wd.ellipse((cx - i, cy - i, cx + i, cy + i),
                       fill=(int(20 + 40 * (1 - t)), int(38 + 50 * (1 - t)), int(66 + 60 * (1 - t)), int(170 + 70 * t)))
        bg = Image.alpha_composite(bg, well)
        # navy label plate so white names read on the ice
        plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(plate).rounded_rectangle(s["label_box"], radius=14, fill=(*NAVY, 225))
        bg = Image.alpha_composite(bg, plate)

    plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle(FOOTER_BOX, radius=14, fill=(*NAVY, 225))
    return Image.alpha_composite(bg, plate)


def build_overlay() -> Image.Image:
    """Goes on top of the headshots: gold ring + glow, inner navy line, gloss highlight."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for s in SLOTS:
        r = s["d"] // 2; cx, cy = s["cx"], s["cy"]
        ring = max(6, s["d"] // 45)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6), outline=(*GOLD, 150), width=ring + 8)
        ov = Image.alpha_composite(ov, glow.filter(ImageFilter.GaussianBlur(6)))
        od = ImageDraw.Draw(ov)
        od.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*GOLD, 255), width=ring)
        od.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), outline=(*NAVY, 200), width=2)
        gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
        gr = int(r * 0.78)
        gd.arc((cx - gr, cy - gr, cx + gr, cy + gr), 195, 255, fill=(*ICE, 170), width=max(5, r // 14))
        hx, hy, hr = cx - int(r * .52), cy - int(r * .52), max(5, r // 16)
        gd.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(255, 255, 255, 200))
        ov = Image.alpha_composite(ov, gl.filter(ImageFilter.GaussianBlur(1.5)))
    return ov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, default=None,
                        help="also render the latest board's ranks 11-15 to this PNG")
    args = parser.parse_args()

    check_layout()
    build_background().convert("RGB").save(BUBBLE_BACKGROUND)
    build_overlay().save(BUBBLE_OVERLAY)
    BUBBLE_SLOTS.parent.mkdir(parents=True, exist_ok=True)
    BUBBLE_SLOTS.write_text(json.dumps({
        "canvas": [W, H], "slots": SLOTS, "footer": FOOTER,
        "notes": f"Sizes fixed by rank (#11 largest). Paste circular headshot into each slot, "
                 f"then composite '{BUBBLE_OVERLAY.name}' on top.",
    }, indent=2), encoding="utf-8")
    print("ok", [(s["rank"], s["d"]) for s in SLOTS], "->", BUBBLE_BACKGROUND.name, BUBBLE_OVERLAY.name)

    if args.preview:
        import pandas as pd
        from config import CURRENT_PRED_DIR
        from publish.render_graphics import render_bubble_graphic
        board = pd.read_csv(CURRENT_PRED_DIR / "board_hart_latest.csv")
        chip_path = CURRENT_PRED_DIR / "chip_data" / "chip_data_latest.csv"
        chips = pd.read_csv(chip_path).fillna("") if chip_path.exists() else pd.DataFrame()
        out = render_bubble_graphic(board, chips, args.preview.parent, int(board["season"].iloc[0]),
                                    int(board["week"].iloc[0]), date.fromisoformat(str(board["stats_through"].iloc[0])))
        if out:
            out.replace(args.preview)


if __name__ == "__main__":
    main()
