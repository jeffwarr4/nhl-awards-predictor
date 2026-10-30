# config.py
# ---------------------------------------------------------------
# Single source of truth for nhl-awards-predictor: seasons, paths,
# feature list and in-season scoring knobs.
#
# Seasons are Hockey-Reference *end* years: 2027 = the 2026–27 season.
# Bump SEASON_END_YEAR and TRAIN_END each season; nothing else hardcodes a year.
# ---------------------------------------------------------------

import os
from pathlib import Path

# ── Seasons ───────────────────────────────────────────────────────
SEASON_END_YEAR = 2027   # season being predicted RIGHT NOW
TRAIN_START     = 2008   # first season of training data
TRAIN_END       = 2026   # last season with a complete Hart vote

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
DATA_RAW       = ROOT_DIR / "Data" / "Raw"
DATA_PROCESSED = ROOT_DIR / "Data" / "Processed"
ARTIFACTS_DIR  = ROOT_DIR / "Models" / "Artifacts"

TRAINING_CSV    = DATA_RAW / "nhl_historical_training.csv"
MODEL_REG_PATH  = ARTIFACTS_DIR / "hart_vote_reg_v2.pkl"
MODEL_TOP5_PATH = ARTIFACTS_DIR / "hart_top5_clf_v2.pkl"
FEATURES_PATH   = ARTIFACTS_DIR / "hart_features_v2.json"
BACKTEST_CSV    = ARTIFACTS_DIR / "hart_backtest_v2.csv"

# ── Model features (shared by training and live scoring) ─────────
# team_pts_pct = team points / (2 × team games): season-length invariant, so it
# means the same thing in a 48-game 2013, an 82-game season, and in November.
NUMERIC_FEATURES = [
    "Age", "GP", "G", "A", "PTS",
    "G_per_GP", "A_per_GP", "PTS_per_GP", "SOG_per_GP",
    "PPG", "PPG_per_GP",
    "team_pts_pct",
    "is_goalie",
]
CATEGORICAL_FEATURES = ["Pos"]

# Counting features scaled to an 82-game pace before live scoring. Rates aren't.
PACE_COLS = ["GP", "G", "A", "PTS", "PPG"]

# ── Training ──────────────────────────────────────────────────────
MIN_GP_TRAIN = 20        # full-season rows only

# ── In-season scoring ─────────────────────────────────────────────
SEASON_GAMES        = 82
MIN_TEAM_GAME_SHARE = 0.60   # must have played ≥60% of his team's games

# ── Weekly board + publishing (publish/) ──────────────────────────
PREDICTIONS      = ROOT_DIR / "predictions"
CURRENT_PRED_DIR = PREDICTIONS / str(SEASON_END_YEAR)
CANDIDATES_CSV   = PREDICTIONS / "nhl_candidates.csv"   # pulled by stinger-assets' headshot sync
PAGES_DIR        = ROOT_DIR / "pages"                     # legacy Google Sheet feed
FONTS_DIR        = ROOT_DIR / "assets" / "fonts"

# 1080×1350 backgrounds. The bubble background, overlay and slot geometry come
# from publish/make_bubble_bg.py.
BOARD_BACKGROUND  = ROOT_DIR / "Hart_Trophy_bg.png"
BUBBLE_BACKGROUND = ROOT_DIR / "Hart_Bubble_bg.png"
BUBBLE_OVERLAY    = ROOT_DIR / "Hart_Bubble_overlay.png"
BUBBLE_SLOTS      = ROOT_DIR / "assets" / "bubble_slots.json"

# Sibling repo with headshots/logos. CI checks it out and sets the env var.
STINGER_ASSETS_DIR = Path(os.environ.get("STINGER_ASSETS_DIR", str(ROOT_DIR.parent / "stinger-assets")))

TOP_N            = 10
BOARD_DEPTH      = 30        # saved board depth; everyone on it is a headshot candidate
BUBBLE_RANKS     = (11, 15)  # inclusive; the "On the Bubble" slide
ARCHIVES_TO_KEEP = 4
