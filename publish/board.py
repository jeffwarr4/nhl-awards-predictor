"""
Weekly Hart board: the scored current-season frame from Models/nhl_predict_current.py
→ predictions/{season}/ CSVs, movement, headshot candidates, then chips, graphics, caption.

Port of nfl-awards-predictor/src/Predictions/predict_current.py (board half).

Outputs (predictions/{season}/)
  top10_hart_latest.csv    ← top TOP_N, BOARD_COLS (ends with chips, why_blurb)
  board_hart_latest.csv    ← BOARD_DEPTH deep, same columns; the website's NHL page reads it from GitHub
  {stem}_{ts}.csv          ← archives (ARCHIVES_TO_KEEP most recent of each)
  .prev_board_hart.csv     ← last week's 30-deep board; drives rank_delta / movement
  chip_data/, graphics/    ← see publish/generate_chips.py, render_graphics.py, generate_captions.py
predictions/nhl_candidates.csv ← everyone on the board, for stinger-assets' headshot sync
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import (
    ARCHIVES_TO_KEEP, BOARD_DEPTH, BUBBLE_RANKS, CANDIDATES_CSV, PAGES_DIR, PREDICTIONS, TOP_N,
)

# Column order of top10_hart_latest.csv / board_hart_latest.csv (a contract: the
# website reads board_hart_latest.csv by these names). chips (top 3, pipe-separated) then why_blurb end it.
BOARD_COLS = [
    "player_key", "rank", "rank_delta", "movement", "player_name", "team", "position",
    "model_score",
    "games", "team_games", "goals", "assists", "points", "ppg", "pp_goals", "shots", "plus_minus", "atoi",
    "goalie_wins", "save_pct", "gaa",
    "team_record", "team_win_pct", "team_points",
    "season", "week", "stats_through", "updated_utc",
    "chips", "why_blurb",
]


# ------------------------
# Keys
# ------------------------
def headshot_key(name: str, team: str) -> str:
    """firstname_lastname_team slug for stinger-assets headshots.
    Must match safe_filename(f"{name} {team}") in stinger-assets/sync_espn_headshots.py:
    accents stripped, apostrophes and periods dropped, hyphens split.
    e.g. 'David Pastrňák', 'BOS' → 'david_pastrnak_bos'; 'Marc-André Fleury', 'MIN' → 'marc_andre_fleury_min'
    """
    n = unicodedata.normalize("NFKD", f"{name} {team}")
    n = "".join(ch for ch in n if not unicodedata.combining(ch)).lower()
    n = n.replace("'", "").replace(".", "").replace("-", " ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", n)).strip("_")


def season_week(first_game: date, through: date) -> int:
    """Week N of the season, counted in 7-day blocks from opening night."""
    return (through - first_game).days // 7 + 1


# ------------------------
# Board
# ------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_board(scored: pd.DataFrame, season: int, week: int, through: date) -> pd.DataFrame:
    """BOARD_DEPTH rows from the ranked frame, display columns only (actual stats, not pace)."""
    df = scored.sort_values("Hart_Rank").head(BOARD_DEPTH).reset_index(drop=True)
    goalie = df["Pos"].eq("G")
    w, l, ol = (_num(df[c]).fillna(0).astype(int) for c in ("W_team", "L_team", "OL_team"))

    board = pd.DataFrame({
        "rank": df["Hart_Rank"].astype(int),
        "player_name": df["Player"],
        "team": df["Team"],
        "position": df["Pos"],
        "model_score": _num(df["Hart_FinalScore"]).round(6),
        "games": _num(df["GP"]).astype(int),
        "team_games": _num(df["team_gp"]).astype(int),
        "goals": _num(df["G"]).fillna(0).astype(int),
        "assists": _num(df["A"]).fillna(0).astype(int),
        "points": _num(df["PTS"]).fillna(0).astype(int),
        "ppg": _num(df["PTS_per_GP"]).round(2),
        "pp_goals": _num(df["PPG"]).where(~goalie),
        "shots": _num(df["SOG"]).where(~goalie),
        "plus_minus": _num(df["+/-"]).where(~goalie),
        "atoi": df["ATOI"].where(~goalie),
        "goalie_wins": _num(df["W"]).where(goalie),
        "save_pct": _num(df["SV%"]).where(goalie),
        "gaa": _num(df["GAA"]).where(goalie),
        "team_record": [f"{a}-{b}-{c}" for a, b, c in zip(w, l, ol)],
        "team_win_pct": _num(df["team_win_pct"]).round(3),
        "team_points": _num(df["PTS_team"]).astype("Int64"),
    })
    for c in ("pp_goals", "shots", "plus_minus", "goalie_wins"):
        board[c] = board[c].astype("Int64")

    keys = [headshot_key(n, t) for n, t in zip(board["player_name"], board["team"])]
    # Two players with one name on one team (VAN's two Elias Petterssons): the lower-ranked
    # one gets a position suffix so movement and chips stay per-player.
    seen: set[str] = set()
    for i, (k, pos) in enumerate(zip(keys, board["position"])):
        if k in seen:
            keys[i] = f"{k}_{str(pos).lower()}"
        seen.add(keys[i])
    board["player_key"] = keys
    board["season"] = season
    board["week"] = week
    board["stats_through"] = through.isoformat()
    board["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    board["chips"] = ""
    board["why_blurb"] = ""
    return board


def legacy_prev_ranks(season: int, week: int, first_game: date) -> dict[str, int]:
    """Before the first .prev_board snapshot exists: last week's full board from
    pages/nhl_hart_predictions_v2_latest.csv (the Google Sheet feed), if it is this
    season and from an earlier week. Otherwise {} (everyone shows NEW)."""
    path = PAGES_DIR / "nhl_hart_predictions_v2_latest.csv"
    if not path.exists():
        return {}
    p = pd.read_csv(path, encoding="utf-8-sig")
    if p.empty or int(p["Season"].iloc[0]) != season:
        return {}
    run_day = pd.to_datetime(p["model_run_ts"].iloc[0]).date()
    # its stats ran through the day before it ran
    if season_week(first_game, run_day - timedelta(days=1)) >= week:
        return {}
    p = p.sort_values("Hart_Rank").head(BOARD_DEPTH)
    return {headshot_key(n, t): int(r) for n, t, r in zip(p["Player"], p["Team"], p["Hart_Rank"])}


def add_movement(board: pd.DataFrame, prev_path: Path, legacy: dict[str, int]) -> pd.DataFrame:
    """rank_delta: +up / -down vs last week ("" if NEW); movement: ↑3 / ↓2 / – / NEW."""
    board = board.copy()
    if prev_path.exists():
        p = pd.read_csv(prev_path)
        prev = dict(zip(p["player_key"], p["rank"].astype(int)))
    else:
        prev = legacy
    deltas, moves = [], []
    for key, rank in zip(board["player_key"], board["rank"]):
        if key not in prev:
            deltas.append(""); moves.append("NEW")
        else:
            d = prev[key] - int(rank)
            deltas.append(d)
            moves.append(f"↑{d}" if d > 0 else (f"↓{abs(d)}" if d < 0 else "–"))
    board["rank_delta"] = deltas
    board["movement"] = moves
    return board


def roll_prev_snapshot(latest_path: Path, prev_path: Path, week: int) -> bool:
    """The movement baseline is the last board from an EARLIER week, so a same-week
    rerun (manual dispatch, a fix) keeps last week's arrows. True if rolled."""
    if not latest_path.exists():
        return False
    latest = pd.read_csv(latest_path)
    if len(latest) and int(latest["week"].iloc[0]) < week:
        latest[["player_key", "rank", "week"]].to_csv(prev_path, index=False)
        return True
    return False


def write_candidates(board: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sport": "NHL", "player_name": board["player_name"],
                  "team": board["team"], "player_key": board["player_key"]}) \
      .drop_duplicates("player_key").to_csv(path, index=False)
    print(f"Headshot candidates -> {path} ({len(board)} players)")


def write_boards(board: pd.DataFrame, outdir: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    board = board[BOARD_COLS]
    for df, stem in ((board.head(TOP_N), "top10_hart"), (board, "board_hart")):
        df.to_csv(outdir / f"{stem}_latest.csv", index=False)
        df.to_csv(outdir / f"{stem}_{ts}.csv", index=False)
        for old in sorted(outdir.glob(f"{stem}_2*.csv"))[:-ARCHIVES_TO_KEEP]:
            old.unlink()


# ------------------------
# Main
# ------------------------
def publish_board(scored: pd.DataFrame, season: int, first_game: date, through: date,
                  outdir: Path | None = None, publish: bool = True) -> pd.DataFrame:
    """Board CSVs + movement + candidates; then chips, graphics and caption when publish."""
    default_out = PREDICTIONS / str(season)
    outdir = Path(outdir or default_out)
    outdir.mkdir(parents=True, exist_ok=True)
    # A dry run into another folder must not overwrite the live candidates feed.
    candidates = CANDIDATES_CSV if outdir.resolve() == default_out.resolve() else outdir / CANDIDATES_CSV.name

    week = season_week(first_game, through)
    board = build_board(scored, season, week, through)
    if len(board) < BUBBLE_RANKS[1]:
        # Fail loudly — never replace the published board with one too short for the bubble slide.
        raise SystemExit(f"ERROR: only {len(board)} eligible players for {season} week {week}; "
                         f"board not written.")

    latest, prev = outdir / "board_hart_latest.csv", outdir / ".prev_board_hart.csv"
    roll_prev_snapshot(latest, prev, week)
    board = add_movement(board, prev, {} if prev.exists() else legacy_prev_ranks(season, week, first_game))
    write_boards(board, outdir)

    print(f"\n=== {season - 1}–{str(season)[-2:]} Hart — week {week}, stats through {through} ===")
    show = ["rank", "movement", "player_name", "team", "position", "games", "team_games",
            "goals", "assists", "points", "team_record"]
    print(board.head(TOP_N)[show].to_string(index=False))
    print(f"Saved -> {latest} ({len(board)} deep)")

    if publish:
        write_candidates(board, candidates)
        publish_outputs(board, outdir, season, week, through)
    return board


def publish_outputs(board: pd.DataFrame, outdir: Path, season: int, week: int, through: date) -> None:
    """Chips + bubble Whys (needs ANTHROPIC_API_KEY) → board CSVs get chips + why_blurb →
    board and bubble graphics → carousel caption (needs key)."""
    from publish.generate_chips import generate_chips, load_chips
    from publish.render_graphics import render_board, render_bubble_graphic
    from publish.generate_captions import generate_caption_file

    gfx = outdir / "graphics"
    chip_df = generate_chips(board, outdir)
    if chip_df.empty:
        # No key or the API failed: keep this week's earlier chips rather than blanking them.
        chip_df = load_chips(outdir, week)
    if not chip_df.empty:
        whys = dict(zip(chip_df["player_key"], chip_df["Why"]))
        chips = dict(zip(chip_df["player_key"], chip_df["Chip"].fillna("")))
        board = board.assign(chips=board["player_key"].map(chips).fillna(""),
                             why_blurb=board["player_key"].map(whys).fillna(""))
        for df, name in ((board.head(TOP_N), "top10_hart_latest.csv"), (board, "board_hart_latest.csv")):
            df[BOARD_COLS].to_csv(outdir / name, index=False)

    render_board(board.head(TOP_N), chip_df, gfx, season, week, through)
    # The bubble slide is secondary: a failure there must not cost the main board's caption.
    try:
        render_bubble_graphic(board, chip_df, gfx, season, week, through)
    except Exception as e:
        print(f"  [WARN] Bubble graphic failed: {e}")
    generate_caption_file(board, chip_df, gfx, season, week)
