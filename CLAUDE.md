# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHL Awards Predictor — a Python ML pipeline that scrapes Hockey-Reference.com, trains models on historical Hart Trophy voting data, and produces weekly predictions for the current season. Predictions are auto-committed to `main` on a GitHub Actions schedule. The sibling `nfl-awards-predictor` repo is the reference pipeline this one is being brought in line with.

## Environment

- Venv lives outside the repo: `C:\DevVenvs2\nhl-awards-venv\Scripts\python.exe` (Python 3.12). The in-repo `.venv` points at a Python install that no longer exists.
- `pip install -r requirements.txt`. scikit-learn is pinned (1.7.2) because the `.pkl` models are pickled pipelines.

## Common Commands

```powershell
# Rebuild historical training data (TRAIN_START..TRAIN_END; ~76 throttled Hockey-Reference fetches, ~15 min)
python Models/nhl_historical.py

# Leave-one-season-out backtest + retrain on all seasons (prints a winner-rank table per season)
python Models/nhl_train_v2.py
python Models/nhl_train_v2.py --no-backtest

# Score the current season + weekly board, candidates, chips, graphics, caption
python Models/nhl_predict_current.py
python Models/nhl_predict_current.py --outdir <scratch> --no-publish   # board CSVs only, elsewhere

# Re-run one publish step from the latest board (run from the repo root)
python -m publish.render_graphics
python -m publish.generate_chips
python -m publish.generate_captions
python -m publish.make_ice_bg                        # regenerate Hart_Ice_bg.png (then make_bubble_bg)
python -m publish.make_bubble_bg [--preview <png>]   # regenerate bubble background/overlay/slots
```

There are no tests or linters. `ANTHROPIC_API_KEY` enables chips, Whys and the caption; without it they are skipped with a warning. `STINGER_ASSETS_DIR` overrides the default `../stinger-assets` for headshots/logos.

## Pipeline Architecture

```
nhl_historical.py  →  Data/Raw/nhl_historical_training.csv
       ↓
nhl_train_v2.py    →  Models/Artifacts/hart_vote_reg_v2.pkl   (GradientBoostingRegressor on vote_share)
                   →  Models/Artifacts/hart_top5_clf_v2.pkl   (LogisticRegression on is_top5)
                   →  Models/Artifacts/hart_features_v2.json  (features, trained seasons, backtest summary)
                   →  Models/Artifacts/hart_backtest_v2.csv   (per-season winner rank)
       ↓
nhl_predict_current.py  →  Data/Processed/nhl_hart_predictions_v2_{season}.csv
                        →  Data/Processed/nhl_hart_top5_flat_{season}.csv
                        →  Data/Processed/nhl_hart_6to10_flat_{season}.csv
```

**`config.py` (repo root) is the single source of truth** for seasons (`SEASON_END_YEAR`, `TRAIN_START`, `TRAIN_END`), paths, the feature list and the in-season knobs. Seasons are Hockey-Reference *end* years (2027 = 2026–27). Bump `SEASON_END_YEAR` and `TRAIN_END` each season; nothing else hardcodes a year.

**`Models/hart_common.py` holds all shared cleaning and features**: team mapping, `collapse_multi_team`, `parse_standings`, `add_team_context`, `engineer_features`, `project_to_pace`, and `hart_scores` (the blend). Both `nhl_historical.py` and `nhl_predict_current.py` build player rows through it. Add features there, never in one script only.

## Key Design Decisions

**Dual-model scoring**: `hart_scores` blends 0.50 × min-max-scaled `Hart_VoteShare_Pred` (GBR) + 0.40 × `Hart_Top5_Prob` (LogReg) + 0.10 × `Hart_Win_Prob` (softmax over the top 25). In the backtest the blend beats either model alone on mean winner rank (the classifier alone hits #1 more often but misses badly more often).

**Evaluation is leave-one-season-out; the shipped models train on every season** 2008–`TRAIN_END`. (Before Oct 2026 the last 3 seasons were held out of the shipped model entirely.) Seasons are never split randomly. Backtest after the Oct 2026 rebuild: winner #1 in 37% of 19 seasons, top 3 in 63%, mean winner rank 3.3, 2.8 of the actual top 5 in the model's top 5. Goalie winners are the weak spot (Hellebuyck 2025: #14) — there are no goalie-quality features.

**Team strength = `team_pts_pct`** (team points / (2 × team games)), season-length invariant. `team_win_pct` (W/(W+L)) is still computed for the published CSV but is not a model feature. Until Oct 2026 the model never saw team success: Hockey-Reference marks playoff teams `*` and the name map didn't strip it (every playoff team lost its record), and training read goalie W/L instead of the team's.

**Live scoring** (`nhl_predict_current.py`), in order:
1. One row per player: the skaters page also lists goalies, so goalie rows come from the goalies page only; a traded player keeps his season-total row (`2TM`/`TOT`) under the team of his last per-team row (his current club).
2. Eligibility: GP ≥ 60% of his team's games (`MIN_TEAM_GAME_SHARE`). The old `GP >= 20` gate emptied the board until late November.
3. Counting features (`PACE_COLS`: GP, G, A, PTS, PPG) are projected to an 82-game pace before scoring; displayed stats stay actual. 2025–26 replay: without pace, Kucherov (eventual winner) sat 25th in mid-December; with pace + the team fix he was #1 every week from Feb 4.
4. Exits non-zero before writing anything if standings can't be parsed or nobody is eligible.

**Training rows** use `GP >= MIN_GP_TRAIN` (20). Same-name players in one season (two Sebastian Ahos in 2019; two Elias Petterssons on VAN in 2027) keep Hart votes only on the ballot-team player.

## Weekly board (`publish/`, port of nfl-awards-predictor)

`nhl_predict_current.py` hands the scored frame to `publish/board.py`, which writes `predictions/{season}/`:
- `top10_hart_latest.csv` and `board_hart_latest.csv` (30 deep, `BOARD_DEPTH`): same columns, `BOARD_COLS` in `board.py` (a contract: the website reads `board_hart_latest.csv` from GitHub, like NFL's). They end with `chips` (top 3, pipe-separated) and then `why_blurb` (top-3 and bubble Whys); both are blank without a key. The website's NHL page and home Trending read `board_hart_latest.csv` from GitHub; the Google Sheet is no longer used by the site. 4 rolling timestamped archives each.
- **No win probability** on the board or graphics (the softmax sits near 5% for everyone in-season). `Hart_Win_Prob` stays in `pages/` only, for the Sheet.
- `week` counts 7-day blocks from opening night and `stats_through` is the last date with completed games, both from the Hockey-Reference schedule page.
- `player_key = first_last_team`, lowercase, accents stripped, apostrophes/periods dropped, hyphens split. It must equal `safe_filename(f"{name} {team}")` in stinger-assets. Teams are Hockey-Reference codes (`VEG`, `TBL`, `LAK`…); a traded player's key uses his current team. Two same-name players on one team (VAN's two Elias Petterssons): the lower-ranked one gets a `_{pos}` suffix.
- **Movement**: `.prev_board_hart.csv` (hidden, 30 rows) is last week's board. It is rolled only when `week` advances, so a same-week rerun keeps last week's arrows; deleting it resets every row to NEW. Until it first exists, `pages/nhl_hart_predictions_v2_latest.csv` is used, and only if it is this season and from an earlier week.
- **Guards**: nothing is written if fewer than 15 players are eligible (the bubble slide needs ranks 11–15). If the chips fail or the key is missing on a same-week rerun, this week's saved `chip_data_latest.csv` is reused instead of blanking the Whys.
- `predictions/nhl_candidates.csv` (`sport, player_name, team, player_key`): everyone on the 30-deep board, so headshots exist before a player reaches the graphic.
- **Chips** (`generate_chips.py`, `claude-haiku-5-5`, effort `low`): Chip + Why for the top 3; a Why of at most 75 characters (enforced in code) for ranks 11–15. Each bubble player is compared with the lowest-ranked top-10 player in the same role (forward C/LW/RW/F/W, defenseman, goalie), falling back to #10. The top-3 prompt sees one player's stats only, so it forbids "leads"/"ranks first" claims: the first live run said MacKinnon and Bouchard "lead" while McDavid was #1 with 36 points.
- **Graphics** (`render_graphics.py`): the main board on `Hart_Ice_bg.png` (light ice, so every text block sits on a navy plate). `make_ice_bg.py` generates it (deterministic, seed 11): original ice, scratches, an embossed honeycomb, and rink markings under the surface (red center line, blue lines, center and end-zone face-off circles). Only the header band (wordmark, bees, title, rule) comes from `Hart_Trophy_bg.png` (`HEADER_SOURCE`), feathered in below the rule. That image's body has stock-site watermarks, which is why it is no longer used as the background. After regenerating the ice, rerun `make_bubble_bg.py`, which builds on it. If a top-3 headshot is missing, a large team logo takes its place. The "On the Bubble" slide uses `Hart_Bubble_bg.png` + `Hart_Bubble_overlay.png` + `assets/bubble_slots.json`, all generated by `make_bubble_bg.py`. Fonts are OFL copies in `assets/fonts/`. `graphics/caption_hart.txt` covers the two-slide carousel.

**Weekly and final boards (`scripts/snapshot_board.py`, the same file in all four awards repos).** The
workflow's "Keep this week's board" step, just before the commit, copies `board_hart_latest.csv` to `board_hart_wkNN.csv` (NN = `week`), which is never
pruned (`ARCHIVES_TO_KEEP` only prunes the timestamped run files). A same-week rerun overwrites that copy. After the
season's last regular-season board, run `python scripts/snapshot_board.py --final` once and commit: it writes
`board_hart_final.csv`, which the website's season archive pages read. Don't edit or delete the weekly or final files; they're the
season history.

## Weekly reel (`publish/render_reel.py`, port of nfl-awards-predictor)

One story per week as a silent 1080×1920 MP4 (20 s + 1 s hold) plus a narration script Jeff voices over in Instagram
(he adds the Captions sticker there; nothing is burned in). The reel posts first, the afternoon the board completes;
the static board posts the next day (order reversed 2026-11-24). `python -m publish.render_reel [--outdir DIR]
[--frames-only] [--no-script] [--story FILE]`; tests: `python -m pytest tests/`.

Reworked Nov 2026 for retention, with the NBA and NFL reels (the first NHL reel held viewers ~6 of 27 s). Four scenes,
`REEL_SCENES` hook 0–3 / proof 3–13 / context 13–16 / ask 16–20. Frame 0 already shows the hook, headshot and last
week's rank (no fade-ins), the rank flips by 0.8 s, and the reel ends on a comment question, not a spoken URL.
- `reel_story.pick_story` (pure) takes the first match: new No. 1 (or "back on top" if he held it on an earlier
  `board_hart_wkNN.csv`), biggest climber into the top 5, 11+ into the top 10, biggest drop by a former top-5
  player, else a profile of No. 1. Players are matched by `player_key`, as in `.prev_board_hart.csv`, so a player
  traded since last week reads as NEW. Hooks are short and lead with the number; a climb uses his season path when
  it's bigger than last week's move ("No. 30 → No. 3 in 3 weeks", `season_lows`). Weeks 5–7 of 2026–27 were
  backfilled from the dated boards.
- Then vs now: `comparison_row` picks his row on an earlier weekly board (the path hook's week, else the earliest
  within `REEL_LOOKBACK_WEEKS` where he was at least as far from this week's rank as last week, so the window covers
  the move the hook names). `then_now_lines` gives the script his rank, the team record (W-L-OTL) and the record
  since then (left out after a trade), his rates (`stats.rate_cols`: points/goals/assists a game, or a goalie's save %
  and GAA) and his points (or wins) since then. `MODEL_NOTE` says in plain words what the model weighs.
- `reel_script.py` writes four timed beats in Jeff's first-person voice on `REEL_MODEL` (Opus 5.5, with the API's
  refusal fallback; chips stay on Haiku), plus the on-screen `question`, the Instagram caption and 5 hashtags. v3
  (Nov 2026) makes it flow: the hook names him ("Kirill Kaprizov: No. 15 to No. 3 in one week."), the whole script uses ONE time window, the proof
  gives ONE reason after a lead-in that rotates by week (`PROOF_OPENERS`: "So what changed?", "Here's why.", "The reason?", "What happened?", offset per sport by `OPENER_OFFSET` so the three sports differ; the model may skip it when the hook makes the reason obvious, and `problems()` flags any other lead-in) in connected sentences (at most two numbers), the context is the gap to
  the milestone, and the ask is a yes/no or two-way question about it. `story_frame()` hands the script:
  - the window: the `comparison_row` week to now ("the last three weeks (Week 8 to Week 11), his last three
    games"), or "this week" with no earlier board; the hook's ranks are the window's
  - the driver: `player` if a `REEL_FLAT_STATS` rate (points/goals/assists a game, or a goalie's save % and GAA; absolute amounts; lower GAA is better) moved by at least its
    amount with his rank; else `team` if the team's points % (W-L-OTL) in the window is `REEL_TEAM_SWING` above/below its
    rate before (3+ games); else `board` if he played no games in the window (a bye) and players ahead of him
    dropped; else `unclear`. The script may only claim a cause the driver states.
  - `crossed()`: the players ahead of him at the window's start who are behind him now (or, for a drop, the
    reverse), from the window-start board
  - the milestone (`milestone()`): hold No. 1; catch No. 1 from No. 2-3; the top 3 from 4-5; the top 5 from 6-10;
    the top 10 from 11+
  `problems()` checks `REEL_WORDS` (40-50), the hook (≤ `REEL_HOOK_MAX_WORDS`, names him, no "our model", and reads like a fan saying it: "Jack Eichel jumped from 15 to 5 in one week." with a verb and bare numbers; `ROBOT_HOOK` flags a colon or "No." before a rank, and the team name before his name is left out as a tongue twister), the
  proof (≤ `REEL_PROOF_MAX_WORDS`), his full name (suffix optional, accents folded) or a `REEL_NICKNAMES` short
  name somewhere, no then/since/now without a time anchor in the same sentence (`unanchored_time`: "this week",
  "Week 8", "three weeks ago", "his last three games"...), a context that doesn't repeat the hook's ranks or name
  the No. 1 (unless it's a No. 1 story or he's No. 2-3), the last beat ending with "?", no stock phrases or "my
  model", no probability language (a save % is fine, win % never), and 5 hashtags. The caption's site line is added in code (`caption_with_cta`): whatever site line the model wrote is cut off and "Full board at stingercollectibles.com." follows as its own sentence. The prompt carries two style examples
  (Jeff's, edited to pass the checks; facts marked as examples only). One rewrite request; then `finish()` swaps in
  `fallback_question()` about the milestone ("Can Kaprizov catch McDavid?"). The result, `reel_meta.json` and the script file keep the
  frame, `first_draft_problems` and `rewritten`. `render_reel --script-only` writes just the script and meta (for
  reviewing scripts on replays); a `--story` file without its own cards now gets the then-vs-now comparison too.
  Tests: `tests/test_reel_script.py`.
- The Proof scene: with a comparison, a "WEEK 5 → WEEK 8" label and 3 then → now cards (`then_now_cards`: the team
  record unless he was traded, then the rates that moved most). Without one, 3 stat cards from `player_stats`
  (goalies get the goalie set, win % left out), the record line and the board's `why_blurb`. The background is the
  board's rink tinted navy, and the header band is a gold-bordered badge. Settings are `REEL_*` in `config.py`.
- Outputs in `predictions/{season}/reel/`: `reel_script.txt` and `reel_meta.json` are committed (the meta keeps the
  keys stinger-website reads and adds `question`); `reel_latest.mp4` is gitignored and the workflow uploads it to
  the repo's `reel-latest` release as `nhl_reel_{season}_wk{NN}.mp4`. stinger-website's social-drafts job emails it
  with the script. Both workflow steps are `continue-on-error`, after the headshot fetch. `nhl_predict_current.py`
  doesn't render the reel; run it on its own locally. Needs anthropic 2.x (`fallbacks`).
- One-off reels (a correction, a feature): a story file names the player and hook and can set labels, cards and a
  script (four beats, `question`, caption, hashtags); it writes `reel_<name>.*` beside the weekly files. Upload the
  MP4 by hand. `story_kaprizov_wk08.json` (Week 8, the first reel) uses the old five-beat script and is history.

## stinger-assets contract

Headshots: `nhl/headshots/{player_key}.png` (600×436). Logos: `nhl/logos/{team}.png`, lowercase Hockey-Reference codes, except `LAK` → `la.png` (`publish/stats.py: LOGO_CODES`). The runner is case-sensitive; Windows isn't. stinger-assets' `scripts/merge_candidates.py` must list `nhl-awards-predictor/main/predictions/nhl_candidates.csv`. Its `sync_espn_headshots.py` needs NHL team aliases (Hockey-Reference → ESPN: `LAK→LA, NJD→NJ, SJS→SJ, TBL→TB, UTA→UTAH, VEG→VGK`) for team-narrowed matching.

## Output contract (website)

The new stinger-website (Cloudflare) reads `predictions/{season}/board_hart_latest.csv` straight from GitHub (`site/src/_data/awards.json`, `csv` for the NHL entry). Its columns use these board fields by name: `team, position, games, goals, assists, points, ppg, plus_minus, team_record` (team record shown as written), `goalie_wins, save_pct, gaa` for goalie bubble rows, and `rank_delta`/`movement`, `chips`, `why_blurb`. Renaming any of them breaks the page.

`pages/` is the legacy feed for the old Squarespace site, which reads a Google Sheet whose tabs IMPORTDATA those files. Keep it working until that site is retired. `nhl_hart_predictions_v2_latest.csv` must keep exactly `PAGES_COLUMNS` (in `nhl_predict_current.py`); the Sheet derives `mvp_prob` from `Hart_Win_Prob` and `week_label` from `model_run_label`. `top5.csv` / `top6to10.csv` are wide one-row-per-run CSVs (`r1_player`, `r1_team`, …).

## Data Sources & Scraping

All data comes from Hockey-Reference.com with exponential backoff for 429/5xx. `clean_team_name` strips only markers (`*`, `+`, daggers, a leading `x-` clinch prefix). An older version stripped the letters x/y/z anywhere, which is why typo entries like "Calgar Flames" existed in old team maps.

## Model Artifacts

`.pkl` files in `Models/Artifacts/` **are tracked in git**. The weekly workflow does not retrain: retrain locally and commit the artifacts.

## Automation

`.github/workflows/nhl_predict.yml` is started Thursdays at 15:17 UTC, October–April, by the stinger-scheduler Cloudflare Worker (`stinger-website/scheduler`); its own cron (Thursdays 17:17 UTC) is a late backup. Plus manual dispatch:
1. Checks out stinger-assets as a sibling (`STINGER_ASSETS_DIR`) and runs `nhl_predict_current.py` on Python 3.12.
2. Fails if the top-10 board is empty. The script itself also exits non-zero before writing a short board.
3. Copies the Sheet CSVs to `pages/` and uploads the CSV/PNG/TXT outputs as run artifacts.
4. Commits `pages/` and `predictions/` (plus `git add -f` of `.prev_board_hart.csv`) with `[skip ci]`.

A `guard` job runs first: a scheduled start (the cron, or a dispatch with `scheduled=true`, which is how the stinger-scheduler Cloudflare Worker starts it) is skipped when a scheduled run already succeeded in the last 20 hours, so a late cron after an on-time run is a no-op. Manual runs from the Actions tab (`scheduled` left false) always run. Run names end in `(scheduled)` or `(manual)`; the guard relies on that suffix.

After the board is written, `stinger-assets/sync_espn_headshots.py fetch --strict` downloads any missing headshot for
ranks 1–15 (a player new to the slides isn't in stinger-assets until its daily sync reads the committed candidates
file, i.e. after this run), and the slides are re-rendered if it downloaded any. If one still can't be found the run
fails before committing, so a slide with a blank circle never reaches the site or the social draft: add the PNG to
stinger-assets by hand (or fix the team alias there) and rerun.

It needs the `ANTHROPIC_API_KEY` secret for chips and captions.

## Git

This repo lives in OneDrive, which has deleted tracked files before. Stage files by name (never `git add -A`) and stop if `git status` shows deletions you didn't make. Push as the `jeffwarr4` GitHub account, never `outdoor-escapes`; Windows Credential Manager may hold either, so `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push` uses gh's active `jeffwarr4` login.
