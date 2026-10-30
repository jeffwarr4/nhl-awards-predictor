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

# Score the current season
python Models/nhl_predict_current.py
```

There are no tests or linters.

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

## Output contract (website)

stinger-website's NHL page reads a Google Sheet whose tabs IMPORTDATA the files in `pages/` (the workflow copies them there from `Data/Processed/`). `nhl_hart_predictions_v2_latest.csv` must keep exactly `PAGES_COLUMNS` (in `nhl_predict_current.py`); the Sheet derives `mvp_prob` from `Hart_Win_Prob` and `week_label` from `model_run_label`. `top5.csv` / `top6to10.csv` are wide one-row-per-run CSVs (`r1_player`, `r1_team`, …).

## Data Sources & Scraping

All data comes from Hockey-Reference.com with exponential backoff for 429/5xx. `clean_team_name` strips only markers (`*`, `+`, daggers, a leading `x-` clinch prefix). An older version stripped the letters x/y/z anywhere, which is why typo entries like "Calgar Flames" existed in old team maps.

## Model Artifacts

`.pkl` files in `Models/Artifacts/` **are tracked in git** (the `.gitignore` rules contradict each other, but they were committed before). The weekly workflow does not retrain: retrain locally and commit the artifacts.

## Automation

`.github/workflows/nhl_predict.yml` runs `nhl_predict_current.py` weekly, copies the CSVs to `pages/`, and auto-commits to `main`. As of Oct 2026 the workflow is **disabled** in GitHub and still targets Python 3.10; re-enabling it is a manual step.
