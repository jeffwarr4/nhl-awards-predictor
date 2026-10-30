"""
nhl_train_v2.py

Trains the two Hart models on Data/Raw/nhl_historical_training.csv:
  - GradientBoostingRegressor on vote_share  → Models/Artifacts/hart_vote_reg_v2.pkl
  - LogisticRegression on is_top5             → Models/Artifacts/hart_top5_clf_v2.pkl

Evaluation is leave-one-season-out: for each season, fit on every other season,
score it, and report where the actual winner and top 5 landed on the blended
board (hart_common.hart_scores). Seasons are never split randomly. The shipped
models are then fit on ALL seasons TRAIN_START..TRAIN_END.

Run:
    python Models/nhl_train_v2.py                 # backtest + fit + save
    python Models/nhl_train_v2.py --no-backtest   # fit + save only
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hart_common import engineer_features, feature_frame, hart_scores
from config import (
    ARTIFACTS_DIR, BACKTEST_CSV, CATEGORICAL_FEATURES, FEATURES_PATH, MIN_GP_TRAIN,
    MODEL_REG_PATH, MODEL_TOP5_PATH, NUMERIC_FEATURES, ROOT_DIR, TRAIN_END, TRAIN_START, TRAINING_CSV,
)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    for col in ["VotePoints", "vote_share", "vote_rank", "is_winner", "is_top5", "team_pts_pct"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not in {TRAINING_CSV.name}. "
                             "Re-run Models/nhl_historical.py.")

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df = df[df["Season"].between(TRAIN_START, TRAIN_END)]
    df = engineer_features(df)
    df = df[df["GP"].fillna(0) >= MIN_GP_TRAIN].copy()

    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce").fillna(0)
    df["is_top5"] = pd.to_numeric(df["is_top5"], errors="coerce").fillna(0).astype(int)

    # Only seasons that actually have a vote
    voted = df.groupby("Season")["vote_share"].sum()
    return df[df["Season"].isin(voted[voted > 0].index)].reset_index(drop=True)


# ---------------------------------------------
# MODELS
# ---------------------------------------------
def make_models() -> tuple[Pipeline, Pipeline]:
    def preprocessor():
        return ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ], remainder="drop")

    reg = Pipeline([("preprocessor", preprocessor()),
                    ("regressor", GradientBoostingRegressor(random_state=42, n_estimators=300,
                                                            learning_rate=0.05, max_depth=3))])
    clf = Pipeline([("preprocessor", preprocessor()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    return reg, clf


def fit(df: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    reg, clf = make_models()
    X = feature_frame(df)
    reg.fit(X, df["vote_share"])
    clf.fit(X, df["is_top5"])
    return reg, clf


# ---------------------------------------------
# LEAVE-ONE-SEASON-OUT BACKTEST
# ---------------------------------------------
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(df["Season"].unique()):
        train, test = df[df["Season"] != season], df[df["Season"] == season].copy()
        reg, clf = fit(train)
        X = feature_frame(test)
        test["score"] = hart_scores(reg.predict(X), clf.predict_proba(X)[:, 1])["final"]
        test = test.sort_values("score", ascending=False).reset_index(drop=True)
        test["model_rank"] = test.index + 1

        real = test[test["vote_share"] > 0].sort_values("vote_share", ascending=False)
        top5 = real.head(5)
        winner = real.iloc[0]
        rows.append({
            "season": int(season),
            "winner": winner["Player"],
            "winner_rank": int(winner["model_rank"]),
            "top5_in_top5": int((top5["model_rank"] <= 5).sum()),
            "top5_in_top10": int((top5["model_rank"] <= 10).sum()),
            "model_top3": ", ".join(test.head(3)["Player"]),
        })
        print(f"  {season}: winner {winner['Player']} ranked #{rows[-1]['winner_rank']}")
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    return {
        "seasons": len(bt),
        "winner_rank1_pct": round(float((bt["winner_rank"] == 1).mean()), 3),
        "winner_top3_pct": round(float((bt["winner_rank"] <= 3).mean()), 3),
        "winner_rank_mean": round(float(bt["winner_rank"].mean()), 2),
        "top5_in_top5_mean": round(float(bt["top5_in_top5"].mean()), 2),
        "top5_in_top10_mean": round(float(bt["top5_in_top10"].mean()), 2),
    }


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def train_models(run_backtest: bool = True):
    df = load_training()
    seasons = sorted(int(s) for s in df["Season"].unique())
    print(f"[INFO] {len(df):,} player-seasons (GP >= {MIN_GP_TRAIN}), seasons {seasons[0]}–{seasons[-1]}")
    cov = df.loc[df["vote_share"] > 0, "team_pts_pct"].notna().mean()
    print(f"[INFO] team_pts_pct coverage among vote-getters: {cov:.1%}")

    summary = None
    if run_backtest:
        print("[INFO] Leave-one-season-out backtest...")
        bt = backtest(df)
        bt.to_csv(BACKTEST_CSV, index=False)
        summary = summarize(bt)
        print("\n" + bt.to_string(index=False))
        print(f"\n[METRIC] {summary}")
        print(f"[INFO] Backtest → {BACKTEST_CSV}")

    print(f"[INFO] Fitting shipped models on all {len(seasons)} seasons...")
    reg, clf = fit(df)
    joblib.dump(reg, MODEL_REG_PATH)
    joblib.dump(clf, MODEL_TOP5_PATH)

    features_info = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "trained_seasons": seasons,
        "evaluation": "leave-one-season-out over trained_seasons",
        "backtest_summary": summary,
        "targets": {"regression": "vote_share", "classification": "is_top5"},
        "models": {
            "regression_model_path": MODEL_REG_PATH.relative_to(ROOT_DIR).as_posix(),
            "top5_classifier_path": MODEL_TOP5_PATH.relative_to(ROOT_DIR).as_posix(),
        },
    }
    FEATURES_PATH.write_text(json.dumps(features_info, indent=2))
    print(f"[INFO] Saved {MODEL_REG_PATH.name}, {MODEL_TOP5_PATH.name}, {FEATURES_PATH.name}")
    print("[DONE] v2 training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-backtest", action="store_true", help="skip the leave-one-season-out backtest")
    train_models(run_backtest=not parser.parse_args().no_backtest)
