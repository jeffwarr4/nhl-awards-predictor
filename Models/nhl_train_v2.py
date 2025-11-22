import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------
# PATHS
# ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "Data" / "Raw" / "nhl_historical_training.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "Models" / "Artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REG_PATH = ARTIFACTS_DIR / "hart_vote_reg_v2.pkl"
MODEL_TOP5_PATH = ARTIFACTS_DIR / "hart_top5_clf_v2.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "hart_features_v2.json"


# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
def load_historical() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure Season is int
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    # Ensure vote-related columns exist
    for col in ["VotePoints", "vote_share", "vote_rank", "is_winner", "is_top5"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in historical data. "
                "Did you run the updated nhl_historical.py?"
            )

    # Coerce types
    df["VotePoints"] = pd.to_numeric(df["VotePoints"], errors="coerce").fillna(0)
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce").fillna(0)
    df["is_winner"] = pd.to_numeric(df["is_winner"], errors="coerce").fillna(0).astype(int)
    df["is_top5"] = pd.to_numeric(df["is_top5"], errors="coerce").fillna(0).astype(int)

    return df


# ---------------------------------------------
# FEATURE ENGINEERING (aligned with predict_current)
# ---------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Filter out very low GP seasons
    if "GP" in df.columns:
        df = df[df["GP"].fillna(0) >= 20].copy()

    # Helper: safe ratio
    def safe_ratio(num_col, den_col, new_col):
        if num_col in df.columns and den_col in df.columns:
            denom = df[den_col].replace(0, np.nan)
            df[new_col] = df[num_col] / denom
        else:
            df[new_col] = np.nan

    # Base per-game stats
    safe_ratio("G", "GP", "G_per_GP")
    safe_ratio("A", "GP", "A_per_GP")
    safe_ratio("PTS", "GP", "PTS_per_GP")
    safe_ratio("SOG", "GP", "SOG_per_GP")
    safe_ratio("PPG", "GP", "PPG_per_GP")
    safe_ratio("SHG", "GP", "SHG_per_GP")

    # Team context: team_win_pct
    # Historical set usually has W and L from standings merge
    if {"W", "L"}.issubset(df.columns):
        games = (df["W"].fillna(0) + df["L"].fillna(0)).replace(0, np.nan)
        df["team_win_pct"] = df["W"] / games
    elif "team_win_pct" not in df.columns:
        df["team_win_pct"] = np.nan

    # Make sure PTS_team exists (even if NaN)
    if "PTS_team" not in df.columns:
        df["PTS_team"] = np.nan

    # Age numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Goalie flag
    if "is_goalie" not in df.columns:
        if "Pos" in df.columns:
            df["is_goalie"] = (df["Pos"] == "G").astype(int)
        else:
            df["is_goalie"] = 0

    # Fill basic counting stats NaNs with 0 where it makes sense
    for col in ["G", "A", "PTS", "SOG", "PPG", "SHG", "W", "L"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ---------------------------------------------
# BUILD X, y FOR REGRESSION + CLASSIFICATION
# ---------------------------------------------
def build_xy(df: pd.DataFrame):
    df = df.copy()

    # Numeric features: smaller, stronger set
    numeric_candidates = [
        "Age",
        "GP",
        "G",
        "A",
        "PTS",
        "G_per_GP",
        "A_per_GP",
        "PTS_per_GP",
        "SOG_per_GP",
        "PPG",
        "PPG_per_GP",
        "team_win_pct",
        "PTS_team",
        "is_goalie",
    ]
    numeric_features = [c for c in numeric_candidates if c in df.columns]

    # Categorical features: keep only position
    cat_candidates = ["Pos"]
    categorical_features = [c for c in cat_candidates if c in df.columns]

    # Targets
    if "vote_share" not in df.columns or "is_top5" not in df.columns:
        raise ValueError("Expected 'vote_share' and 'is_top5' in dataframe.")

    # Drop rows missing vote_share (shouldn't be many)
    df = df[df["vote_share"].notna()].copy()

    X = df[numeric_features + categorical_features].copy()
    y_reg = df["vote_share"].astype(float)
    y_top5 = df["is_top5"].astype(int)

    return X, y_reg, y_top5, numeric_features, categorical_features, df


# ---------------------------------------------
# TRAIN MODELS
# ---------------------------------------------
def train_models():
    print(f"[INFO] Loading historical data from: {DATA_RAW}")
    df = load_historical()
    print(f"[INFO] Loaded {len(df):,} rows before feature engineering")

    df = engineer_features(df)
    print(f"[INFO] {len(df):,} rows after GP filter & feature engineering")

    # Restrict to seasons that actually have voting data (just in case)
    seasons_with_votes = df.groupby("Season")["VotePoints"].sum()
    good_seasons = seasons_with_votes[seasons_with_votes > 0].index.tolist()
    df = df[df["Season"].isin(good_seasons)].copy()

    X, y_reg, y_top5, numeric_features, categorical_features, df_xy = build_xy(df)

    print(f"[INFO] Using {len(numeric_features)} numeric features and "
          f"{len(categorical_features)} categorical features")
    print("[DEBUG] Numeric features:", numeric_features)
    print("[DEBUG] Categorical features:", categorical_features)

    # Train/test split by season to avoid leakage
    seasons = sorted(df_xy["Season"].dropna().unique())
    if len(seasons) > 3:
        test_seasons = seasons[-3:]
    else:
        test_seasons = seasons[-1:]

    train_mask = ~df_xy["Season"].isin(test_seasons)
    test_mask = df_xy["Season"].isin(test_seasons)

    X_train, X_test = X[train_mask], X[test_mask]
    y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
    y_top5_train, y_top5_test = y_top5[train_mask], y_top5[test_mask]

    print(f"[INFO] Training on {len(X_train):,} rows from seasons "
          f"{sorted(df_xy['Season'][train_mask].dropna().unique())}")
    print(f"[INFO] Testing  on {len(X_test):,} rows from seasons {sorted(test_seasons)}")

    # Preprocessing: impute missing, scale numeric, one-hot encode categorical
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # --------------------------
    # Model A: Regression on vote_share
    # --------------------------
    regressor = GradientBoostingRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
    )

    reg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    print("[INFO] Fitting regression model (vote_share)...")
    reg_model.fit(X_train, y_reg_train)

    y_reg_train_pred = reg_model.predict(X_train)
    y_reg_test_pred = reg_model.predict(X_test)

    reg_train_r2 = r2_score(y_reg_train, y_reg_train_pred)
    reg_test_r2 = r2_score(y_reg_test, y_reg_test_pred)
    reg_train_mae = mean_absolute_error(y_reg_train, y_reg_train_pred)
    reg_test_mae = mean_absolute_error(y_reg_test, y_reg_test_pred)

    print(f"[METRIC][REG] Train R2:  {reg_train_r2:.3f}")
    print(f"[METRIC][REG] Test  R2:  {reg_test_r2:.3f}")
    print(f"[METRIC][REG] Train MAE: {reg_train_mae:.4f}")
    print(f"[METRIC][REG] Test  MAE: {reg_test_mae:.4f}")

    # --------------------------
    # Model B: Classification on is_top5
    # --------------------------
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    clf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    print("[INFO] Fitting classification model (is_top5)...")
    clf_model.fit(X_train, y_top5_train)

    y_clf_train_prob = clf_model.predict_proba(X_train)[:, 1]
    y_clf_test_prob = clf_model.predict_proba(X_test)[:, 1]

    try:
        clf_train_auc = roc_auc_score(y_top5_train, y_clf_train_prob)
        clf_test_auc = roc_auc_score(y_top5_test, y_clf_test_prob)
        clf_train_ap = average_precision_score(y_top5_train, y_clf_train_prob)
        clf_test_ap = average_precision_score(y_top5_test, y_clf_test_prob)
    except ValueError:
        clf_train_auc = np.nan
        clf_test_auc = np.nan
        clf_train_ap = np.nan
        clf_test_ap = np.nan

    print(f"[METRIC][TOP5] Train ROC-AUC: {clf_train_auc:.3f}")
    print(f"[METRIC][TOP5] Test  ROC-AUC: {clf_test_auc:.3f}")
    print(f"[METRIC][TOP5] Train AP:      {clf_train_ap:.3f}")
    print(f"[METRIC][TOP5] Test  AP:      {clf_test_ap:.3f}")

    # --------------------------
    # Save artifacts
    # --------------------------
    print(f"[INFO] Saving regression model to: {MODEL_REG_PATH}")
    joblib.dump(reg_model, MODEL_REG_PATH)

    print(f"[INFO] Saving top5 classifier model to: {MODEL_TOP5_PATH}")
    joblib.dump(clf_model, MODEL_TOP5_PATH)

    # feature metadata
    test_seasons_serializable = [int(s) for s in test_seasons]

    features_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_seasons": test_seasons_serializable,
        "targets": {
            "regression": "vote_share",
            "classification": "is_top5",
        },
        "models": {
            "regression_model_path": str(MODEL_REG_PATH),
            "top5_classifier_path": str(MODEL_TOP5_PATH),
        },
    }

    print(f"[INFO] Saving feature info to: {FEATURES_PATH}")
    FEATURES_PATH.write_text(json.dumps(features_info, indent=2))

    print("[DONE] v2 training complete.")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    train_models()
