import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ---------------------------------------------
# PATHS
# ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "Data" / "Raw" / "nhl_historical_training.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "Models" / "Artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "hart_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "hart_preprocessor.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "hart_features.json"


# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
def load_historical() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)

    # Normalize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # Ensure Season is int
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(int)

    # Ensure VotePoints exists
    if "VotePoints" not in df.columns:
        df["VotePoints"] = 0

    # Some seasons may have NaN VotePoints; treat as 0
    df["VotePoints"] = df["VotePoints"].fillna(0)

    return df


# ---------------------------------------------
# LABEL TARGET: HART WINNER EACH SEASON
# ---------------------------------------------
def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Winner: player with max VotePoints in each season
    df["is_winner"] = 0

    # Only seasons where at least one player got votes
    seasons_with_votes = df.groupby("Season")["VotePoints"].max()
    seasons_with_votes = seasons_with_votes[seasons_with_votes > 0].index.tolist()

    for season in seasons_with_votes:
        season_mask = df["Season"] == season
        season_df = df[season_mask]

        if season_df["VotePoints"].max() <= 0:
            continue

        # Mark all tied max as winners (usually just one)
        max_pts = season_df["VotePoints"].max()
        winner_idx = season_df.index[season_df["VotePoints"] == max_pts]
        df.loc[winner_idx, "is_winner"] = 1

    return df


# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic filters: drop players with almost no games
    if "GP" in df.columns:
        df = df[df["GP"].fillna(0) >= 20]  # you can tweak this threshold

    # Create some derived features where columns exist
    def safe_ratio(num_col, den_col, new_col):
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / df[den_col].replace(0, np.nan)
        else:
            df[new_col] = np.nan

    # Offensive ratios
    safe_ratio("G", "GP", "G_per_GP")
    safe_ratio("A", "GP", "A_per_GP")
    safe_ratio("PTS", "GP", "PTS_per_GP")
    safe_ratio("SOG", "GP", "SOG_per_GP")

    # Special teams goals per game
    safe_ratio("PPG", "GP", "PPG_per_GP")
    safe_ratio("SHG", "GP", "SHG_per_GP")

    # Shooting percentage already exists as SPCT on skaters
    # For goalies we care about SV%, GAA, etc., when present.

    # Goalie-specific metrics will mostly already be in raw columns:
    # SV%, GAA, GSAA, QS%, etc. We'll just include them as numeric features if present.

    # Boolean: is goalie (already present in historical builder as is_goalie)
    if "is_goalie" not in df.columns:
        df["is_goalie"] = (df.get("Pos", "") == "G").astype(int)

    # Team context: team points, win pct
    if {"PTS", "W", "L"}.issubset(df.columns):
        df["team_win_pct"] = df["W"] / (df["W"] + df["L"]).replace(0, np.nan)
    else:
        df["team_win_pct"] = np.nan

    # Handle encoding / types
    # Age sometimes comes as float; keep as numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Fill some obvious NaNs with 0 for stats that are naturally 0 if missing
    for col in ["G", "A", "PTS", "PIM", "SOG", "PPG", "SHG", "W", "L", "VotePoints"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------
# BUILD X, y, FEATURE LISTS
# ---------------------------------------------
def build_xy(df: pd.DataFrame):
    df = df.copy()

    # Ensure target exists
    if "is_winner" not in df.columns:
        raise ValueError("Target column 'is_winner' not found. Did you call add_target_labels()?")

    # Candidate numeric features (use what exists)
    numeric_candidates = [
    "Age", "GP", "G", "A", "PTS", "+/-", "PIM", "SOG", "SPCT",
    "PPG", "SHG",
    "G_per_GP", "A_per_GP", "PTS_per_GP", "SOG_per_GP",
    "PPG_per_GP", "SHG_per_GP",
    "team_win_pct",
    "PTS_team",
    "is_goalie",
    "SV%", "GAA", "GSAA", "QS%", "GPS",
]


    numeric_features = [c for c in numeric_candidates if c in df.columns]

    # Candidate categorical features
    cat_candidates = [
        "Pos",
        #"Team",
    ]
    categorical_features = [c for c in cat_candidates if c in df.columns]

    # Drop any rows with missing target
    df = df[df["is_winner"].notna()].copy()

    # Feature matrix
    X = df[numeric_features + categorical_features].copy()
    y = df["is_winner"].astype(int)

    return X, y, numeric_features, categorical_features


# ---------------------------------------------
# TRAIN MODEL
# ---------------------------------------------
def train_model():
    print(f"[INFO] Loading historical data from: {DATA_RAW}")
    df = load_historical()
    print(f"[INFO] Loaded {len(df):,} rows")

    df = add_target_labels(df)
    df = engineer_features(df)

    # Remove any seasons with no winner label (shouldn't happen, but safe)
    seasons_has_winner = df.groupby("Season")["is_winner"].max()
    good_seasons = seasons_has_winner[seasons_has_winner == 1].index.tolist()
    df = df[df["Season"].isin(good_seasons)].copy()

    X, y, numeric_features, categorical_features = build_xy(df)

    print(f"[INFO] Using {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
    print("[DEBUG] Numeric features:", numeric_features)
    print("[DEBUG] Categorical features:", categorical_features)

    # Train/test split by season to avoid leakage
    seasons = sorted(df["Season"].unique())
    # Use last 3 seasons as test set if possible
    if len(seasons) > 3:
        test_seasons = seasons[-3:]
    else:
        test_seasons = seasons[-1:]

    train_mask = ~df["Season"].isin(test_seasons)
    test_mask = df["Season"].isin(test_seasons)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[INFO] Training on {len(X_train):,} rows from seasons {sorted(df['Season'][train_mask].unique())}")
    print(f"[INFO] Testing  on {len(X_test):,} rows from seasons {sorted(test_seasons)}")

    # Preprocessing: scale numeric, one-hot encode categorical
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    from sklearn.impute import SimpleImputer

    # Preprocessing: impute missing, scale numeric, one-hot encode categorical
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )


    # Classifier: start with logistic regression (probabilities are nice)
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",  # winners are rare
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    print("[INFO] Fitting model...")
    model.fit(X_train, y_train)

    # Evaluate
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    try:
        train_auc = roc_auc_score(y_train, train_probs)
        test_auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        train_auc = np.nan
        test_auc = np.nan

    print(f"[METRIC] Train ROC-AUC: {train_auc:.3f}")
    print(f"[METRIC] Test  ROC-AUC: {test_auc:.3f}")

    # Save artifacts
    print(f"[INFO] Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # Convert numpy int64s to plain Python ints for JSON
    test_seasons_serializable = [int(s) for s in test_seasons]


    # Also save feature metadata for use in current-season predictions
    features_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_seasons": test_seasons_serializable,
    }
    print(f"[INFO] Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # Also save feature metadata for use in current-season predictions
    test_seasons_serializable = [int(s) for s in test_seasons]
    features_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_seasons": test_seasons_serializable,
    }
    print(f"[INFO] Saving feature info to: {FEATURES_PATH}")
    FEATURES_PATH.write_text(json.dumps(features_info, indent=2))

    print("[DONE] Training complete.")



# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    train_model()
