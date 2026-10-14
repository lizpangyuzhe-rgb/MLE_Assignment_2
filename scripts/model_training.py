import os
import json
import joblib
from datetime import datetime, timezone

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ========================================================
# 1. CONFIGURATION
# ========================================================

FEATURE_PATH = (
    "datamart/gold/feature_store/"
    "gold_feature_store.parquet"
)

LABEL_PATH = "datamart/gold/label_store/"

MODEL_DIR = "model_bank"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_pipeline.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

FINAL_THRESHOLD = 0.45

os.makedirs(MODEL_DIR, exist_ok=True)


# ========================================================
# 2. LOAD GOLD FEATURE + LABEL STORES
# ========================================================

print("Loading Gold feature and label stores...")

features = pd.read_parquet(
    FEATURE_PATH
)

labels = pd.read_parquet(
    LABEL_PATH
)

print("Feature rows:", len(features))
print("Label rows:", len(labels))


# ========================================================
# 3. BUILD MODELING DATASET
# ========================================================

df = features.merge(
    labels[
        [
            "Customer_ID",
            "label"
        ]
    ],
    on="Customer_ID",
    how="inner"
)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

print("Joined rows:", len(df))
print(
    "Overall default rate:",
    round(df["label"].mean(), 4)
)


# ========================================================
# 4. TEMPORAL TRAIN / OOT SPLIT
# ========================================================
#
# Model development used:
#
# Train:
# before 2024-01-01
#
# Validation:
# 2024-01-01 to 2024-06-30
#
# Final production model:
# Train + Validation combined
#
# OOT test:
# 2024-07-01 onwards
#
# ========================================================

final_train_df = df[
    df["snapshot_date"] < "2024-07-01"
].copy()

oot_df = df[
    df["snapshot_date"] >= "2024-07-01"
].copy()

print("\nFinal temporal split:")
print(
    "Final training rows:",
    len(final_train_df)
)
print(
    "OOT rows:",
    len(oot_df)
)


# ========================================================
# 5. DEFINE MODEL FEATURES
# ========================================================

non_model_columns = [
    "Customer_ID",
    "snapshot_date",
    "label"
]

feature_columns = [
    c
    for c in df.columns
    if c not in non_model_columns
]

X_train = final_train_df[
    feature_columns
]

y_train = final_train_df[
    "label"
]

X_oot = oot_df[
    feature_columns
]

y_oot = oot_df[
    "label"
]


categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()
)

numeric_features = [
    c
    for c in feature_columns
    if c not in categorical_features
]

print("\nModel inputs:")
print(
    "Total features:",
    len(feature_columns)
)
print(
    "Numeric:",
    len(numeric_features)
)
print(
    "Categorical:",
    len(categorical_features)
)


# ========================================================
# 6. PREPROCESSING PIPELINE
# ========================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ========================================================
# 7. CLASS IMBALANCE
# ========================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative_count
    / positive_count
)

print(
    "\nFinal training scale_pos_weight:",
    round(
        scale_pos_weight,
        4
    )
)


# ========================================================
# 8. FINAL XGBOOST CONFIGURATION
# ========================================================
#
# Selected using validation PR-AUC.
#
# ========================================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=1.0,
    colsample_bytree=0.6,
    gamma=0.1,
    reg_alpha=0.0,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


model_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        xgb_model
    )
])


# ========================================================
# 9. TRAIN FINAL MODEL
# ========================================================

print(
    "\nTraining final XGBoost model..."
)

model_pipeline.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# ========================================================
# 10. FINAL OOT EVALUATION
# ========================================================

oot_probabilities = (
    model_pipeline
    .predict_proba(
        X_oot
    )[:, 1]
)

oot_predictions = (
    oot_probabilities
    >= FINAL_THRESHOLD
).astype(int)


oot_metrics = {
    "roc_auc": float(
        roc_auc_score(
            y_oot,
            oot_probabilities
        )
    ),

    "pr_auc": float(
        average_precision_score(
            y_oot,
            oot_probabilities
        )
    ),

    "accuracy": float(
        accuracy_score(
            y_oot,
            oot_predictions
        )
    ),

    "precision": float(
        precision_score(
            y_oot,
            oot_predictions,
            zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_oot,
            oot_predictions,
            zero_division=0
        )
    ),

    "f1": float(
        f1_score(
            y_oot,
            oot_predictions,
            zero_division=0
        )
    )
}


print(
    "\n=== FINAL OOT PERFORMANCE ==="
)

for metric, value in oot_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


cm = confusion_matrix(
    y_oot,
    oot_predictions
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ========================================================
# 11. SAVE PRODUCTION MODEL
# ========================================================

joblib.dump(
    model_pipeline,
    MODEL_PATH
)


# ========================================================
# 12. SAVE MODEL METADATA
# ========================================================

metadata = {

    "model_name":
        "xgboost_default_risk",

    "model_type":
        "XGBClassifier",

    "trained_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "decision_threshold":
        FINAL_THRESHOLD,

    "training_period":
        "before 2024-07-01",

    "oot_period":
        "2024-07-01 onwards",

    "training_rows":
        int(
            len(final_train_df)
        ),

    "oot_rows":
        int(
            len(oot_df)
        ),

    "feature_count":
        len(feature_columns),

    "feature_columns":
        feature_columns,

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "scale_pos_weight":
        float(
            scale_pos_weight
        ),

    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 0.6,
        "gamma": 0.1,
        "reg_alpha": 0.0,
        "reg_lambda": 1
    },

    "model_selection": {
        "primary_metric":
            "PR-AUC",

        "threshold_selection":
            "Maximum validation F1",

        "validation_threshold":
            0.45
    },

    "oot_metrics":
        oot_metrics,

    "oot_confusion_matrix":
        cm.tolist()
}


with open(
    METADATA_PATH,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ========================================================
# 13. SUMMARY
# ========================================================

print(
    "\n=== PRODUCTION ARTIFACTS SAVED ==="
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Metadata:",
    METADATA_PATH
)

print(
    "Decision threshold:",
    FINAL_THRESHOLD
)