import os
import json
import joblib
import pandas as pd

from datetime import datetime, timezone


# ========================================================
# 1. PATHS
# ========================================================

FEATURE_PATH = (
    "datamart/gold/feature_store/"
    "gold_feature_store.parquet"
)

MODEL_PATH = (
    "model_bank/"
    "xgboost_pipeline.joblib"
)

METADATA_PATH = (
    "model_bank/"
    "model_metadata.json"
)

PREDICTION_DIR = (
    "datamart/gold/"
    "prediction_store/"
)

OUTPUT_PATH = os.path.join(
    PREDICTION_DIR,
    "predictions.parquet"
)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)


# ========================================================
# 2. LOAD MODEL + METADATA
# ========================================================

print("Loading production model...")

model_pipeline = joblib.load(
    MODEL_PATH
)

with open(
    METADATA_PATH,
    "r"
) as f:
    metadata = json.load(f)


decision_threshold = metadata[
    "decision_threshold"
]

feature_columns = metadata[
    "feature_columns"
]

model_name = metadata[
    "model_name"
]


print(
    "Model:",
    model_name
)

print(
    "Decision threshold:",
    decision_threshold
)


# ========================================================
# 3. LOAD GOLD FEATURE STORE
# ========================================================

features = pd.read_parquet(
    FEATURE_PATH
)

features["snapshot_date"] = (
    pd.to_datetime(
        features["snapshot_date"]
    )
)

print(
    "Gold feature rows:",
    len(features)
)


# ========================================================
# 4. DEFINE SCORING BATCH
# ========================================================

scoring_df = features[
    features["snapshot_date"]
    >= "2024-07-01"
].copy()

print(
    "Customers to score:",
    len(scoring_df)
)


# ========================================================
# 5. PREPARE MODEL INPUT
# ========================================================

X_score = scoring_df[
    feature_columns
]


# ========================================================
# 6. RUN INFERENCE
# ========================================================

print(
    "Running batch inference..."
)

default_probability = (
    model_pipeline
    .predict_proba(
        X_score
    )[:, 1]
)

prediction = (
    default_probability
    >= decision_threshold
).astype(int)


# ========================================================
# 7. BUILD PREDICTION STORE
# ========================================================

prediction_df = pd.DataFrame({

    "Customer_ID":
        scoring_df[
            "Customer_ID"
        ].values,

    "snapshot_date":
        scoring_df[
            "snapshot_date"
        ].values,

    "default_probability":
        default_probability,

    "prediction":
        prediction,

    "model_name":
        model_name,

    "decision_threshold":
        decision_threshold,

    "model_trained_at_utc":
        metadata[
            "trained_at_utc"
        ],

    "scored_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat()
})


# ========================================================
# 8. SAVE PREDICTIONS
# ========================================================

prediction_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ========================================================
# 9. SUMMARY
# ========================================================

print(
    "\n=== INFERENCE COMPLETE ==="
)

print(
    "Prediction rows:",
    len(prediction_df)
)

print(
    "Predicted defaults:",
    int(
        prediction_df[
            "prediction"
        ].sum()
    )
)

print(
    "Predicted default rate:",
    round(
        prediction_df[
            "prediction"
        ].mean(),
        4
    )
)

print(
    "Saved predictions to:",
    OUTPUT_PATH
)

print(
    "\nSample predictions:"
)

print(
    prediction_df.head(10)
)