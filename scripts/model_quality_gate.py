import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
)


# ========================================================
# 1. PATHS
# ========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = (
    PROJECT_ROOT
    / "datamart/gold/feature_store/gold_feature_store.parquet"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "datamart/gold/label_store"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "model_bank/xgboost_pipeline.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "model_bank/model_metadata.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "model_bank/quality_gate_report.json"
)


# ========================================================
# 2. MINIMUM ACCEPTANCE CRITERIA
# ========================================================

# Demonstration thresholds, not regulatory standards

MIN_ROC_AUC = 0.70
MIN_PR_AUC = 0.45
MIN_RECALL = 0.60

OOT_START = "2024-07-01"


# ========================================================
# 3. CHECK MODEL ARTIFACTS
# ========================================================

print("\n=== MODEL QUALITY GATE ===")

print("Checking model artifacts...")

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model artifact missing.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("Model metadata missing.")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]
threshold = float(metadata["decision_threshold"])

if not 0 < threshold < 1:
    raise ValueError("Invalid decision threshold.")

print("Model loaded successfully.")
print("Decision threshold:", threshold)


# ========================================================
# 4. LOAD EVALUATION DATA
# ========================================================

features = pd.read_parquet(FEATURE_PATH)
labels = pd.read_parquet(LABEL_PATH)

features["snapshot_date"] = pd.to_datetime(
    features["snapshot_date"]
)

# Historical OOT evaluation population
oot_features = features[
    features["snapshot_date"] >= OOT_START
].copy()

if oot_features.empty:
    raise ValueError("OOT evaluation population is empty.")

# Check feature schema
missing_features = [
    c for c in feature_columns
    if c not in oot_features.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

# Check customer uniqueness
if oot_features["Customer_ID"].duplicated().any():
    raise ValueError("Duplicate customers in OOT features.")

if labels["Customer_ID"].duplicated().any():
    raise ValueError("Duplicate customers in labels.")

# Join matured labels for evaluation
df = oot_features.merge(
    labels[["Customer_ID", "label"]],
    on="Customer_ID",
    how="inner",
    validate="one_to_one"
)

if len(df) != len(oot_features):
    raise ValueError("Some OOT customers have missing labels.")

print("OOT evaluation rows:", len(df))


# ========================================================
# 5. EVALUATE MODEL
# ========================================================

X = df[feature_columns]
y = df["label"]

if y.nunique() != 2:
    raise ValueError("Evaluation requires both label classes.")

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "roc_auc": float(
        roc_auc_score(y, probabilities)
    ),
    "pr_auc": float(
        average_precision_score(y, probabilities)
    ),
    "recall": float(
        recall_score(y, predictions, zero_division=0)
    )
}

print("\n=== PERFORMANCE CHECK ===")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# ========================================================
# 6. APPLY QUALITY GATE
# ========================================================

checks = {
    "roc_auc_pass": metrics["roc_auc"] >= MIN_ROC_AUC,
    "pr_auc_pass": metrics["pr_auc"] >= MIN_PR_AUC,
    "recall_pass": metrics["recall"] >= MIN_RECALL
}

passed = all(checks.values())

status = "PASS" if passed else "FAIL"

print("\n=== QUALITY GATE RESULT ===")

for check, result in checks.items():
    print(check, "PASS" if result else "FAIL")

print("Overall status:", status)


# ========================================================
# 7. SAVE QUALITY GATE REPORT
# ========================================================

report = {
    "generated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model_name": metadata["model_name"],
    "evaluation_period": f"{OOT_START} onward",
    "evaluation_rows": len(df),
    "decision_threshold": threshold,

    "metrics": metrics,

    "minimum_thresholds": {
        "roc_auc": MIN_ROC_AUC,
        "pr_auc": MIN_PR_AUC,
        "recall": MIN_RECALL
    },

    "checks": checks,
    "quality_gate_status": status
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print("\nSaved report to:", REPORT_PATH)


# ========================================================
# 8. BLOCK DEPLOYMENT IF FAILED
# ========================================================

if not passed:
    raise RuntimeError(
        "MODEL QUALITY GATE FAILED. "
        "Deployment should be blocked."
    )

print("\nModel approved for batch inference.")