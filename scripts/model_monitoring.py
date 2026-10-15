import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, timezone

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)


# ========================================================
# 1. PATHS
# ========================================================

FEATURE_PATH = (
    "datamart/gold/feature_store/"
    "gold_feature_store.parquet"
)

LABEL_PATH = (
    "datamart/gold/label_store/"
)

PREDICTION_PATH = (
    "datamart/gold/prediction_store/"
    "predictions.parquet"
)

MODEL_PATH = (
    "model_bank/"
    "xgboost_pipeline.joblib"
)

METADATA_PATH = (
    "model_bank/"
    "model_metadata.json"
)

MONITORING_DIR = (
    "datamart/gold/monitoring/"
)

REPORT_PATH = os.path.join(
    MONITORING_DIR,
    "monitoring_report.json"
)

os.makedirs(
    MONITORING_DIR,
    exist_ok=True
)


# ========================================================
# 2. LOAD DATA + MODEL METADATA
# ========================================================

print("Loading monitoring inputs...")

features = pd.read_parquet(
    FEATURE_PATH
)

labels = pd.read_parquet(
    LABEL_PATH
)

predictions = pd.read_parquet(
    PREDICTION_PATH
)

model_pipeline = joblib.load(
    MODEL_PATH
)

with open(
    METADATA_PATH,
    "r"
) as f:
    metadata = json.load(f)


features["snapshot_date"] = pd.to_datetime(
    features["snapshot_date"]
)

predictions["snapshot_date"] = pd.to_datetime(
    predictions["snapshot_date"]
)


# ========================================================
# 3. JOIN PREDICTIONS WITH MATURED LABELS
# ========================================================

monitor_df = predictions.merge(
    labels[
        [
            "Customer_ID",
            "label"
        ]
    ],
    on="Customer_ID",
    how="inner"
)

print(
    "Prediction rows:",
    len(predictions)
)

print(
    "Rows with matured labels:",
    len(monitor_df)
)


# ========================================================
# 4. PERFORMANCE METRICS
# ========================================================

y_true = monitor_df["label"]

y_prob = monitor_df[
    "default_probability"
]

y_pred = monitor_df[
    "prediction"
]


performance_metrics = {

    "roc_auc": float(
        roc_auc_score(
            y_true,
            y_prob
        )
    ),

    "pr_auc": float(
        average_precision_score(
            y_true,
            y_prob
        )
    ),

    "accuracy": float(
        accuracy_score(
            y_true,
            y_pred
        )
    ),

    "precision": float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    ),

    "f1": float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    ),

    "brier_score": float(
        brier_score_loss(
            y_true,
            y_prob
        )
    )
}


# ========================================================
# 5. BASIC OPERATIONAL METRICS
# ========================================================

operational_metrics = {

    "scored_rows":
        int(
            len(predictions)
        ),

    "predicted_defaults":
        int(
            predictions[
                "prediction"
            ].sum()
        ),

    "predicted_default_rate":
        float(
            predictions[
                "prediction"
            ].mean()
        ),

    "mean_default_probability":
        float(
            predictions[
                "default_probability"
            ].mean()
        )
}


# ========================================================
# 6. PSI FUNCTION
# ========================================================

def calculate_psi(
    expected,
    actual,
    bins=10
):

    expected = (
        pd.Series(expected)
        .dropna()
        .astype(float)
    )

    actual = (
        pd.Series(actual)
        .dropna()
        .astype(float)
    )

    if (
        len(expected) == 0
        or len(actual) == 0
    ):
        return None

    breakpoints = np.unique(
        np.quantile(
            expected,
            np.linspace(
                0,
                1,
                bins + 1
            )
        )
    )

    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints
    )

    expected_pct = (
        expected_counts
        / expected_counts.sum()
    )

    actual_pct = (
        actual_counts
        / actual_counts.sum()
    )

    epsilon = 1e-6

    expected_pct = np.clip(
        expected_pct,
        epsilon,
        None
    )

    actual_pct = np.clip(
        actual_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (
            actual_pct
            - expected_pct
        )
        * np.log(
            actual_pct
            / expected_pct
        )
    )

    return float(psi)


def psi_status(value):

    if value is None:
        return "not_available"

    if value < 0.10:
        return "stable"

    elif value < 0.25:
        return "moderate_drift"

    else:
        return "significant_drift"


# ========================================================
# 7. DEFINE BASELINE + CURRENT POPULATIONS
# ========================================================

feature_columns = metadata[
    "feature_columns"
]

numeric_features = metadata[
    "numeric_features"
]


# Baseline = final training population
baseline_df = features[
    features["snapshot_date"]
    < "2024-07-01"
].copy()


# Current = production/OOT scoring population
current_df = features[
    features["snapshot_date"]
    >= "2024-07-01"
].copy()


print(
    "\nBaseline rows:",
    len(baseline_df)
)

print(
    "Current rows:",
    len(current_df)
)


# ========================================================
# 8. FEATURE PSI
# ========================================================

feature_psi = {}

for feature in numeric_features:

    psi = calculate_psi(
        baseline_df[feature],
        current_df[feature]
    )

    feature_psi[feature] = {
        "psi": (
            None
            if psi is None
            else round(psi, 6)
        ),
        "status": psi_status(psi)
    }


# ========================================================
# 9. MODEL SCORE PSI
# ========================================================

baseline_scores = (
    model_pipeline
    .predict_proba(
        baseline_df[
            feature_columns
        ]
    )[:, 1]
)

current_scores = predictions[
    "default_probability"
].values


score_psi = calculate_psi(
    baseline_scores,
    current_scores
)


score_stability = {
    "psi": round(
        score_psi,
        6
    ),
    "status": psi_status(
        score_psi
    )
}


# ========================================================
# 10. HIGHEST-DRIFT FEATURES
# ========================================================

ranked_features = sorted(
    [
        (
            feature,
            values["psi"]
        )
        for feature, values
        in feature_psi.items()
        if values["psi"] is not None
    ],
    key=lambda x: x[1],
    reverse=True
)

top_drift_features = [
    {
        "feature": feature,
        "psi": psi,
        "status": psi_status(psi)
    }
    for feature, psi
    in ranked_features[:10]
]


# ========================================================
# 11. MONTHLY PERFORMANCE
# ========================================================

monthly_performance = []

for month, group in monitor_df.groupby(
    monitor_df[
        "snapshot_date"
    ].dt.to_period("M")
):

    # Need both classes for AUC calculations
    if group["label"].nunique() < 2:
        continue

    monthly_performance.append({

        "month":
            str(month),

        "rows":
            int(
                len(group)
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    group["label"],
                    group[
                        "default_probability"
                    ]
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    group["label"],
                    group[
                        "default_probability"
                    ]
                )
            ),

        "recall":
            float(
                recall_score(
                    group["label"],
                    group[
                        "prediction"
                    ],
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    group["label"],
                    group[
                        "prediction"
                    ],
                    zero_division=0
                )
            )
    })


# ========================================================
# 12. SIMPLE MONITORING STATUS
# ========================================================

significant_drift_count = sum(
    1
    for values
    in feature_psi.values()
    if values["status"]
    == "significant_drift"
)

moderate_drift_count = sum(
    1
    for values
    in feature_psi.values()
    if values["status"]
    == "moderate_drift"
)


if (
    score_stability["status"]
    == "significant_drift"
    or significant_drift_count > 0
):

    monitoring_status = (
        "REVIEW_REQUIRED"
    )

elif (
    score_stability["status"]
    == "moderate_drift"
    or moderate_drift_count > 0
):

    monitoring_status = (
        "WATCH"
    )

else:

    monitoring_status = (
        "STABLE"
    )


# ========================================================
# 13. BUILD MONITORING REPORT
# ========================================================

report = {

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model_name":
        metadata[
            "model_name"
        ],

    "decision_threshold":
        metadata[
            "decision_threshold"
        ],

    "monitoring_status":
        monitoring_status,

    "performance_metrics":
        performance_metrics,

    "operational_metrics":
        operational_metrics,

    "stability_metrics": {

        "model_score_psi":
            score_stability,

        "significant_drift_feature_count":
            significant_drift_count,

        "moderate_drift_feature_count":
            moderate_drift_count,

        "top_drift_features":
            top_drift_features,

        "all_numeric_feature_psi":
            feature_psi
    },

    "monthly_performance":
        monthly_performance,

    "psi_thresholds": {
        "stable":
            "< 0.10",

        "moderate_drift":
            "0.10 - 0.25",

        "significant_drift":
            "> 0.25"
    }
}


# ========================================================
# 14. SAVE REPORT
# ========================================================

with open(
    REPORT_PATH,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ========================================================
# 15. SUMMARY
# ========================================================

print(
    "\n=== MODEL MONITORING ==="
)

print(
    "Monitoring status:",
    monitoring_status
)

print(
    "\nPerformance:"
)

for metric, value in performance_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


print(
    "\nModel score PSI:",
    round(
        score_psi,
        4
    ),
    "-",
    psi_status(
        score_psi
    )
)


print(
    "Moderate drift features:",
    moderate_drift_count
)

print(
    "Significant drift features:",
    significant_drift_count
)


print(
    "\nTop drift features:"
)

for item in top_drift_features[:5]:

    print(
        item[
            "feature"
        ],
        ":",
        round(
            item[
                "psi"
            ],
            4
        ),
        "-",
        item[
            "status"
        ]
    )


print(
    "\nSaved monitoring report to:",
    REPORT_PATH
)