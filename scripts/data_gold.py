import os
import glob
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession

import utils.data_processing_gold_features as gold_features
import utils.data_processing_gold_table as gold_labels


# ========================================================
# 1. PROJECT SETUP
# ========================================================

# Ensure relative paths work when executed by Airflow
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


# ========================================================
# 2. DATE HELPER
# ========================================================

def generate_first_of_month_dates(start_date_str, end_date_str):

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []

    current_date = datetime(
        start_date.year,
        start_date.month,
        1
    )

    while current_date <= end_date:

        dates.append(
            current_date.strftime("%Y-%m-%d")
        )

        if current_date.month == 12:
            current_date = datetime(
                current_date.year + 1, 1, 1
            )
        else:
            current_date = datetime(
                current_date.year,
                current_date.month + 1,
                1
            )

    return dates


# ========================================================
# 3. MAIN
# ========================================================

if __name__ == "__main__":

    print("\n=== GOLD PIPELINE START ===")

    spark = (
        SparkSession.builder
        .appName("assignment2_gold")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        # ====================================================
        # 4. DIRECTORIES
        # ====================================================

        silver_attributes_directory = (
            "datamart/silver/attributes/"
        )

        silver_financials_directory = (
            "datamart/silver/financials/"
        )

        silver_clickstream_directory = (
            "datamart/silver/clickstream/"
        )

        silver_loan_daily_directory = (
            "datamart/silver/loan_daily/"
        )

        gold_feature_store_directory = (
            "datamart/gold/feature_store/"
        )

        gold_label_store_directory = (
            "datamart/gold/label_store/"
        )

        os.makedirs(
            gold_feature_store_directory,
            exist_ok=True
        )

        os.makedirs(
            gold_label_store_directory,
            exist_ok=True
        )

        # ====================================================
        # 5. GOLD FEATURE STORE
        # ====================================================

        print("\nBuilding Gold feature store...")

        feature_df = (
            gold_features.process_gold_feature_store(
                silver_attributes_directory,
                silver_financials_directory,
                silver_clickstream_directory,
                gold_feature_store_directory,
                spark
            )
        )

        feature_rows = feature_df.count()
        feature_columns = len(feature_df.columns)

        print("\n=== FEATURE STORE SUMMARY ===")
        print("Rows:", feature_rows)
        print("Columns:", feature_columns)

        # ====================================================
        # 6. GOLD LABEL STORE
        # ====================================================

        print("\nBuilding Gold label store...")

        # Preserve original A1 LMS snapshot range
        lms_snapshot_dates = generate_first_of_month_dates(
            "2023-01-01",
            "2025-11-01"
        )

        for date_str in lms_snapshot_dates:

            gold_labels.process_labels_gold_table(
                date_str,
                silver_loan_daily_directory,
                gold_label_store_directory,
                spark,
                dpd=30,
                mob=6
            )

        # ====================================================
        # 7. LOAD GENERATED LABEL STORE
        # ====================================================

        label_files = sorted(
            glob.glob(
                gold_label_store_directory
                + "gold_label_store_*.parquet"
            )
        )

        if not label_files:
            raise RuntimeError(
                "No Gold label Parquet files were generated."
            )

        label_df = spark.read.parquet(
            *label_files
        )

        label_rows = label_df.count()

        print("\n=== LABEL STORE SUMMARY ===")
        print("Rows:", label_rows)
        print("Snapshot files:", len(label_files))

        # ====================================================
        # 8. FEATURE-LABEL COMPATIBILITY
        # ====================================================

        print("\nChecking feature-label compatibility...")

        joined_df = feature_df.join(
            label_df.select(
                "Customer_ID",
                "label"
            ),
            on="Customer_ID",
            how="inner"
        )

        joined_rows = joined_df.count()

        print("\n=== FINAL GOLD VALIDATION ===")

        print("Feature rows:", feature_rows)
        print("Feature columns:", feature_columns)
        print("Label rows:", label_rows)
        print("Joined rows:", joined_rows)

        if joined_rows != feature_rows:
            raise ValueError(
                "Feature-label row count mismatch. "
                "Check missing or duplicate Customer_ID values."
            )

        print("\n=== GOLD PIPELINE COMPLETE ===")

    finally:
        spark.stop()