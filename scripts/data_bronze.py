import os
from datetime import datetime

from pyspark.sql import SparkSession

import utils.data_processing_bronze_features as bronze_features
from utils.data_processing_bronze_table import process_bronze_table


# ========================================================
# DATE HELPER
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
                current_date.year + 1,
                1,
                1
            )
        else:
            current_date = datetime(
                current_date.year,
                current_date.month + 1,
                1
            )

    return dates


# ========================================================
# MAIN
# ========================================================

if __name__ == "__main__":

    print("\n=== BRONZE PIPELINE START ===")

    spark = (
        SparkSession.builder
        .appName("assignment2_bronze")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ----------------------------------------------------
    # Date ranges used in Assignment 1
    # ----------------------------------------------------

    feature_snapshot_dates = generate_first_of_month_dates(
        "2023-01-01",
        "2025-01-01"
    )

    clickstream_snapshot_dates = generate_first_of_month_dates(
        "2023-01-01",
        "2024-12-01"
    )

    lms_snapshot_dates = generate_first_of_month_dates(
        "2023-01-01",
        "2025-11-01"
    )

    # ----------------------------------------------------
    # Output directories
    # ----------------------------------------------------

    bronze_attributes_directory = (
        "datamart/bronze/attributes/"
    )

    bronze_financials_directory = (
        "datamart/bronze/financials/"
    )

    bronze_clickstream_directory = (
        "datamart/bronze/clickstream/"
    )

    bronze_directory = "datamart/bronze"

    for directory in [
        bronze_attributes_directory,
        bronze_financials_directory,
        bronze_clickstream_directory
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )

    # ----------------------------------------------------
    # Attributes + financials
    # ----------------------------------------------------

    print("\nProcessing attributes and financials...")

    for date_str in feature_snapshot_dates:

        bronze_features.process_bronze_attributes(
            date_str,
            bronze_attributes_directory,
            spark
        )

        bronze_features.process_bronze_financials(
            date_str,
            bronze_financials_directory,
            spark
        )

    # ----------------------------------------------------
    # Clickstream
    # ----------------------------------------------------

    print("\nProcessing clickstream...")

    for date_str in clickstream_snapshot_dates:

        bronze_features.process_bronze_clickstream(
            date_str,
            bronze_clickstream_directory,
            spark
        )

    # ----------------------------------------------------
    # Loan repayment / LMS
    # ----------------------------------------------------

    print("\nProcessing loan repayment data...")

    bronze_lms_directory = (
        "datamart/bronze/loan_daily/"
    )

    os.makedirs(
        bronze_lms_directory,
        exist_ok=True
    )

    for date_str in lms_snapshot_dates:

        process_bronze_table(
            date_str,
            bronze_lms_directory,
            spark
        )

    print("\n=== BRONZE PIPELINE COMPLETE ===")

    spark.stop()