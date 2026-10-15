import os
from datetime import datetime

from pyspark.sql import SparkSession

import utils.data_processing_silver_features as silver_features
from utils.data_processing_silver_table import process_silver_table


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

    print("\n=== SILVER PIPELINE START ===")

    spark = (
        SparkSession.builder
        .appName("assignment2_silver")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # ----------------------------------------------------
    # Date ranges
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
    # Directories
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

    silver_attributes_directory = (
        "datamart/silver/attributes/"
    )

    silver_financials_directory = (
        "datamart/silver/financials/"
    )

    silver_clickstream_directory = (
        "datamart/silver/clickstream/"
    )

    bronze_directory = "datamart/bronze"
    silver_directory = "datamart/silver"

    for directory in [
        silver_attributes_directory,
        silver_financials_directory,
        silver_clickstream_directory
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )

    # ----------------------------------------------------
    # Attributes + financials
    # ----------------------------------------------------

    print("\nCleaning attributes and financials...")

    for date_str in feature_snapshot_dates:

        silver_features.process_silver_attributes(
            date_str,
            bronze_attributes_directory,
            silver_attributes_directory,
            spark
        )

        silver_features.process_silver_financials(
            date_str,
            bronze_financials_directory,
            silver_financials_directory,
            spark
        )

    # ----------------------------------------------------
    # Clickstream
    # ----------------------------------------------------

    print("\nCleaning clickstream...")

    for date_str in clickstream_snapshot_dates:

        silver_features.process_silver_clickstream(
            date_str,
            bronze_clickstream_directory,
            silver_clickstream_directory,
            spark
        )

    # ----------------------------------------------------
    # Loan repayment / LMS
    # ----------------------------------------------------

    print("\nCleaning loan repayment data...")

    bronze_lms_directory = "datamart/bronze/loan_daily/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"

    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    for date_str in lms_snapshot_dates:

        process_silver_table(
            date_str,
            bronze_lms_directory,
            silver_loan_daily_directory,
            spark
        )

    print("\n=== SILVER PIPELINE COMPLETE ===")

    spark.stop()