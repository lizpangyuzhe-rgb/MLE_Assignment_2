import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

import utils.data_processing_bronze_table
import utils.data_processing_bronze_features
import utils.data_processing_silver_table
import utils.data_processing_silver_features
import utils.data_processing_gold_table
import utils.data_processing_gold_features


# ============================================================
# INITIALIZE SPARK
# ============================================================

spark = (
    pyspark.sql.SparkSession.builder
    .appName("assignment_1")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# DATE HELPER
# ============================================================

def generate_first_of_month_dates(start_date_str, end_date_str):

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []

    current_date = datetime(
        start_date.year,
        start_date.month,
        1
    )

    while current_date <= end_date:

        first_of_month_dates.append(
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

    return first_of_month_dates


# ============================================================
# DATE RANGES
# ============================================================

# Attributes + Financials
feature_snapshot_dates = generate_first_of_month_dates(
    "2023-01-01",
    "2025-01-01"
)

# Clickstream
clickstream_snapshot_dates = generate_first_of_month_dates(
    "2023-01-01",
    "2024-12-01"
)

# LMS repayment history
lms_snapshot_dates = generate_first_of_month_dates(
    "2023-01-01",
    "2025-11-01"
)


# ============================================================
# 1. FEATURE STORE PIPELINE
# ============================================================

# ============================================================
# 1A. FEATURE STORE - BRONZE
# ============================================================

bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

for directory in [
    bronze_attributes_directory,
    bronze_financials_directory,
    bronze_clickstream_directory
]:
    os.makedirs(directory, exist_ok=True)


# Attributes + Financials Bronze

for date_str in feature_snapshot_dates:

    utils.data_processing_bronze_features.process_bronze_attributes(
        date_str,
        bronze_attributes_directory,
        spark
    )

    utils.data_processing_bronze_features.process_bronze_financials(
        date_str,
        bronze_financials_directory,
        spark
    )


# Clickstream Bronze

for date_str in clickstream_snapshot_dates:

    utils.data_processing_bronze_features.process_bronze_clickstream(
        date_str,
        bronze_clickstream_directory,
        spark
    )


# ============================================================
# 1B. FEATURE STORE - SILVER
# ============================================================

silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"

for directory in [
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory
]:
    os.makedirs(directory, exist_ok=True)


# Attributes + Financials Silver

for date_str in feature_snapshot_dates:

    utils.data_processing_silver_features.process_silver_attributes(
        date_str,
        bronze_attributes_directory,
        silver_attributes_directory,
        spark
    )

    utils.data_processing_silver_features.process_silver_financials(
        date_str,
        bronze_financials_directory,
        silver_financials_directory,
        spark
    )


# Clickstream Silver

for date_str in clickstream_snapshot_dates:

    utils.data_processing_silver_features.process_silver_clickstream(
        date_str,
        bronze_clickstream_directory,
        silver_clickstream_directory,
        spark
    )

# ============================================================
# 1C. FEATURE STORE - GOLD
# ============================================================

gold_feature_store_directory = "datamart/gold/feature_store/"

os.makedirs(
    gold_feature_store_directory,
    exist_ok=True
)

gold_feature_df = (
    utils.data_processing_gold_features.process_gold_feature_store(
        silver_attributes_directory,
        silver_financials_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark
    )
)

gold_feature_df.show(
    10,
    truncate=False
)

# ============================================================
# 2. LABEL STORE PIPELINE
# ============================================================

# ============================================================
# 2A. LABEL STORE - BRONZE LMS
# ============================================================

bronze_lms_directory = "datamart/bronze/lms/"

os.makedirs(
    bronze_lms_directory,
    exist_ok=True
)

for date_str in lms_snapshot_dates:

    utils.data_processing_bronze_table.process_bronze_table(
        date_str,
        bronze_lms_directory,
        spark
    )


# ============================================================
# 2B. LABEL STORE - SILVER LMS
# ============================================================

silver_loan_daily_directory = "datamart/silver/loan_daily/"

os.makedirs(
    silver_loan_daily_directory,
    exist_ok=True
)

for date_str in lms_snapshot_dates:

    utils.data_processing_silver_table.process_silver_table(
        date_str,
        bronze_lms_directory,
        silver_loan_daily_directory,
        spark
    )


# ============================================================
# 2C. LABEL STORE - GOLD
# ============================================================

gold_label_store_directory = "datamart/gold/label_store/"

os.makedirs(
    gold_label_store_directory,
    exist_ok=True
)

for date_str in lms_snapshot_dates:

    utils.data_processing_gold_table.process_labels_gold_table(
        date_str,
        silver_loan_daily_directory,
        gold_label_store_directory,
        spark,
        dpd=30,
        mob=6
    )


# ============================================================
# VALIDATE GOLD LABEL STORE
# ============================================================

folder_path = gold_label_store_directory

files_list = [
    folder_path + os.path.basename(f)
    for f in glob.glob(
        os.path.join(folder_path, "*")
    )
]

if files_list:

    label_df = spark.read.parquet(
        *files_list
    )

    print(
        "Gold label store row count:",
        label_df.count()
    )

    label_df.show()

else:

    print(
        "No Gold label files were created."
    )