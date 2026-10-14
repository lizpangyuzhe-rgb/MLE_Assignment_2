import os
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType, DateType


# ============================================================
# SILVER ATTRIBUTES
# ============================================================

def process_silver_attributes(
    snapshot_date_str,
    bronze_attributes_directory,
    silver_attributes_directory,
    spark
):

    partition_name = (
        "bronze_attributes_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    bronze_filepath = bronze_attributes_directory + partition_name

    # If no Bronze file exists for this month, skip
    if not os.path.exists(bronze_filepath):
        print("no Bronze attributes file for:", snapshot_date_str)
        return None

    df = spark.read.csv(
        bronze_filepath,
        header=True,
        inferSchema=True
    )

    # ----------------------------
    # Clean Age
    # ----------------------------

    # Remove underscores, then cast to integer
    df = df.withColumn(
        "Age",
        F.regexp_replace(
            col("Age").cast("string"),
            "_",
            ""
        ).cast(IntegerType())
    )

    # Clearly impossible ages become NULL
    df = df.withColumn(
        "Age",
        F.when(
            (col("Age") >= 0) & (col("Age") <= 120),
            col("Age")
        ).otherwise(F.lit(None).cast(IntegerType()))
    )

    # ----------------------------
    # Clean Occupation
    # ----------------------------

    df = df.withColumn(
        "Occupation",
        F.when(
            col("Occupation").isNull()
            | (F.trim(col("Occupation")) == "")
            | col("Occupation").rlike("^_+$"),
            F.lit("UNKNOWN")
        ).otherwise(col("Occupation"))
    )

    # Enforce snapshot date
    df = df.withColumn(
        "snapshot_date",
        col("snapshot_date").cast(DateType())
    )

    # Remove exact duplicate records if any
    df = df.dropDuplicates()

    # ----------------------------
    # Save Silver
    # ----------------------------

    output_name = (
        "silver_attributes_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )

    output_path = silver_attributes_directory + output_name

    df.write.mode("overwrite").parquet(output_path)

    print(
        snapshot_date_str,
        "silver attributes row count:",
        df.count()
    )

    print("saved silver attributes to:", output_path)

    return df


# ============================================================
# SILVER FINANCIALS
# ============================================================

def process_silver_financials(
    snapshot_date_str,
    bronze_financials_directory,
    silver_financials_directory,
    spark
):

    partition_name = (
        "bronze_financials_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    bronze_filepath = bronze_financials_directory + partition_name

    if not os.path.exists(bronze_filepath):
        print("no Bronze financials file for:", snapshot_date_str)
        return None

    df = spark.read.csv(
        bronze_filepath,
        header=True,
        inferSchema=True
    )

    # ----------------------------
    # Numeric-string cleaning
    # ----------------------------

    double_columns = [
        "Annual_Income",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance"
    ]

    integer_columns = [
        "Num_of_Loan",
        "Num_of_Delayed_Payment"
    ]

    for column_name in double_columns:
        df = df.withColumn(
            column_name,
            F.regexp_replace(
                col(column_name).cast("string"),
                "_",
                ""
            ).cast(DoubleType())
        )

    for column_name in integer_columns:
        df = df.withColumn(
            column_name,
            F.regexp_replace(
                col(column_name).cast("string"),
                "_",
                ""
            ).cast(IntegerType())
        )

    # ----------------------------
    # Standardize categorical unknowns
    # ----------------------------

    df = df.withColumn(
        "Credit_Mix",
        F.when(
            col("Credit_Mix").isNull()
            | (F.trim(col("Credit_Mix")) == "")
            | col("Credit_Mix").rlike("^_+$"),
            F.lit("UNKNOWN")
        ).otherwise(col("Credit_Mix"))
    )

    df = df.withColumn(
        "Payment_of_Min_Amount",
        F.when(
            col("Payment_of_Min_Amount") == "NM",
            F.lit("UNKNOWN")
        ).otherwise(col("Payment_of_Min_Amount"))
    )

    # ----------------------------
    # Credit history age
    # "10 Years and 9 Months" -> 129 months
    # ----------------------------

    years = F.regexp_extract(
        col("Credit_History_Age"),
        r"(\d+)\s+Years",
        1
    ).cast(IntegerType())

    months = F.regexp_extract(
        col("Credit_History_Age"),
        r"(\d+)\s+Months",
        1
    ).cast(IntegerType())

    df = df.withColumn(
        "Credit_History_Age_Months",
        years * 12 + months
    )

    df = df.withColumn(
        "snapshot_date",
        col("snapshot_date").cast(DateType())
    )

    df = df.dropDuplicates()

    # ----------------------------
    # Save Silver
    # ----------------------------

    output_name = (
        "silver_financials_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )

    output_path = silver_financials_directory + output_name

    df.write.mode("overwrite").parquet(output_path)

    print(
        snapshot_date_str,
        "silver financials row count:",
        df.count()
    )

    print("saved silver financials to:", output_path)

    return df


# ============================================================
# SILVER CLICKSTREAM
# ============================================================

def process_silver_clickstream(
    snapshot_date_str,
    bronze_clickstream_directory,
    silver_clickstream_directory,
    spark
):

    partition_name = (
        "bronze_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    bronze_filepath = bronze_clickstream_directory + partition_name

    if not os.path.exists(bronze_filepath):
        print("no Bronze clickstream file for:", snapshot_date_str)
        return None

    df = spark.read.csv(
        bronze_filepath,
        header=True,
        inferSchema=True
    )

    # Enforce numeric feature types
    for i in range(1, 21):
        feature_name = f"fe_{i}"

        df = df.withColumn(
            feature_name,
            col(feature_name).cast(IntegerType())
        )

    df = df.withColumn(
        "snapshot_date",
        col("snapshot_date").cast(DateType())
    )

    # One row per customer per month
    df = df.dropDuplicates([
        "Customer_ID",
        "snapshot_date"
    ])

    # ----------------------------
    # Save Silver
    # ----------------------------

    output_name = (
        "silver_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )

    output_path = silver_clickstream_directory + output_name

    df.write.mode("overwrite").parquet(output_path)

    print(
        snapshot_date_str,
        "silver clickstream row count:",
        df.count()
    )

    print("saved silver clickstream to:", output_path)

    return df
    