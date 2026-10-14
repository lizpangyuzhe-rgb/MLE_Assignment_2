from datetime import datetime

from pyspark.sql.functions import col


def process_bronze_attributes(
    snapshot_date_str,
    bronze_attributes_directory,
    spark
):
    """
    Load one monthly snapshot from the raw customer attributes source
    and persist it to the Bronze layer.
    """

    snapshot_date = datetime.strptime(
        snapshot_date_str,
        "%Y-%m-%d"
    )

    # Raw source
    csv_file_path = "data/features_attributes.csv"

    # In production, this would ingest from the actual backend/source system.
    df = (
        spark.read
        .csv(
            csv_file_path,
            header=True,
            inferSchema=True
        )
        .filter(col("snapshot_date") == snapshot_date)
    )

    print(
        snapshot_date_str,
        "attributes row count:",
        df.count()
    )

    # Save source-faithful monthly Bronze partition
    partition_name = (
        "bronze_attributes_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    filepath = (
        bronze_attributes_directory
        + partition_name
    )

    df.toPandas().to_csv(
        filepath,
        index=False
    )

    print("saved attributes to:", filepath)

    return df


def process_bronze_financials(
    snapshot_date_str,
    bronze_financials_directory,
    spark
):
    """
    Load one monthly snapshot from the raw financials source
    and persist it to the Bronze layer.
    """

    snapshot_date = datetime.strptime(
        snapshot_date_str,
        "%Y-%m-%d"
    )

    # Raw source
    csv_file_path = "data/features_financials.csv"

    df = (
        spark.read
        .csv(
            csv_file_path,
            header=True,
            inferSchema=True
        )
        .filter(col("snapshot_date") == snapshot_date)
    )

    print(
        snapshot_date_str,
        "financials row count:",
        df.count()
    )

    partition_name = (
        "bronze_financials_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    filepath = (
        bronze_financials_directory
        + partition_name
    )

    df.toPandas().to_csv(
        filepath,
        index=False
    )

    print("saved financials to:", filepath)

    return df


def process_bronze_clickstream(
    snapshot_date_str,
    bronze_clickstream_directory,
    spark
):
    """
    Load one monthly clickstream snapshot and persist the raw
    customer-month records to the Bronze layer.

    No aggregation is performed here.
    """

    snapshot_date = datetime.strptime(
        snapshot_date_str,
        "%Y-%m-%d"
    )

    # Raw source
    csv_file_path = "data/feature_clickstream.csv"

    df = (
        spark.read
        .csv(
            csv_file_path,
            header=True,
            inferSchema=True
        )
        .filter(col("snapshot_date") == snapshot_date)
    )

    print(
        snapshot_date_str,
        "clickstream row count:",
        df.count()
    )

    partition_name = (
        "bronze_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    filepath = (
        bronze_clickstream_directory
        + partition_name
    )

    df.toPandas().to_csv(
        filepath,
        index=False
    )

    print("saved clickstream to:", filepath)

    return df
