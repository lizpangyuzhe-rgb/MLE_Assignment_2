import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_gold_feature_store(
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark
):

    # ========================================================
    # 1. LOAD ALL SILVER TABLES
    # ========================================================

    attributes_df = spark.read.parquet(
        silver_attributes_directory + "*"
    )

    financials_df = spark.read.parquet(
        silver_financials_directory + "*"
    )

    clickstream_df = spark.read.parquet(
        silver_clickstream_directory + "*"
    )


    # ========================================================
    # 2. JOIN ATTRIBUTES + FINANCIALS
    # ========================================================

    base_df = (
        attributes_df
        .select(
            "Customer_ID",
            "snapshot_date",
            "Age",
            "Occupation"
        )
        .join(
            financials_df,
            on=["Customer_ID", "snapshot_date"],
            how="inner"
        )
    )


    # ========================================================
    # 3. FINANCIAL FEATURE ENGINEERING
    # ========================================================

    # Existing feature:
    # Debt relative to annual income
    base_df = base_df.withColumn(
        "Debt_to_Income_Ratio",
        F.when(
            col("Annual_Income") > 0,
            col("Outstanding_Debt") / col("Annual_Income")
        )
    )

    # Existing feature:
    # Monthly EMI burden relative to monthly salary
    base_df = base_df.withColumn(
        "EMI_to_Income_Ratio",
        F.when(
            col("Monthly_Inhand_Salary") > 0,
            col("Total_EMI_per_month")
            / col("Monthly_Inhand_Salary")
        )
    )

    # Existing feature:
    # Remaining balance relative to monthly salary
    base_df = base_df.withColumn(
        "Balance_to_Income_Ratio",
        F.when(
            col("Monthly_Inhand_Salary") > 0,
            col("Monthly_Balance")
            / col("Monthly_Inhand_Salary")
        )
    )
    # --------------------------------------------------------
    # Sanitize implausible count values before engineering
    # peer-derived financial features
    # --------------------------------------------------------

    for feature in [
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Num_of_Loan"
    ]:
        base_df = base_df.withColumn(
            feature,
            F.when(
                (col(feature) >= 0) & (col(feature) <= 20),
                col(feature)
            )
        )

        median_value = base_df.approxQuantile(
            feature,
            [0.5],
            0.01
        )[0]

        base_df = base_df.fillna({
            feature: median_value
        })
    
    # --------------------------------------------------------
    # Cross-team features adopted from peer datamart
    # --------------------------------------------------------

    # Total number of financial products held
    base_df = base_df.withColumn(
        "Num_Fin_Pdts",
        col("Num_Bank_Accounts")
        + col("Num_Credit_Card")
        + col("Num_of_Loan")
    )

    # Salary remaining after monthly EMI commitments
    base_df = base_df.withColumn(
        "Repayment_Ability",
        col("Monthly_Inhand_Salary")
        - col("Total_EMI_per_month")
    )

    # Payment delay normalized by number of loans
    base_df = base_df.withColumn(
        "Loan_Extent",
        F.when(
            col("Num_of_Loan") > 0,
            F.greatest(
                col("Delay_from_due_date"),
                F.lit(0)
            ) / col("Num_of_Loan")
        ).otherwise(0.0)
    )


    # ========================================================
    # 4. POINT-IN-TIME CLICKSTREAM
    # ========================================================

    applications_df = base_df.select(
        "Customer_ID",
        col("snapshot_date").alias("application_date")
    )

    # Only use clickstream BEFORE loan application
    # and within the previous 3 months
    joined_clickstream = (
        applications_df.alias("a")
        .join(
            clickstream_df.alias("c"),
            (
                (col("a.Customer_ID") == col("c.Customer_ID"))
                &
                (col("c.snapshot_date") < col("a.application_date"))
                &
                (
                    col("c.snapshot_date")
                    >= F.add_months(
                        col("a.application_date"),
                        -3
                    )
                )
            ),
            how="inner"
        )
    )


    # ========================================================
    # 5. AGGREGATE CLICKSTREAM TO CUSTOMER LEVEL
    # ========================================================

    aggregation_expressions = [
        F.avg(col(f"c.fe_{i}")).alias(f"fe_{i}_3m_avg")
        for i in range(1, 21)
    ]

    clickstream_features = (
        joined_clickstream
        .groupBy(
            col("a.Customer_ID").alias("Customer_ID"),
            col("a.application_date").alias("snapshot_date")
        )
        .agg(
            F.count("*").alias("clickstream_months_available"),
            *aggregation_expressions
        )
    )


    # ========================================================
    # 6. JOIN CLICKSTREAM FEATURES TO BASE FEATURES
    # ========================================================

    gold_df = base_df.join(
        clickstream_features,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )

    # Customers without clickstream history
    gold_df = gold_df.withColumn(
        "has_clickstream_history",
        F.when(
            col("clickstream_months_available").isNotNull(),
            1
        ).otherwise(0)
    )

    gold_df = gold_df.fillna(
        0,
        subset=["clickstream_months_available"]
    )

    # Fill missing clickstream aggregates with 0.
    # Indicator above distinguishes "no history" from actual activity.
    clickstream_feature_names = [
        f"fe_{i}_3m_avg"
        for i in range(1, 21)
    ]

    gold_df = gold_df.fillna(
        0,
        subset=clickstream_feature_names
    )


    # ========================================================
    # 7. REMOVE NON-MODEL / REDUNDANT RAW FIELDS
    # ========================================================

    gold_df = gold_df.drop(
        "Type_of_Loan",
        "Credit_History_Age",
        "Balance_to_Income_Ratio"
    )


    # ========================================================
    # 8. HANDLE REMAINING MISSING VALUES
    # ========================================================

    for feature in ["Age", "Changed_Credit_Limit"]:

        median_value = gold_df.approxQuantile(
            feature,
            [0.5],
            0.01
        )[0]

        gold_df = gold_df.fillna({
            feature: median_value
        })


    # ========================================================
    # 9. SAVE GOLD FEATURE STORE
    # ========================================================

    output_path = (
        gold_feature_store_directory
        + "gold_feature_store.parquet"
    )

    gold_df.write.mode("overwrite").parquet(
        output_path
    )

    print(
        "Gold feature store row count:",
        gold_df.count()
    )

    print(
        "Gold feature store columns:",
        len(gold_df.columns)
    )

    print(
        "saved Gold feature store to:",
        output_path
    )

    return gold_df