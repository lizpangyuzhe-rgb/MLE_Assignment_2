from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta


# ========================================================
# 1. DAG CONFIGURATION
# ========================================================

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "MLE_Assignment_2",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ========================================================
# 2. MONTHLY BACKFILL DAG
# ========================================================

with DAG(

    dag_id="assignment2_monthly_backfill",

    description="Monthly historical data backfill",

    default_args=default_args,

    # Run on the first day of each month
    schedule_interval="0 0 1 * *",

    # Limit demonstration to three historical months
    start_date=datetime(2024, 8, 1),
    end_date=datetime(2024, 10, 1),

    # Automatically create missed historical runs
    catchup=True,

    # Avoid overlapping historical runs
    max_active_runs=1,

    # Keep paused until we are ready to test
    is_paused_upon_creation=True,

    tags=["MLE", "Backfill"],

) as dag:

    # ====================================================
    # BRONZE
    # ====================================================

    bronze = BashOperator(

        task_id="backfill_bronze",

        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -u scripts/data_bronze.py "
            '--snapshotdate "{{ ds }}"'
        ),

        do_xcom_push=False,
    )

    # ====================================================
    # SILVER
    # ====================================================

    silver = BashOperator(

        task_id="backfill_silver",

        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -u scripts/data_silver.py "
            '--snapshotdate "{{ ds }}"'
        ),

        do_xcom_push=False,
    )

    # ====================================================
    # DEPENDENCIES
    # ====================================================

    bronze >> silver