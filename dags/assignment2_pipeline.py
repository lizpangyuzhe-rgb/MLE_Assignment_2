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
    "retry_delay": timedelta(minutes=5),
}


# ========================================================
# 2. TASK HELPER
# ========================================================

def create_python_task(task_id, script_name):

    return BashOperator(
        task_id=task_id,

        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"python -u scripts/{script_name}"
        ),

        do_xcom_push=False,

        execution_timeout=timedelta(minutes=90),
    )


# ========================================================
# 3. DEFINE DAG
# ========================================================

with DAG(

    dag_id="assignment2_ml_pipeline",

    description=(
        "End-to-end loan default prediction ML pipeline"
    ),

    default_args=default_args,

    # Initial testing: manual trigger
    schedule_interval=None,

    start_date=datetime(2026, 10, 1),

    catchup=False,

    max_active_runs=1,

    tags=[
        "MLE",
        "Assignment2",
        "LoanDefault"
    ],

) as dag:

    # ====================================================
    # DATA PIPELINE
    # ====================================================

    bronze = create_python_task(
        "bronze_ingestion",
        "data_bronze.py"
    )

    silver = create_python_task(
        "silver_processing",
        "data_silver.py"
    )

    gold = create_python_task(
        "build_gold_feature_label",
        "data_gold.py"
    )

    # ====================================================
    # MODEL DEVELOPMENT
    # ====================================================

    training = create_python_task(
        "train_xgboost",
        "model_training.py"
    )

    quality_gate = create_python_task(
        "model_quality_gate",
        "model_quality_gate.py"
    )

    # ====================================================
    # MODEL DEPLOYMENT
    # ====================================================

    inference = create_python_task(
        "batch_inference",
        "model_inference.py"
    )

    # ====================================================
    # MODEL MONITORING
    # ====================================================

    monitoring = create_python_task(
        "model_monitoring",
        "model_monitoring.py"
    )

    # ====================================================
    # TASK DEPENDENCIES
    # ====================================================

    (
        bronze
        >> silver
        >> gold
        >> training
        >> quality_gate
        >> inference
        >> monitoring
    )