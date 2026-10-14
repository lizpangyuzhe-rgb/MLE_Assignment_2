from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="assignment2_ml_pipeline",
    description="Assignment 2 end-to-end ML pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,          # manual trigger for now
    catchup=False,
    tags=["MLE", "Assignment2"],
) as dag:

    start = EmptyOperator(
        task_id="start"
    )

    check_project_files = BashOperator(
        task_id="check_project_files",
        bash_command="""
        echo "Checking mounted Assignment 2 project..."
        cd /opt/airflow/project
        pwd
        ls -la
        """
    )

    end = EmptyOperator(
        task_id="end"
    )

    start >> check_project_files >> end