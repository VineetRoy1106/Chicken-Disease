from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from src.pipeline.training_pipeline import Train
import numpy as np

training_pipeline = Train()

default_args = {"retries":2}

with DAG(
    default_args=default_args,
    dag_id="training_pipeline",
    start_date=pendulum.datetime(2024, 1, 1),
    description = "This DAG is used to train the model for delivery time prediction" ,
    schedule_interval="@daily",
    catchup=False,
    tags=["Delivery Time Prediction","Machine Learning","Training"],
) as dag:

    dag.doc_md = __doc__

    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"] # Task Instance
        train_data, test_data = training_pipeline.initiate_data_ingestion()
        ti.xcom_push("Data Ingestion Artifact", {"train_data": train_data, "test_data": test_data})

    def data_transformation(**kwargs):
        ti = kwargs["ti"] # Task Instance
        data_ingestion_artifact = ti.xcom_pull(key="Data Ingestion Artifact", task_ids="data_ingestion")
        train_arr,test_arr,engg_pkl,transformation_pkl = training_pipeline.initiate_data_transformation(data_ingestion_artifact["train_data"])
        train_arr = train_arr.to_list()
        test_arr = test_arr.to_list()
        ti.xcom_push("Data Transformation Artifact", {"train_arr": train_arr, "test_arr": test_arr })

    def model_training(**kwargs):
        ti = kwargs["ti"] # Task Instance
        data_transformation_artifact = ti.xcom_pull(key="Data Transformation Artifact", task_ids="data_transformation")
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        training_pipeline.initiate_model_training(train_arr, test_arr)
        
        # model = training_pipeline.initiate_model_training(data_transformation_artifact

                                                      
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
        #### Task Documentation
        This task is used to ingest the data from the source
        """
    )

    data_transformation_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation,

    )
    data_transformation_task.doc_md = dedent(
        """\
        #### Task Documentation
        This task is used to transform the data
        """
    )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=model_training,

    )
    model_training_task.doc_md = dedent(
        """\
        #### Task Documentation
        This task is used to train the model
        """
    )

    data_ingestion_task >> data_transformation_task >> model_training_task


