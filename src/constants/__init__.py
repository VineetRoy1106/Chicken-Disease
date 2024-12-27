import os, sys
from datetime import datetime

# artifact -> pipelien folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "Data"
DATA_DIR_KEY = "finalTrain.csv"

#new_machine_learning/DATA_DIR/DATA_DIR_KEY

#ROOT_DIR_KEY / ARTIFACT_DIR_KEY / DATA_INGESTION_KEY / DATA_INGESTION_INGESTED_DATA_DIR_KEY / TRAIN_DATA_DIR_KEY

ARTIFACT_DIR_KEY = "Artifacts"

# Data Ingestion Related variables

DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_dir"
# DATA_INGESTION_TRAIN_DATA_KEY = "train_data"
# DATA_INGESTION_TEST_DATA_KEY = "test_data"

RAW_DATA_DIR_KEY = "raw_data.csv"
TRAIN_DATA_DIR_KEY = "train_data.csv"
TEST_DATA_DIR_KEY = "test_data.csv"



# os.path.join(ROOT_DIR_KEY,DATA_DIR,DATA_INGESTION_INGESTED_DATA_DIR_KEY,TRAIN_DATA_DIR_KEY )


# Data Transformation Related variables

# artifact > Data Transformation > 
            #  Processor> preprocessor.pkl
            #  transformation > train.csv and test.csv

DATA_TRANSFORMATION_KEY = "data_transformation"
DATA_PREPROCESSOR_DIR_KEY = "processor"
DATA_TRANSFORMATION_PROCESSOR_KEY = "preprocessor.pkl"
DATA_TRANSFORMED_DIR_KEY = "transformation"

DATA_TRANSFORMED_TRAIN_DIR_KEY = "train.csv"
DATA_TRANSFORMED_TEST_DIR_KEY = "test.csv"

# PREPROCESSOR_OBJ_FILE = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_PREPROCESSOR_DIR_KEY,
#                                         DATA_TRANSFORMATION_PROCESSOR_KEY)

# PREPROCESSOR_OBJ_FILE = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_PREPROCESSOR_DIR_KEY,
#                                         "feaure_engg.pkl")

# TRANSFORM_TRAIN_FILE_PATH = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_TRANSFORMED_DIR_KEY,
#                                         DATA_TRANSFORMED_TRAIN_DIR_KEY)

# TRANSFORM_TEST_FILE_PATH = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_TRANSFORMED_DIR_KEY,
#                                         DATA_TRANSFORMED_TEST_DIR_KEY)

# FEATURE_ENGINEERING_FILE_PATH = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_PREPROCESSOR_DIR_KEY,
#                                         DATA_TRANSFORMATION_PROCESSOR_KEY)


MODEL_TRAINER_KEY ="model_trainer"
MODEL_OBJECT = "model.pkl"
