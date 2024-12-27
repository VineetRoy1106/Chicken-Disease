from src.constants import *
import os

ROOT_DIR = ROOT_DIR_KEY

DATASET_PATH = os.path.join(ROOT_DIR, DATA_DIR, DATA_DIR_KEY)

RAW_DATA_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, 
                                DATA_INGESTION_RAW_DATA_DIR_KEY, RAW_DATA_DIR_KEY)

TRAIN_DATA_PATH = os.path.join(ROOT_DIR, 
                               ARTIFACT_DIR_KEY,
                               DATA_INGESTION_KEY, 
                               DATA_INGESTION_INGESTED_DATA_DIR_KEY , 
                               TRAIN_DATA_DIR_KEY)

TEST_DATA_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY,
                               DATA_INGESTION_INGESTED_DATA_DIR_KEY, TEST_DATA_DIR_KEY)




# DATA TRANSFORMATION RELATED PATHS

# PREPROCESSOR_OBJ_FILE = os.path.join(ROOT_DIR_KEY,
#                                         ARTIFACT_DIR_KEY,
#                                         DATA_TRANSFORMATION_KEY,
#                                         DATA_PREPROCESSOR_DIR_KEY,
#                                         DATA_TRANSFORMED_DIR_KEY)


PREPROCESSOR_OBJ_FILE = os.path.join(ROOT_DIR_KEY,
                                        ARTIFACT_DIR_KEY,
                                        DATA_TRANSFORMATION_KEY,
                                        DATA_PREPROCESSOR_DIR_KEY,
                                        DATA_TRANSFORMATION_PROCESSOR_KEY
                                        )

TRANSFORM_TRAIN_FILE_PATH = os.path.join(ROOT_DIR_KEY,
                                        ARTIFACT_DIR_KEY,
                                        DATA_TRANSFORMATION_KEY,
                                        DATA_TRANSFORMED_DIR_KEY,
                                        DATA_TRANSFORMED_TRAIN_DIR_KEY)

TRANSFORM_TEST_FILE_PATH = os.path.join(ROOT_DIR_KEY,
                                        ARTIFACT_DIR_KEY,
                                        DATA_TRANSFORMATION_KEY,
                                        DATA_TRANSFORMED_DIR_KEY,
                                        DATA_TRANSFORMED_TEST_DIR_KEY)

FEATURE_ENGINEERING_FILE_PATH = os.path.join(ROOT_DIR_KEY,
                                        ARTIFACT_DIR_KEY,
                                        DATA_TRANSFORMATION_KEY,
                                        DATA_PREPROCESSOR_DIR_KEY,
                                        "feaure_engg.pkl")


MODEL_FILE_PATH = os.path.join(ROOT_DIR_KEY,
                                        ARTIFACT_DIR_KEY,
                                        MODEL_TRAINER_KEY,
                                        MODEL_OBJECT,
                                        )