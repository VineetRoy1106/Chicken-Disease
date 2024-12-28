from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
import pickle
from src.utils import load_model

from sklearn.pipeline import Pipeline

PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction_csv"
PREDICCTION_FILE = "prediction.csv"
FEATURE_ENGG_FOLDER = "feature_engg"

ROOT_DIR = os.getcwd()
BATCH_PREDICTION = os.path.join(ROOT_DIR,PREDICTION_FOLDER, PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR,PREDICTION_FOLDER, FEATURE_ENGG_FOLDER)

class BatchPredictionConfig:
    def __init__(self,
                 input_file_path,
                 model_file_path,
                 transformation_file_path,
                 feature_engg_file_path,
                 ):
        self.prediction_csv = BATCH_PREDICTION
        self.prediction_file = PREDICCTION_FILE
        self.feature_engg_folder = FEATURE_ENGG_FOLDER

        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformation_file_path = transformation_file_path
        self.feature_engg_file_path = feature_engg_file_path


    def start_batch_prediction(self):
        try:
            
            # with open(self.feature_engg_file_path, 'rb') as f:
            #     feature_engg = pickle.load(f)

            # with open(self.transformation_file_path, 'rb') as f:
            #     preprocessor = pickle.load(f)

            feature_engg = load_model(file_path=self.feature_engg_file_path)

            preprocessor = load_model(file_path=self.transformation_file_path)

            model = load_model(file_path=self.model_file_path)

            # create feature_engineering pipeline

            feature_engg_pipeline = Pipeline([
                ('feature_engineering', feature_engg)
            ])


            df = pd.read_csv(self.input_file_path)

            #apply feature engineering pipeline

            df = feature_engg_pipeline.transform(df)

            df.to_csv("df_delivery.csv")

            FEATURE_ENGGINERING_PATH = FEATURE_ENG

            os.makedirs(FEATURE_ENGGINERING_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENGGINERING_PATH, "df_delivery.csv")
            df.to_csv(file_path, index=False)

            # Time Taken
            df = df.drop("Time_taken (min)", axis=1)
            df.to_csv("time_taken_drop.csv")

            # # create transformation pipeline
            # transformation_pipeline = Pipeline([
            #     ('preprocessor', preprocessor)
            # ])

            transform_data = preprocessor.transform(df)

            logging.info(f"Transformed Data Shape: {transform_data.shape}")
            
            logging.info(f"Loaded numpy from batch prediciton :{transform_data}")

            file_path = os.path.join(FEATURE_ENGGINERING_PATH, "processor.csv")

            prediction = model.predict(transform_data )

            df_predciction = pd.DataFrame(prediction, columns=["prediction"])

            BATCH_PREDICTION_PATH = BATCH_PREDICTION

            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, "output.csv")
            
            df_predciction.to_csv(csv_path, index=False)
            logging.info("Batch Prediction Completed")









        except Exception as e:
            raise CustomException(e, sys)
    
    