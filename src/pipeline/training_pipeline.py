from src.constants import *
from src.config.configuration import *
import os, sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.data_ingestion import DataIngestion

class Train:
    def __init__(self):
        self.c = 0
        print(f"***********{self.c}***********")


    def main(self):
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        data_transform = DataTransformation()
        train_arr,test_arr,engg_pkl,transformation_pkl = data_transform.initiate_data_transformation(train_data, test_data)
        model_trainer = ModelTrainer()
        print(model_trainer.initate_model_training(train_arr,test_arr))