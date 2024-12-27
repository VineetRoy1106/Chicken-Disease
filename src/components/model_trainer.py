from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model, save_obj



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import logging
import sys
from src.exception import CustomException
from src.utils import save_obj

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Random Forest': RandomForestRegressor(),
                'XGB Regressor': XGBRegressor()
            }

            # Assuming evaluate_model function is defined elsewhere
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # Get best model based on R2 score
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            # Save the best model to file
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.exception('Exception occurred at Model Training')
            raise CustomException(e, sys)
