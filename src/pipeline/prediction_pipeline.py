import os,sys
import pandas as pd
import pickle
from dataclasses import dataclass
from src.logger import logging
from src.config.configuration import *
from src.exception import CustomException
from src.utils import load_model


class PredictionPipeline:
    def __init__(self):
        pass

    def start_predict(self,features):
        try:
            
            preprocesor_path = PREPROCESSOR_OBJ_FILE
            model_path = MODEL_FILE_PATH 

            preprocessor = load_model(file_path=preprocesor_path)
            model = load_model(file_path=model_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
        

        except Exception as e:
            raise CustomException(e,sys) from e
        

class CustomData:
    def __init__(self,
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                #  Delivery_Distance:float,
                 Road_traffic_density:str,
                 Vehicle_condition:str,
                 distance:float,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 Festival:str,
                 City:str,
                 multiple_deliveries:int

       ):   
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        # self.Delivery_Distance = Delivery_Distance
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.distance = distance
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        self.multiple_deliveries = multiple_deliveries

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Weather_conditions': [self.Weather_conditions],
                # 'Delivery_Distance': [self.Delivery_Distance],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                'distance': [self.distance],
                'Type_of_order': [self.Type_of_order],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'Festival': [self.Festival],
                'City': [self.City],
                'multiple_deliveries': [self.multiple_deliveries]
            }

            df = pd.DataFrame(custom_data_input_dict)

            return df
        except Exception as e:
            logging.info("Error occured in Custom pipeline Dataframe")
            raise CustomException(e,sys) from e
        
