from src.constants import *
from src.config.configuration import *
import os, sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import *
 

class Feature_Engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("******************feature Engineering started******************")

    def distance_numpy(self, df, lat1, lon1,lat2, lon2 ):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12734 * np.arccos(np.sort(a))

    def transform_data(self,df):
        try:
            df.drop(['ID'], axis=1, inplace=True)


            self.distance_numpy(df, "Restaurant_latitude",
                                "Restaurant_longitude",
                                "Delivery_location_latitude",
                                "Delivery_location_longitude"
                                )
            
            df.drop(["Restaurant_latitude","Restaurant_longitude",
                    "Delivery_location_latitude","Delivery_location_longitude"], axis=1, inplace=True)
            
            logging.info("dropping columns from our original dataset")


            return df
        


        except Exception as e:
            CustomException(e, sys)

    def fit(self, x, y=None):
            return self

    def transform(self, x: pd.DataFrame, y=None):
        try:

            transformed_df = self.transform_data(x)
            return transformed_df
                    
        except Exception as e:
            CustomException(e, sys)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = PREPROCESSOR_OBJ_FILE
    transform_train_file_path = TRANSFORM_TRAIN_FILE_PATH   
    transform_test_file_path = TRANSFORM_TEST_FILE_PATH
    feature_engineering_file_path = FEATURE_ENGINEERING_FILE_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # self.feature_engineering = Feature_Engineering()

    def get_data_transformation_obj(self):
        try:
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density', 'Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition',
                              'multiple_deliveries','distance']

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # ordinal Pipeline
            ordinal_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])


            preprocssor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline,numerical_column ),
                ('categorical_pipeline', categorical_pipeline,categorical_columns ),
                ('ordinal_pipeline', ordinal_pipeline,ordinal_encoder )
            ])

            logging.info("Pipeline Steps Completed")
            return preprocssor

        except Exception as e:
            raise CustomException( e,sys)
        
        

    def get_feature_engineering_object(self):
        
        try:
            feature_engineering = Pipeline(steps=[
                ("fe" , Feature_Engineering())
            ])
        
            return feature_engineering
        
        
        except Exception as e:
            raise CustomException( e,sys)
        
    
    # def initiate_data_transformation(self,train_path,test_path):

    #     try:

    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)

    #         logging.info("obtaining feature engineering object")

    #         fe_obj = self.get_feature_engineering_object()

    #         train_df = fe_obj.fit_transform(train_df)
    #         test_df = fe_obj.transform(test_df)

    #         train_df.to_csv("train.csv")
    #         test_df.to_csv("test.csv")


    #         processing_obj = self.get_data_transformation_obj()

    #         target_column = "Time_taken (min)"

    #         x_train = train_df.drop(target_column,axis = 1)
    #         y_train = train_df[target_column]

    #         x_test = train_df.drop(target_column,axis = 1)
    #         y_test = train_df[target_column]

    #         x_train = processing_obj.fit_transform(x_train)
    #         x_test = processing_obj.transform(x_test)

    #         train_arr = np.c_[x_train,np.array(y_train) ]
    #         test_arr = np.c_[x_test,np.array(y_test) ]


    #         df_train = pd.DataFrame(train_arr)
    #         df_test = pd.DataFrame(test_arr)

    #         os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_file_path), exist_ok=True )
    #         df_train.to_csv(self.data_transformation_config.transform_test_file_path, index=False , header=True)

    #         os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_file_path), exist_ok=True )
    #         df_test.to_csv(self.data_transformation_config.transform_test_file_path, index=False , header=True) 


    #         save_obj(file_path=self.data_transformation_config.feature_engineering_file_path,
    #                  obj=fe_obj)
    #         save_obj(file_path=self.data_transformation_config.preprocessor_obj_file,
    #                  obj=fe_obj)

    #         return (
    #             train_arr,
    #             test_arr,
    #             self.data_transformation_config.feature_engineering_file_path,
    #             self.data_transformation_config.feature_engineering_file_path
    #         )

    #     except Exception  as e:
    #         raise CustomException( e,sys)


    # def initiate_data_transformation(self, train_path, test_path):
    #     try:
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)

    #         logging.info("Obtaining feature engineering object...")
    #         fe_obj = self.get_feature_engineering_object()

    #         train_df = fe_obj.fit_transform(train_df)
    #         test_df = fe_obj.transform(test_df)

    #         logging.info("Applying preprocessing pipeline...")
    #         processing_obj = self.get_data_transformation_obj()

    #         target_column = "Time_taken (min)"
    #         x_train = train_df.drop(target_column, axis=1)
    #         y_train = train_df[target_column]

    #         x_test = test_df.drop(target_column, axis=1)
    #         y_test = test_df[target_column]

    #         x_train = processing_obj.fit_transform(x_train)
    #         x_test = processing_obj.transform(x_test)

    #         train_arr = np.c_[x_train, np.array(y_train)]
    #         test_arr = np.c_[x_test, np.array(y_test)]

    #         os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_file_path), exist_ok=True)
    #         pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transform_train_file_path, index=False, header=True)

    #         os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_file_path), exist_ok=True)
    #         pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transform_test_file_path, index=False, header=True)

    #         save_obj(file_path=self.data_transformation_config.feature_engineering_file_path, obj=fe_obj)
    #         save_obj(file_path=self.data_transformation_config.preprocessor_obj_file, obj=processing_obj)

    #         return (train_arr, test_arr, self.data_transformation_config.feature_engineering_file_path, 
    #                 self.data_transformation_config.preprocessor_obj_file)

    #     except Exception as e:
    #         raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining feature engineering object...")
            fe_obj = self.get_feature_engineering_object()

            # Feature engineering
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)

            logging.info("Applying preprocessing pipeline...")
            processing_obj = self.get_data_transformation_obj()

            # Separate target column
            target_column = "Time_taken (min)"
            x_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            x_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            # Preprocessing
            x_train = processing_obj.fit_transform(x_train)
            x_test = processing_obj.transform(x_test)

            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]

            # Save results
            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_file_path), exist_ok=True)
            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transform_train_file_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_file_path), exist_ok=True)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transform_test_file_path, index=False, header=True)

            # Save transformers
            save_obj(file_path=self.data_transformation_config.feature_engineering_file_path, obj=fe_obj)
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file, obj=processing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.feature_engineering_file_path,
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e, sys)

