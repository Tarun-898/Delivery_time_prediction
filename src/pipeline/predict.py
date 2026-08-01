import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj
from src.logger import logging
from src.config.configuration import *


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # model_path='artifact\model_trainer_dir\model.pkl'
            # processor_path='artifact\data_transformation\2026-07-29-12-47-26\processor_dir\processor.pkl'
            # feature_path='artifact\data_transformation\2026-07-29-12-47-26\processor_dir\feature_engi.pkl'

            model_path = MODEL_FILE_PATH
            processor_path = PREPROCESSING_FILE
            feature_path = FEATURE_ENGI_FILE

            model = load_obj(file_path=model_path)
            feature = load_obj(file_path=feature_path)
            processor = load_obj(file_path=processor_path)

            fe_data = feature.transform(features)
            data_scaled = processor.transform(fe_data)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        Delivery_person_Age: int,
        Delivery_person_Ratings: float,
        Vehicle_condition: int,
        multiple_deliveries: int,
        Restaurant_latitude: float,
        Restaurant_longitude: float,
        Delivery_location_latitude: float,
        Delivery_location_longitude: float,
        Type_of_order: str,
        Type_of_vehicle: str,
        Festival: str,
        City: str,
        Road_traffic_density: str,
        Weather_conditions: str,
    ):

        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
        self.Road_traffic_density = Road_traffic_density
        self.Weather_conditions = Weather_conditions

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Vehicle_condition": [self.Vehicle_condition],
                "multiple_deliveries": [self.multiple_deliveries],
                "Restaurant_latitude": [self.Restaurant_latitude],
                "Restaurant_longitude": [self.Restaurant_longitude],
                "Delivery_location_latitude": [self.Delivery_location_latitude],
                "Delivery_location_longitude": [self.Delivery_location_longitude],
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "City": [self.City],
                "Road_traffic_density": [self.Road_traffic_density],
                "Weather_conditions": [self.Weather_conditions],
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info("Custom DataFrame Created Successfully")

            return df

        except Exception as e:
            raise CustomException(e, sys)
