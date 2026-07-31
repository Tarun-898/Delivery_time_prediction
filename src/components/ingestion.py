import os,sys
import pandas as pd
import numpy as np
from src.constant import *
from src.config.configuration import *
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path=TRAIN_FILE_PATH
    test_data_path=TEST_FILE_PATH
    raw_data_path=RAW_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading dataset")
            df=pd.read_csv(DATASET_PATH)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("dataset loaded")

            logging.info("dataset splitting in test and train")
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("dataset splitted sucessfully in test and train")
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

