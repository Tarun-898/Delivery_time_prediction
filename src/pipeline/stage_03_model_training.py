import sys
import os

sys.path.append(os.getcwd())

import dagshub
import mlflow

from src.components.model_trainer import ModelTrainer
from src.components.ingestion import DataIngestion
from src.components.data_Transformation import DataTransformation


dagshub.init(repo_owner="Tarun-898", repo_name="P01", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Tarun-898/P01.mlflow")
mlflow.set_experiment("Delivery_Time_Prediction")


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    train_arr, test_arr, _ = transformation.initate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()

    trainer.initiate_model_training(train_arr, test_arr)
