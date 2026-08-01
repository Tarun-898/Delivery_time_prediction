import sys
import os

sys.path.append(os.getcwd())


from src.components.data_Transformation import DataTransformation
from src.components.ingestion import DataIngestion


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    transformation.initate_data_transformation(train_path, test_path)
