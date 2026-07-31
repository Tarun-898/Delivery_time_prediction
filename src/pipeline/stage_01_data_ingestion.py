import sys
import os

sys.path.append(os.getcwd())


from src.components.ingestion import DataIngestion

if __name__ == "__main__":

    obj = DataIngestion()

    train_path, test_path = obj.initiate_data_ingestion()

    print(train_path)
    print(test_path)