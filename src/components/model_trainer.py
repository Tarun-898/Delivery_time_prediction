# # import os,sys
# # import pandas as pd
# # import numpy as np
# # from src.constant import *
# # from src.config.configuration import *
# # from dataclasses import dataclass
# # from src.logger import logging
# # from src.exception import CustomException
# # from sklearn.base import BaseEstimator,TransformerMixin
# # from sklearn.svm import SVR
# # from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
# # from sklearn.tree import DecisionTreeRegressor
# # from xgboost import XGBRegressor
# # from src.utils import save_obj,evalute_model

# # @dataclass
# # class ModelTrainerConfig:
# #     trained_model_file_path=MODEL_FILE_PATH

# # class ModelTrainer:
# #         def __init__(self):
# #             self.model_tariner_config=ModelTrainerConfig()

# #         def initiate_model_training(self,train_arr,test_arr):
# #             try:
# #                 X_train,y_train,X_test,y_test=(train_arr[ :, :-1],train_arr[:, -1],test_arr[:, :-1],test_arr[:, -1])

# #                 models={

# #                     "XGBRegressor":XGBRegressor(),
# #                     "DecisionTreeRegressor":DecisionTreeRegressor(),
# #                     "GradientBoostingRegressor":GradientBoostingRegressor(),
# #                     "RandomForestRegressor":RandomForestRegressor(),
# #                     "SVR":SVR()

# #                 }

# #                 model_report:dict=evalute_model(X_train,y_train,X_test,y_test,models)
# #                 print(model_report)

              

# #                 best_model_name = max(model_report, key=model_report.get)
# #                 best_model_score = model_report[best_model_name]

# #                 best_model=models[best_model_name]
                

# #                 if best_model_score<0.6:
# #                      logging.info("no best model found")
# #                      raise CustomException("no best model found",sys)

# #                 best_model.fit(X_train, y_train)
                    
                
# #                 print(f"best model name: {best_model_name}, with score{best_model_score}")
# #                 logging.info(f"best model name: {best_model_name}, with score{best_model_score}")

# #                 save_obj(file_path=self.model_tariner_config.trained_model_file_path,obj=best_model)
# #             except Exception as e:
# #                 raise CustomException(e,sys)




# import os
# import sys
# from dataclasses import dataclass

# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import r2_score
# from xgboost import XGBRegressor

# from src.constant import *
# from src.config.configuration import *
# from src.logger import logging
# from src.exception import CustomException
# from src.utils import save_obj


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = MODEL_FILE_PATH


# class ModelTrainer:

#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_training(self, train_arr, test_arr):

#         try:

#             X_train = train_arr[:, :-1]
#             y_train = train_arr[:, -1]

#             X_test = test_arr[:, :-1]
#             y_test = test_arr[:, -1]

#             models = {

#                 "XGBRegressor": XGBRegressor(),

#                 "DecisionTreeRegressor": DecisionTreeRegressor(),

#                 "GradientBoostingRegressor": GradientBoostingRegressor(),

#                 "RandomForestRegressor": RandomForestRegressor(),

#                 "SVR": SVR()

#             }

#             params = {

#                 "DecisionTreeRegressor": {

#                     "criterion": ["squared_error", "friedman_mse"],
#                     "max_depth": [3, 5, 10, 20, None],
#                     "min_samples_split": [2, 5, 10]

#                 },

#                 "RandomForestRegressor": {

#                     "n_estimators": [100, 200, 300],
#                     "max_depth": [5, 10, 20, None],
#                     "min_samples_split": [2, 5, 10]

#                 },

#                 "GradientBoostingRegressor": {

#                     "n_estimators": [100, 200],
#                     "learning_rate": [0.01, 0.05, 0.1],
#                     "max_depth": [3, 5, 7]

#                 },

#                 "XGBRegressor": {

#                     "n_estimators": [100, 200],
#                     "learning_rate": [0.01, 0.05, 0.1],
#                     "max_depth": [3, 5, 7]

#                 },

#                 "SVR": {

#                     "kernel": ["linear", "rbf"],
#                     "C": [0.1, 1, 5],
#                     "gamma": ["scale", "auto"]

#                 }

#             }

#             model_report = {}

#             best_models = {}

#             for model_name, model in models.items():

#                 logging.info(f"Hyperparameter tuning started for {model_name}")

#                 random_search = RandomizedSearchCV(

#                     estimator=model,

#                     param_distributions=params[model_name],

#                     n_iter=10,

#                     cv=5,

#                     scoring="r2",

#                     random_state=42,

#                     n_jobs=-1

#                 )

#                 random_search.fit(X_train, y_train)

#                 best_model = random_search.best_estimator_

#                 best_models[model_name] = best_model

#                 y_pred = best_model.predict(X_test)

#                 score = r2_score(y_test, y_pred)

#                 model_report[model_name] = score

#                 logging.info(f"{model_name} Score : {score}")

#                 logging.info(f"Best Parameters : {random_search.best_params_}")

#             print(model_report)

#             best_model_name = max(model_report, key=model_report.get)

#             best_model_score = model_report[best_model_name]

#             best_model = best_models[best_model_name]

#             if best_model_score < 0.6:

#                 logging.info("No Best Model Found")

#                 raise CustomException("No Best Model Found", sys)

#             logging.info(
#                 f"Best Model : {best_model_name} Score : {best_model_score}"
#             )

#             print(
#                 f"\nBest Model : {best_model_name}\nScore : {best_model_score}"
#             )

#             save_obj(

#                 file_path=self.model_trainer_config.trained_model_file_path,

#                 obj=best_model

#             )

#             return best_model_score

#         except Exception as e:

#             raise CustomException(e, sys)

import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from dataclasses import dataclass

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

from src.constant import *
from src.config.configuration import *
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):

        try:

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {

                "XGBRegressor": XGBRegressor(),

                "DecisionTreeRegressor": DecisionTreeRegressor(),

                "GradientBoostingRegressor": GradientBoostingRegressor(),

                "RandomForestRegressor": RandomForestRegressor(),

                "SVR": SVR()

            }

            params = {

                "DecisionTreeRegressor": {

                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [3, 5, 10, 20, None],
                    "min_samples_split": [2, 5, 10]

                },

                "RandomForestRegressor": {

                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10]

                },

                "GradientBoostingRegressor": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7]

                },

                "XGBRegressor": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7]

                },

                "SVR": {

                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 5],
                    "gamma": ["scale", "auto"]

                }

            }

            model_report = {}

            best_models = {}

            best_params = {}

            for model_name, model in models.items():

                logging.info(
                    f"Hyperparameter tuning started for {model_name}"
                )

                random_search = RandomizedSearchCV(

                    estimator=model,

                    param_distributions=params[model_name],

                    n_iter=10,

                    cv=5,

                    scoring="r2",

                    random_state=42,

                    n_jobs=-1

                )

                random_search.fit(X_train, y_train)

                best_model = random_search.best_estimator_

                best_models[model_name] = best_model

                best_params[model_name] = random_search.best_params_

                y_pred = best_model.predict(X_test)

                score = r2_score(y_test, y_pred)

                model_report[model_name] = score

                logging.info(f"{model_name} Score : {score}")

                logging.info(
                    f"Best Parameters : {random_search.best_params_}"
                )

            print(model_report)

            best_model_name = max(model_report, key=model_report.get)

            best_model_score = model_report[best_model_name]

            best_model = best_models[best_model_name]

            if best_model_score < 0.6:

                logging.info("No Best Model Found")

                raise CustomException(
                    "No Best Model Found",
                    sys
                )
            logging.info(
                f"Best Model : {best_model_name} | Score : {best_model_score}"
            )

            print(
                f"\nBest Model : {best_model_name}"
                f"\nScore : {best_model_score}"
            )
            save_obj(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

                )
            logging.info("Model Saved Successfully")

            # ===========================
            # MLflow Run Starts Here
            # ===========================
            
            with mlflow.start_run(run_name=best_model_name):
                mlflow.set_tag("developer", "Tarun")
                mlflow.set_tag("project", "Delivery Time Prediction")
                

                # Prediction
                y_pred = best_model.predict(X_test)

                # Metrics
                r2 = r2_score(y_test, y_pred)

                mae = mean_absolute_error(
                    y_test,
                    y_pred
                )

                mse = mean_squared_error(
                    y_test,
                    y_pred
                )

                rmse = np.sqrt(mse)

                # -------------------------
                # Log Parameters
                # -------------------------

                mlflow.log_param(
                    "Model",
                    best_model_name
                )

                mlflow.log_params(
                best_model.get_params()
                )

                # -------------------------
                # Log Metrics
                # -------------------------

                mlflow.log_metric(
                    "R2 Score",
                    r2
                )

                mlflow.log_metric(
                    "MAE",
                    mae
                )

                mlflow.log_metric(
                    "MSE",
                    mse
                )

                mlflow.log_metric(
                    "RMSE",
                    rmse
                )

                print("\n========== MLflow Metrics ==========")

                print(f"R2 Score : {r2}")

                print(f"MAE : {mae}")

                print(f"MSE : {mse}")

                print(f"RMSE : {rmse}")

                logging.info("Metrics Logged Successfully")

                logging.info("Parameters Logged Successfully")
                               
                # Log Complete Model
  
                if best_model_name == "XGBRegressor":

                    mlflow.xgboost.log_model(
                        xgb_model=best_model,
                        artifact_path="model"
                    )

                else:

                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model"
                    )

                logging.info("Model Logged Successfully")

                                



                

                # -------------------------
                # Log Saved Model Artifact
                # -------------------------

                mlflow.log_artifact(

                    self.model_trainer_config.trained_model_file_path

                )

                logging.info("Artifact Logged Successfully")

            print("\n========== MLflow Run Completed ==========")

            print(f"Best Model : {best_model_name}")

            print(f"R2 Score : {r2}")

            print(f"Model Saved At : {self.model_trainer_config.trained_model_file_path}")

            return best_model_score

        except Exception as e:

            logging.exception(e)

            raise CustomException(e, sys)