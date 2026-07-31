import os,sys
import pandas as pd
import numpy as np
from src.constant import *
from src.config.configuration import *
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        logging.info("feature engi. started")

    def distance_numpy(self,df,lat1,lon1,lat2,lon2):
        p=np.pi/180
        # a=0.5-np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p)*np.cos(df[lat2]*p)*(1-np.cos((df[lon2]-df[lon1])*p))
        a = (
            0.5
            - np.cos((df[lat2]-df[lat1]) * p) / 2
            + np.cos(df[lat1] * p)
            * np.cos(df[lat2] * p)
            * (1 - np.cos((df[lon2]-df[lon1]) * p))
            / 2
            )
        df['distance']=12734*np.arccos(a)

    # def transform_data(self,df):
    #     try:
    #         # df.drop(['ID'],axis=1,inplace=True)
    #         df.drop(['ID'], axis=1, inplace=True, errors='ignore')

    #         self.distance_numpy(df, 'Restaurant_latitude', 'Restaurant_longitude', 
    #                             'Delivery_location_latitude', 'Delivery_location_longitude') 
    #         df.drop(['Delivery_person_ID', 'Restaurant_latitude','Restaurant_longitude', 
    #                  'Delivery_location_latitude', 'Delivery_location_longitude', 
    #                  'Order_Date','Time_Orderd','Time_Order_picked'], axis=1,inplace=True,errors='ignore') 
 
    #         logging.info("droping columns from our original dataset") 
    #         return df
    #     except Exception as e:
    #         raise CustomException(e,sys)




    def transform_data(self, df):
            
        try:

            # Create Distance Feature
            if (
                'Restaurant_latitude' in df.columns and
                'Restaurant_longitude' in df.columns and
                'Delivery_location_latitude' in df.columns and
                'Delivery_location_longitude' in df.columns
            ):

                self.distance_numpy(
                    df,
                    'Restaurant_latitude',
                    'Restaurant_longitude',
                    'Delivery_location_latitude',
                    'Delivery_location_longitude'
                )

          
            columns_to_drop = [
                'ID',
                'Delivery_person_ID',
                'Restaurant_latitude',
                'Restaurant_longitude',
                'Delivery_location_latitude',
                'Delivery_location_longitude',
                'Order_Date',
                'Time_Orderd',
                'Time_Order_picked'
            ]

            df.drop(
                columns=[col for col in columns_to_drop if col in df.columns],
                inplace=True
            )

            logging.info("Feature Engineering Completed Successfully")

            return df

        except Exception as e:

            raise CustomException(e, sys)
                
    # def fit(self, X, y=None):
    #     return self

    def fit(self, X, y=None):
     logging.info("Feature engineering fitted")
     return self
    
    def transform(self,X: pd.DataFrame,y=None):
        try:
            transform_df=self.transform_data(X)
            return transform_df
        
        except Exception as e:
            raise CustomException(e,sys)

@dataclass
class DataTransformationConfig:
    process_obj_file_path=PREPROCESSING_FILE
    transform_test_path=TRANSFORM_TEST_PATH
    transform_train_path=TRANSFORM_TRAIN_PATH
    feature_engi_obj_path=FEATURE_ENGI_FILE

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def data_transformation_obj(self):
        try:
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam'] 
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy'] 

            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_columns = ['Road_traffic_density', 'Weather_conditions'] 
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition', 'multiple_deliveries','distance']

            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',StandardScaler(with_mean=False))
            ])

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('oneHot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            ordinal_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler',StandardScaler(with_mean=False))
            ])

            processor=ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_columns),
                ('ordinal_pipeline',ordinal_pipeline,ordinal_columns)
            ])

            logging.info("pipeline setup_complete")
            return processor
           

        except Exception as e:
            raise CustomException(e,sys)
    
    def get_feature_eng_obj(self):
        try:
            feature_engin=Pipeline(steps=[('fe',FeatureEngineering())])
            return feature_engin
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("fetaure eng. step obj")
            # fe_obj=self.get_feature_eng_obj()

            # train_df=fe_obj.fit_transform(train_df)
            # test_df=fe_obj.transform(test_df)

            
            fe_obj = self.get_feature_eng_obj()

            fe_obj.fit(train_df)

            train_df = fe_obj.transform(train_df)
            test_df = fe_obj.transform(test_df)
            

            train_df.to_csv("train_data.csv",index=False)
            test_df.to_csv("test_data.csv",index=False)

            process_obj=self.data_transformation_obj()

            target_column_name="Time_taken (min)"

            X_train=train_df.drop(columns=[target_column_name])
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=[target_column_name])
            y_test=test_df[target_column_name]

            X_train=process_obj.fit_transform(X_train)
            X_test=process_obj.transform(X_test)
            
            # train_arr=np.c_(X_train,np.array(y_train))
            # test_arr=np.c_(X_test,np.array(y_test))

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train=pd.DataFrame(train_arr)
            df_test=pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_path,index=False,header=True)

            save_obj(file_path=self.data_transformation_config.process_obj_file_path,obj=process_obj)
            save_obj(file_path=self.data_transformation_config.feature_engi_obj_path,obj=fe_obj)

            return(train_arr,test_arr,self.data_transformation_config.process_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)