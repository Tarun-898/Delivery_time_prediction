# import os,sys
# from src.constant import *
# from src.config.configuration import *
# from src.logger import logging
# from src.exception import CustomException
# from src.components import data_Transformation,ingestion,model_trainer


# logging.info("pipeline started ***************")
# if __name__ == "__main__":
#     obj = ingestion.DataIngestion()
#     train_path, test_path = obj.initiate_data_ingestion()

#     transform = data_Transformation.DataTransformation()
#     train_arr, test_arr, _ = transform.initate_data_transformation(
#         train_path,
#         test_path
#     )

#     trainer = model_trainer.ModelTrainer()
#     logging.info("pipeline complete ***************")
#     print(trainer.initiate_model_training(train_arr, test_arr))


from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict import PredictPipeline,CustomData

application=Flask(__name__)
app=application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        data = CustomData(
            Delivery_person_Age=int(request.form.get("Delivery_person_Age")),
            Delivery_person_Ratings=float(request.form.get("Delivery_person_Ratings")),
            Vehicle_condition=int(request.form.get("Vehicle_condition")),
            multiple_deliveries=int(request.form.get("multiple_deliveries")),
            Restaurant_latitude=float(request.form.get("Restaurant_latitude")),
            Restaurant_longitude=float(request.form.get("Restaurant_longitude")),
            Delivery_location_latitude=float(request.form.get("Delivery_location_latitude")),
            Delivery_location_longitude=float(request.form.get("Delivery_location_longitude")),
            Type_of_order=request.form.get("Type_of_order"),
            Type_of_vehicle=request.form.get("Type_of_vehicle"),
            Festival=request.form.get("Festival"),
            City=request.form.get("City"),
            Road_traffic_density=request.form.get("Road_traffic_density"),
            Weather_conditions=request.form.get("Weather_conditions")
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)