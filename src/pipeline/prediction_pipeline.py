import numpy as np
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def pred(self, features):
        try:
            model_path = "./data storage/model.pkl"
            preprocessor_path = "./data storage/preprocessor.pkl"

            logging.info("Model and preprocessor is loading...!!")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            logging.info("Model and preprocessor has loaded...!!")
      
            scaled_data = preprocessor.transform(features)
            
            prediction = model.predict(scaled_data)

            logging.info(f"Prediction = {prediction}, {type(prediction)}") # fast api support JASON format which is lightweight, human-readable, and machine-readable data interchange format.

            prediction_output = int(prediction[0])
            return prediction_output
        
        except Exception as e:
            raise CustomException(e, sys)


    


