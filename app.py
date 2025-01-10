from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from enum import Enum
import uvicorn
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = FastAPI()

class Gender(str, Enum):
    male = "male"
    female = "female"

class RaceEthnicity(str, Enum):
    group_a = "group A"
    group_b = "group B"
    group_c = "group C"
    group_d = "group D" 
    group_e = "group E"

class ParentalEducationLevel(str, Enum):
    high_school = "high school"
    associate_degree = "associate's degree"
    bachelor_degree = "bachelor's degree"
    master_degree = "master's degree"
    some_college = "some college"
    some_high_school = "some high school"

class Lunch(str, Enum):
    free_reduced = "free/reduced"
    standard = "standard"

class TestPreparation(str, Enum):
    none = "none"
    completed = "completed"

class CustomData(BaseModel):
    gender: Gender
    race_ethnicity: RaceEthnicity
    parental_level_of_education: ParentalEducationLevel
    lunch: Lunch
    test_preparation_course: TestPreparation
    reading_score: Annotated[int, Field(gt=0, le=101)]  
    writing_score: Annotated[int, Field(gt=0, le=101)]


    def get_data_as_dataframe(self):
        try:
            data = {key: (value.value if isinstance(value, Enum) else value) for key, value in self.model_dump().items()}

            data_df = pd.DataFrame([data])
            data_df.rename(columns={
                'gender': 'gender',
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education',
                'lunch': 'lunch',
                'test_preparation_course': 'test preparation course',
                'reading_score': 'reading score',
                'writing_score': 'writing score'
            }, inplace=True)

            logging.info(f"dataframe is ready....{type(data_df)}, {data_df.shape}!!!")
            return data_df
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
class Gender(str, Enum):
            male = "male"
            female = "female"

class RaceEthnicity(str, Enum):
    group_a = "group A"
    group_b = "group B"
    group_c = "group C"
    group_d = "group D" 
    group_e = "group E"

class ParentalEducationLevel(str, Enum):
    high_school = "high school"
    associate_degree = "associate's degree"
    bachelor_degree = "bachelor's degree"
    master_degree = "master's degree"
    some_college = "some college"
    some_high_school = "some high school"

class Lunch(str, Enum):
    free_reduced = "free/reduced"
    standard = "standard"

class TestPreparation(str, Enum):
    none = "none"
    completed = "completed"

class CustomData(BaseModel):
    gender: Gender
    race_ethnicity: RaceEthnicity
    parental_level_of_education: ParentalEducationLevel
    lunch: Lunch
    test_preparation_course: TestPreparation
    reading_score: Annotated[int, Field(gt=0, le=100)]  
    writing_score: Annotated[int, Field(gt=0, le=100)]


    def get_data_as_dataframe(self):
        try:
            data = {key: (value.value if isinstance(value, Enum) else value) for key, value in self.model_dump().items()}

            data_df = pd.DataFrame([data])
            data_df.rename(columns={
                'gender': 'gender',
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education',
                'lunch': 'lunch',
                'test_preparation_course': 'test preparation course',
                'reading_score': 'reading score',
                'writing_score': 'writing score'
            }, inplace=True)

            logging.info(f"dataframe is ready....{type(data_df)}, {data_df.shape}!!!")
            return data_df
        
        except Exception as e:
            raise CustomException(e,sys)


@app.get("/")
def root():
    return 'Welcome to End-To-End ML Project....(:'

@app.post("/predict/")
def predict(data: Annotated[CustomData, Form()]):
    try:
        pred_pipeline = PredictPipeline()
        df = data.get_data_as_dataframe()
        logging.info("Now prediction is taking place...!")

        prediction = pred_pipeline.pred(features = df)

        return f"math score prediction: {prediction}"
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)





