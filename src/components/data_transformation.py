import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.components.config_entity import DataTransformationConfig
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function transform the data....
        '''
        try:
            num_col = ["writing score", "reading score"]
            cat_col = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")), #handle missung values...
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical column standard scaling completed..")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), #handle missung values...
                    ("OneHotEncoder", OneHotEncoder()),  #Handle categorical features...
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical column encoding & standard scaling completed..")

            logging.info(f"Numerical Columns: {num_col}")
            logging.info(f"Categorical Columns: {cat_col}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, num_col),
                ("cat_pipeline", cat_pipeline, cat_col)
                ]
            )
            

            return preprocessor
        

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test csv as Data Frame....!!!")

            logging.info("Preprocessing initiate....!!")

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "math score"
            num_col = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe....!!")
            
            transformed_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            transformed_test_array = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Preprocessing done...!!!")

            train_array = np.c_[transformed_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[transformed_test_array, np.array(target_feature_test_df)]

            logging.info("Saving preprocessed data...!!")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor_obj
                )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
