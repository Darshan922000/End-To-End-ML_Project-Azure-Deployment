import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.config_entity import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered initiate_data_ingestion in DataIngestion...")
        try:
            df = pd.read_csv("Notebook\Data\StudentsPerformance.csv")  #path of the data// It can be databse as well...
            logging.info("Read data csv as data frame...!!")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train_Test_Split initiated...!!")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data have Ingested....!!1")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path= train_data_path, test_path=test_data_path)
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initial_model_trainer(train_array=train_arr, test_array=test_arr)
    print(r2_score)
