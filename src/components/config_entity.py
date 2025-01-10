import os
from dataclasses import dataclass



@dataclass   # use it only when we need to add only parameter...if want to add functions go with the full mode...!!!
class DataIngestionConfig:
    train_data_path: str = os.path.join("data storage", "train.csv")
    test_data_path: str = os.path.join("data storage", "test.csv")
    raw_data_path: str = os.path.join("data storage", "data.csv")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("data storage", "preprocessor.pkl")  #processor will apply scaling and encoding on num and cat feature...


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data storage", "model.pkl")