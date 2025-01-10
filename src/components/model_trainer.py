import os 
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.utils import evalute_models, save_object
from src.exception import CustomException
from src.components.config_entity import ModelTrainerConfig
from src.components.hyperparameters import params


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initial_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data...")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            parameters = params

            model_report:dict = evalute_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)

            # Getting best model from the dictionary...
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
        
            logging.info(f"Best model selected: {best_model_name}...!!")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            pred = best_model.predict(X_test)
            accuracy_score = r2_score(y_test, pred)

            return accuracy_score

        except Exception as e:
            raise CustomException(e, sys)

    