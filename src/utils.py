# It will have some common thing which we probabli going to use in our project...

import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def evalute_models(X_train, y_train, X_test, y_test, models, param, cv = 3, n_jobs = 3, verbose = 1):
        try:
            report = {}

            for i in range(len(list(models))):

                model = list(models.values())[i]

                parameters = param[list(models.keys())[i]]

                gs = GridSearchCV(model, parameters, cv=cv, n_jobs = n_jobs, verbose = verbose)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                logging.info(f"Hyperparameter is done...Best params = {model}")
                

                #model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_accuracy = r2_score(y_train, train_pred)
                test_accuracy = r2_score(y_test, test_pred)

                report[list(models.keys())[i]] = test_accuracy

            return report
        
        except Exception as e:
            raise CustomException(e, sys)
        
def load_object(file_path):
     try:
          with open(file_path, "rb") as file:
               logging.info(f"file from {file_path} has loading...!!")
               return dill.load(file)
          
          
     except Exception as e:
          raise CustomException(e, sys)