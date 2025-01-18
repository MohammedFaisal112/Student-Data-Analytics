import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# To save a file in an binary format to an specific location.
def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj , file_obj)

    except Exception as e:
        logging.info(f"Error while saving an file , The error is {e}")
        raise CustomException(e,sys)


# To load a specific file/ read a binary file from a specific location.
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info(f"Error in loading a file, The Error is {e}")
        raise CustomException(e,sys)



# Training and Evaluating the result of the models.
def evaluate_model(X_train,Y_train,X_test,Y_test,models,param):
    try:
        report = {}
        for i in models:
            model = models[i]
            para = param[i]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            Y_train_Pred = model.predict(X_train)
            Y_test_Pred = model.predict(X_test)
            train_model_performance_score = r2_score(Y_train,Y_train_Pred)
            test_model_performance_score = r2_score(Y_test,Y_test_Pred)
            report[i]=test_model_performance_score
        
        return report

    except Exception as e:
        logging.info(f"Error in evaluate_model function. The Error is: {e}") 
        raise CustomException(e,sys)