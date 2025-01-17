# Train a model.

import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor 
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.util import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting an training and testing array")
            X_train , Y_train , X_test , Y_test = (train_arr[:,:-1] , train_arr[:,-1] , test_arr[:,:-1] , test_arr[:,-1])

            models = {"Linear Regression":LinearRegression() ,"Ridge" : Ridge() ,"Lasso": Lasso(),"Decision Tree":DecisionTreeRegressor(),"Random Forest":RandomForestRegressor(),"K-Neighbour Regressor":KNeighborsRegressor() , "Adaptive Boost Regressor":AdaBoostRegressor(),"Gradient Boosting Regressor":GradientBoostingRegressor(),"Xtreme Gradient Boosting Regressor":XGBRegressor()}

            model_report: dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

            best_score = max(sorted(model_report.values()))
            logging.info(f"The best R2_Score with respect to an model is: {best_score}")


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            logging.info(f"The Model who performed best is: {best_model_name}")

            best_model = models[best_model_name]

            if best_score<0.6:
                raise CustomException("No Best Model found")
            logging.info("Best Model founded.")

            save_object(self.model_trainer_config.model_file_path , obj=best_model)

            return best_model_name,best_score

        except Exception as e:
            logging.info("Error occured in initiate_model_trainer function. The Error is {e}")
            raise CustomException(e,sys)


