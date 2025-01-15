# Transform one form of data into an another form.

import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self,data):
        '''
        This function do a Data Transformation.
        '''
        try:
            numerical_features = [feature for feature in data.columns if data[feature].dtype!='O']
            categorical_features = [feature for feature in data.columns if data[feature].dtype=='O']

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer" ,SimpleImputer(strategy="median")),
                    ("scalar" , StandardScaler())
                ]
            )
            categoric_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder" , OneHotEncoder()),
                    ("scalar" , StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numeric_pipeline,numerical_features),
                    ("categorical_pipeline",categoric_pipeline,categorical_features)
                ]
            )

            logging.info("Numerical Columns StandardScaling Completed")
            logging.info("Categorical Columns Encoding Completed")

            return preprocessor

        except Exception as e:
            logging.info(f"Error in get_data_transformation_object function, The error is: {e}")
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            example = train_data.drop(columns='math_score',axis=1)

            logging.info("Reading of training and testing data completed.")

            logging.info("Obtaining Preprocessing Object")
            preprocessing_obj = self.get_data_transformation_object(example)

            target_column_name = "math_score"

            input_training_feature = train_data.drop(columns=target_column_name,axis=1)
            target_training_feature = train_data[target_column_name]
            
            input_testing_feature = test_data.drop(columns=target_column_name,axis=1)
            target_testing_feature = test_data[target_column_name]

            logging.info("Applying Preprocessing object into an training and testing data")

            input_train_feature_arr = preprocessing_obj.fit_transform(input_training_feature)
            input_testing_feature_arr = preprocessing_obj.transform(input_testing_feature)

            # Concatinating input features and a target feature.
            train_arr = np.c_[input_train_feature_arr , np.array(target_training_feature)]
            test_arr = np.c_[input_testing_feature_arr , np.array(target_testing_feature)]

            logging.info(f"Saved preprocessing object.")
            save_object(file_path = self.data_transformation_config.preprocessor_file_path , obj = preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            logging.info(f"Error in initiate_data_transformation function, The error is: {e}")
            raise CustomException(e,sys)
        