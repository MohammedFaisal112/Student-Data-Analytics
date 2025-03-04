# Reading a files from different sources/databases.

# Importing necessary packages.
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

# In the DataIngestionConfig class, we typically pass the input as arguments or parameters to the class, rather than storing the inputs directly.
@dataclass              # Alternative of an constructor.Shorthand for initialization of an instance variable.
class DataIngestionConfig:
    raw_data_path: str= os.path.join('artifacts',"data.csv")
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\student_data.csv')
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)   

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train_test_split initiated")
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)

            # df2 = pd.read_csv(train_set)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(f"Data Ingestion method having an error: {e}")
            raise CustomException(e, sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr , test_arr,_ = data_transformation.initiate_data_transformation(train_data , test_data)

    model_training = ModelTrainer()
    best_model,best_model_score=model_training.initiate_model_trainer(train_arr,test_arr)

    print(f"The Model which performed well is:{best_model} and the Performance/r2_score of the model is:{best_model_score}")