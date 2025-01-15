import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.exception import CustomException
from src.logger import logging

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj , file_obj)

    except Exception as e:
        logging.info(f"Error while saving an processor.pkl file , The error is {e}")
        raise CustomException(e,sys)


