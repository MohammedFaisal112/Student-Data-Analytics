# Triggered the prediction process.
import pandas as pd
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.util import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            processor_path = os.path.join('artifacts','preprocessor.pkl')
            loading_data_processor = load_object(file_path=processor_path)
            loading_model = load_object(file_path=model_path)
            preprocessing = loading_data_processor.transform(feature)
            model_trained = loading_model.predict(preprocessing)
            return model_trained
        except Exception as e:
            logging.info(f"Error in PredictPipeline , The Error is {e}")
            raise CustomException(e,sys)


# Mapping all the html inputs with the backend with some perticular values.
class CustomData:
    def __init__(self,
            gender : str,
            race_ethnicity : str,
            parental_level_of_education : str,
            lunch : str,
            test_preparation_course : str,
            reading_score : int,
            writing_score : int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course  =test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender' : [self.gender],
                'race_ethnicity' : [self.race_ethnicity],
                'parental_level_of_education' : [self.parental_level_of_education],
                'lunch' : [self.lunch],
                'test_preparation_course' : [self.test_preparation_course],
                'reading_score' : [self.reading_score],
                'writing_score' : [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.info(f"Error in get_data_as_data_frame function. The Error is {e}")
            raise CustomException(e,sys)
