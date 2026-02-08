import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before Loading")
            model_package = load_object(file_path=model_path)

            model = model_package['pipeline']
            threshold = model_package['threshold']

            preprocessor=load_object(file_path=preprocessor_path)
            
            print("After Loading")
            data_scaled=preprocessor.transform(features)

            proba = model.predict_proba(data_scaled)[:, 1]
            pred = (proba >= threshold).astype(int)

            return pred
        
        except Exception as e:
            raise CustomException(e,sys)


##Class for mapping our columns
class CustomData:
    def __init__(self,
                 Gender: str,
                 age: float,
                 education: str,
                 currentSmoker: str,
                 cigsPerDay: float,
                 BP_Meds: str,
                 prevalentStroke: str,
                 prevalentHyp: str,
                 diabetes: str,
                 tot_cholesterol: float,
                 Systolic_BP: float,
                 Diastolic_BP: float,
                 BMI: float,
                 heartRate: float,
                 glucose: float):
        self.Gender = Gender
        self.age = age
        self.education = education
        self.currentSmoker = currentSmoker
        self.cigsPerDay = cigsPerDay
        self.BP_Meds = BP_Meds
        self.prevalentStroke = prevalentStroke
        self.prevalentHyp = prevalentHyp
        self.diabetes = diabetes
        self.tot_cholesterol = tot_cholesterol
        self.Systolic_BP = Systolic_BP
        self.Diastolic_BP = Diastolic_BP
        self.BMI = BMI
        self.heartRate = heartRate
        self.glucose = glucose

    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "age": [self.age],
                "education": [self.education],
                "currentSmoker": [self.currentSmoker],
                "cigsPerDay": [self.cigsPerDay],
                "BP_Meds": [self.BP_Meds],
                "prevalentStroke": [self.prevalentStroke],
                "prevalentHyp": [self.prevalentHyp],
                "diabetes": [self.diabetes],
                "tot_cholesterol": [self.tot_cholesterol],
                "Systolic_BP": [self.Systolic_BP],
                "Diastolic_BP": [self.Diastolic_BP],
                "BMI": [self.BMI],
                "heartRate": [self.heartRate],
                "glucose": [self.glucose]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)