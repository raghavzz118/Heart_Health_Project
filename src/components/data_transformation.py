import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def remove_outlier(self, data):
        """
        Remove outliers using IQR method
        
        """

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return lower_bound, upper_bound

        

    
    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['age', 'cigsPerDay', 'tot_cholesterol', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'heartRate', 'glucose']
            categorical_columns = ['Gender', 'education', 'currentSmoker', 'BP_Meds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                      ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer(transformers=
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='Heart_Attack'

            numerical_columns = ['age', 'cigsPerDay', 'tot_cholesterol', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'heartRate', 'glucose']

            cols_to_treat = [col for col in numerical_columns if col != 'glucose']

            for col in cols_to_treat:
                # Calculate bounds using TRAIN data only
                lower, upper = self.remove_outlier(train_df[col])
                
                # Apply to train data
                train_df[col] = np.where(train_df[col] > upper, upper, train_df[col])
                train_df[col] = np.where(train_df[col] < lower, lower, train_df[col])
                
                # Apply to test data
                test_df[col] = np.where(test_df[col] > upper, upper, test_df[col])
                test_df[col] = np.where(test_df[col] < lower, lower, test_df[col])

            logging.info(f"Outlier treatment completed for {len(cols_to_treat)} columns")            

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            #Concatination
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            
            # Test array (NO SMOTE on test data!)
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]


            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)