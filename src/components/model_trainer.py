import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score, precision_score, f1_score, 
    accuracy_score, roc_auc_score, confusion_matrix
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "model.pkl")
    optimal_threshold: float = 0.4

class ModelTrainer:
    """
    Model Training with ImbPipeline (SMOTE + Model)

    """
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Model Training
        """
        logging.info("STARTING MODEL TRAINING (with ImbPipeline)")
        
        try:
            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)
            
            logging.info("Creating ImbPipeline with SMOTE + KNN")
            
            impipeline = ImbPipeline(steps=[
                ('smote', SMOTE(random_state=42)),
                ('classifier', KNeighborsClassifier())
            ])
            
            param_grid = {
                'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan'],
                'classifier__p': [1, 2]
            }
            
            logging.info(f"Parameter grid: {param_grid}")
            
            logging.info("Setting up GridSearchCV with ImbPipeline")
            
            grid_search = GridSearchCV(
                estimator=impipeline,
                param_grid=param_grid,
                cv=5,
                scoring='recall'
            )

            logging.info("Training model (SMOTE applied in each CV fold)")
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            
            logging.info("BEST HYPERPARAMETERS:")
            for param, value in best_params.items():
                logging.info(f"{param}: {value}")


            y_test_pred = best_model.predict(X_test)
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            
            logging.info("\nTest Set (Default Threshold = 0.5):")
            logging.info(f"  Recall:    {test_recall:.4f}")
            logging.info(f"  Precision: {test_precision:.4f}")
            logging.info(f"  F1-Score:  {test_f1:.4f}")
            logging.info(f"  Accuracy:  {test_accuracy:.4f}")
            logging.info(f"  ROC-AUC:   {test_roc_auc:.4f}")
            
            optimal_threshold = self.model_trainer_config.optimal_threshold
            y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
            
            test_recall_optimal = recall_score(y_test, y_test_pred_optimal)
            test_precision_optimal = precision_score(y_test, y_test_pred_optimal)
            test_f1_optimal = f1_score(y_test, y_test_pred_optimal)
            
            cm = confusion_matrix(y_test, y_test_pred_optimal)
            tn, fp, fn, tp = cm.ravel()
            
            logging.info(f"\nTest Set (Optimal Threshold = {optimal_threshold}):")
            logging.info(f"  Recall:    {test_recall_optimal:.4f}")
            logging.info(f"  Precision: {test_precision_optimal:.4f}")
            logging.info(f"  F1-Score:  {test_f1_optimal:.4f}")
            logging.info(f"\nConfusion Matrix:")
            logging.info(f"  True Negatives:  {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  True Positives:  {tp}")

            model_package = {
                'pipeline': best_model,
                'threshold': optimal_threshold,
                'best_params': best_params,
                'metrics': {
                    'test_recall': test_recall_optimal,
                    'test_precision': test_precision_optimal,
                    'test_f1': test_f1_optimal,
                    'test_roc_auc': test_roc_auc
                }
            }
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_package
            )
            
            logging.info(f'\nPipeline saved to: {self.model_trainer_config.trained_model_file_path}')
            logging.info('MODEL TRAINING COMPLETED')
            
            return test_recall_optimal
            
        except Exception as e:
            raise CustomException(e, sys)