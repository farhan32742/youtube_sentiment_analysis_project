import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # Changed metrics for text
from sklearn.model_selection import cross_val_score  # Added for better text evaluation

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Use protocol=4 for better compatibility with large text vectorizers
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj, protocol=4)

    except Exception as e:
        logging.error(f"Error saving object to {file_path}")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, task_type='classification'):
    """
    Evaluate models with metrics appropriate for text data
    
    Parameters:
    - task_type: 'classification' (default) or 'regression' for text-based tasks
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_test_pred = model.predict(X_test)
            
            # Evaluation metrics based on task type
            if task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),
                    'precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    # Cross-validation score for more robust evaluation
                    'cv_score': np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted'))
                }
          
            
            report[model_name] = metrics

        return report

    except Exception as e:
        logging.error('Exception occurred during model evaluation')
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        # Handle potential memory issues with large text vectorizers
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f'Error loading object from {file_path}')
        raise CustomException(e, sys)

def save_text_vectorizer(vectorizer, file_path):
    """Specialized function to save large text vectorizers efficiently"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Use joblib for more efficient serialization of large numpy arrays
        from sklearn.externals import joblib
        joblib.dump(vectorizer, file_path)
    except Exception as e:
        logging.error(f"Error saving text vectorizer to {file_path}")
        raise CustomException(e, sys)

def load_text_vectorizer(file_path):
    """Specialized function to load text vectorizers"""
    try:
        from sklearn.externals import joblib
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f'Error loading text vectorizer from {file_path}')
        raise CustomException(e, sys)