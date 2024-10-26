import os
import sys
import pickle
import numpy as np
import pandas as pd
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.exception import customexception
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves a Python object to a specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates the performance of multiple models on the test data.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        models: Dictionary of models to evaluate.

    Returns:
        report: Dictionary with model names as keys and R² scores as values.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict on the test data
            y_test_pred = model.predict(X_test)

            # Calculate R² score for the test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise customexception(e, sys)
    
def load_object(file_path):
    """
    Loads a Python object from a specified file path using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise customexception(e, sys)
