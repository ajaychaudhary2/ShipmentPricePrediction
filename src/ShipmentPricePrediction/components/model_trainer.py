import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.exception import customexception
from ShipmentPricePrediction.utils.utils import save_object
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting dependent and independent variables from the transformed data')

            # Split the train_arr into features (X) and target variable (y)
            X_train = train_arr[:, :-1]  # All columns except the last one
            y_train = train_arr[:, -1]    # Last column as target

            # Split the test_arr into features (X) and target variable (y)
            X_test = test_arr[:, :-1]      # All columns except the last one
            y_test = test_arr[:, -1]       # Last column as target

            # Define models to train
            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor()
            }

            # Initialize a report to store performance metrics
            model_report = {}

            # Train and evaluate each model
            for model_name, model in models.items():
                # Fit the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate model performance
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2_value = r2_score(y_test, y_pred)

                # Store model performance metrics
                model_report[model_name] = r2_value

                logging.info(f'{model_name} Performance: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2_value * 100:.2f}%')

            # Get the best model based on R² score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name} with R² Score: {best_model_score * 100:.2f}%')
            logging.info(f'Best Model Found: {best_model_name} with R² Score: {best_model_score * 100:.2f}%')

            # Save the best model to a file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.error('Exception occurred at Model Training')
            raise customexception(e, sys)

# Example usage:
# model_trainer = ModelTrainer()
# train_arr, test_arr = ...  # Assume these are obtained from your data transformation class
# model_trainer.initiate_model_training(train_arr, test_arr)
