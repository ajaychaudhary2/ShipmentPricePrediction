from ShipmentPricePrediction.components.data_ingestion import DataIngestion
from ShipmentPricePrediction.components.data_transformation import DataTransformation
from ShipmentPricePrediction.components.model_trainer import ModelTrainer
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.exception import customexception
import os
import sys

if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_df, test_df = obj.initiate_data_ingestion()  # Ensure this returns DataFrames

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_df, test_df)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)  # Ensure this method is correctly defined

        print("Training pipeline executed successfully.")

    except customexception as e:
        logging.error(f"Custom Exception: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
