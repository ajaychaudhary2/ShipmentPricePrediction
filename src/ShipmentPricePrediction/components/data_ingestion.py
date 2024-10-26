import pandas as pd
import numpy as np
import os
import sys

from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.exception import customexception

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

# Define DataIngestionConfig as a class
@dataclass
class DataIngestionConfig:
    rawdata_path: str = os.path.join("Artifacts", "raw.csv")
    train_path: str = os.path.join("Artifacts", "train.csv")
    test_path: str = os.path.join("Artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        # Initialize DataIngestionConfig
        self.DataIngestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Start")

        try:
            # Read the data from the dataset
            data = pd.read_csv(Path(os.path.join("notebooks/data", "cleaned_data.csv")))
            logging.info("I have read the dataset")

            # Ensure that the directories exist and save the raw data
            os.makedirs(os.path.dirname(self.DataIngestion_config.rawdata_path), exist_ok=True)
            data.to_csv(self.DataIngestion_config.rawdata_path, index=False)
            logging.info("Successfully saved the raw data in the Artifacts folder")

            # Perform train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save train and test data
            train_data.to_csv(self.DataIngestion_config.train_path, index=False)
            test_data.to_csv(self.DataIngestion_config.test_path, index=False)

            logging.info("Successfully saved the train and test data in the Artifacts folder")
            logging.info("Data Ingestion Complete")

            # Return DataFrames instead of paths
            return train_data, test_data  # Return DataFrames directly

        except Exception as e:
            logging.error("Exception occurred during the data ingestion stage")
            raise customexception(e, sys)

