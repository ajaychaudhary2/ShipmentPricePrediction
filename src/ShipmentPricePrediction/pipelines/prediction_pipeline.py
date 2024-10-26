import os
import sys
import numpy as np
import pandas as pd
from ShipmentPricePrediction.exception import customexception
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.utils.utils import load_object

class PredictPipeline:
    
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
           
            # Transforming the features using the preprocessor
            scaled_data = preprocessor.transform(features)

           
            # Making predictions
            pred = model.predict(scaled_data)
            
            return pred
        
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    
    def __init__(self,
                 shipment_mode: str,
                 country: str,
                 vendor_inco_term: str,
                 weight: float,
                 line_item_quantity: int,
                 line_item_insurance: float,
                 line_item_value: float):
        
        self.shipment_mode = shipment_mode
        self.country = country
        self.vendor_inco_term = vendor_inco_term
        self.weight = weight
        self.line_item_quantity = line_item_quantity
        self.line_item_insurance = line_item_insurance
        self.line_item_value = line_item_value
        
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "Shipment Mode": [self.shipment_mode],
                "Country": [self.country],
                "Vendor INCO Term": [self.vendor_inco_term],
                "Weight (Kilograms)": [self.weight],
                "Line Item Quantity": [self.line_item_quantity],
                "Line Item Insurance (USD)": [self.line_item_insurance],
                "Line Item Value": [self.line_item_value]  # Ensure this matches the expected column name
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe gathered")
            return df
            
        except Exception as e:
            logging.info("Exception occurred during prediction pipeline")
            raise customexception(e, sys)
