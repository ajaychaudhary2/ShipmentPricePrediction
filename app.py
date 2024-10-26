import os
import sys
import numpy as np
import pandas as pd
from ShipmentPricePrediction.exception import customexception
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.utils.utils import load_object
from ShipmentPricePrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from flask import Flask, render_template, request

app = Flask(__name__)

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

@app.route('/', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")  # Ensure this template exists
    
    else:
        try:
            # Retrieve form data and create CustomData instance
            data = CustomData(
                shipment_mode=request.form.get("shipment_mode"),
                country=request.form.get("country"),
                vendor_inco_term=request.form.get("vendor_inco_term"),
                weight=float(request.form.get("weight")),
                line_item_quantity=int(request.form.get("line_item_quantity")),
                line_item_insurance=float(request.form.get("line_item_insurance")),
                line_item_value=float(request.form.get("line_item_value"))  # Ensure this is included
            )
            
            
            
            # Convert input data to DataFrame
            final_data = data.get_data_as_df()
            
            
            # Create prediction pipeline instance and predict
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            
            # Round the prediction result
            result = round(pred[0], 2)
            
            # Render the result template with the prediction
            return render_template("result.html", predicted_value=result)
        
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred while processing your request.", 500

if __name__ == '__main__':
    app.run(debug=True)
