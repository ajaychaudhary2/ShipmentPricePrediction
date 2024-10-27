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
