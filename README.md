# Shipment Price Prediction

A machine learning project focused on predicting shipment prices based on various shipment-related features. The project aims to develop a robust model capable of estimating prices, assisting logistics and supply chain decision-makers in cost management and planning.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Features](#features)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview
This project leverages machine learning to predict shipment costs based on various input features such as shipment mode, delivery dates, item details, weight, and more. By analyzing historical data, the model aims to achieve a high accuracy rate, supporting users with informed cost estimates for future shipments.

## Dataset
The dataset contains shipment-related details, including:
- **Basic Information**: Project Code, PQ #, PO / SO #, ASN/DN #, Country
- **Logistics Details**: Managed By, Fulfill Via, Vendor INCO Term, Shipment Mode
- **Date Information**: PQ First Sent to Client Date, PO Sent to Vendor Date, Scheduled Delivery Date, Delivered to Client Date
- **Product Details**: Product Group, Sub Classification, Vendor, Item Description, Molecule/Test Type, Brand, Dosage, Dosage Form
- **Cost Details**: Line Item Quantity, Line Item Value, Pack Price, Unit Price, Freight Cost (USD), Line Item Insurance (USD)

**Note**: Due to privacy or proprietary reasons, the dataset is not included in this repository.

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/ajaychaudhary2/ShipmentPricePrediction.git
   cd ShipmentPricePrediction



Features
1.Outlier Detection & Removal: Identify and remove outliers to improve model performance.
2.Data Preprocessing: Clean and transform data to make it suitable for model training.
3.Model Training Pipeline: An automated training pipeline that manages data processing, model training, and evaluation.
4.Evaluation Metrics: Measure model performance using metrics such as R² score to ensure robust prediction accuracy.


Model Training
The project includes a custom training pipeline, which executes the following:

1.Data Loading: Loads and prepares the dataset for model training.
2.Model Selection: Tests different algorithms and selects the best-performing model based on evaluation metrics.
3.Training: Trains the selected model on the preprocessed data.
4.Saving: Saves the trained model for future predictions.
5.Model Evaluation
6.Evaluate the model performance using metrics such as:

7.R² Score: Indicates the proportion of variance captured by the model.
8.Mean Absolute Error (MAE): The average magnitude of errors in predictions.
9.Root Mean Squared Error (RMSE): Measures the standard deviation of prediction errors.
9.Achieving an R² score of at least 80% is a primary goal for this project.

