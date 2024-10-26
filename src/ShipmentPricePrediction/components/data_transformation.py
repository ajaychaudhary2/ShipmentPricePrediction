import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.exception import customexception
from ShipmentPricePrediction.utils.utils import save_object

class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DatatransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data transformation started")

            # Categorical and numerical columns
            cat_col = ['Shipment Mode', 'Country', 'Vendor INCO Term']
            num_col = ['Weight (Kilograms)', 'Line Item Quantity', 'Line Item Insurance (USD)', 'Line Item Value']

            # Categorical categories
            vendor = ['EXW', 'FCA', 'DDU', 'CIP', 'DDP', 'CIF', 'N/A - From RDC', 'DAP']
            ship_mode = ['Air', 'Truck', 'Air Charter', 'Ocean']
            cont = ['Vietnam', 'Haiti', 'South Africa', 'Rwanda', 'Nigeria', "Côte d'Ivoire", 'Guyana', 'Ethiopia',
                    'Namibia', 'Botswana', 'Zambia', 'Tanzania', 'Kazakhstan', 'Zimbabwe', 'Uganda', 'Kenya',
                    'Mozambique', 'Senegal', 'Benin', 'Lesotho', 'Pakistan', 'Swaziland', 'Ghana', 'Angola',
                    'Lebanon', 'Sierra Leone', 'South Sudan', 'Burundi', 'Congo, DRC', 'Dominican Republic',
                    'Cameroon', 'Sudan', 'Mali', 'Guatemala', 'Malawi', 'Togo', 'Afghanistan', 'Burkina Faso',
                    'Liberia', 'Guinea', 'Libya', 'Belize']

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[ship_mode, cont, vendor]))
            ])

            logging.info("Preprocessing pipelines created.")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, num_col),
                ("cat_pipeline", cat_pipeline, cat_col)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_df, test_df):
        try:
            # Define the target column and columns to drop
            target_column_name = 'Freight Cost (USD)'
            drop_columns = [target_column_name, "ID", "Project Code", "PQ #", "PO / SO #", "ASN/DN #", 
                            "Managed By", "Fulfill Via", "Unit of Measure (Per Pack)", "Pack Price", 
                            "Unit Price", "Manufacturing Site", "First Line Designation"]

            # Prepare training and test data by dropping unnecessary columns
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # Apply the transformation pipeline
            preprocessor = self.get_data_transformation()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Combine transformed features and target variable into final arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Applying preprocessing on train and test datasets")

            # Save the preprocessor object
            save_object(
                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Object saved successfully")

            return train_arr, test_arr  # Return the final training and testing arrays

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise customexception(e, sys)
