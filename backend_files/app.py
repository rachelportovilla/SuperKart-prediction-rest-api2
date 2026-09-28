# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
from pathlib import Path
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_predictor_api = Flask("SuperKart Sales Forecast Predictor")

# Load the trained machine learning model
#model = joblib.load("backend_files/superkart_prediction_model_v1_0.joblib")
model = joblib.load(Path(__file__).with_name("superkart_prediction_model_v1_0.joblib"))

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Forecast API!"

# Define an endpoint for single product prediction (POST request)
@superkart_predictor_api.post('/v1/single')
def predict_single_sale():
    """
    This function handles POST requests to the '/v1/single' endpoint.
    It expects a JSON payload containing product details and returns
    the predicted sales revenue as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data, ensuring names match training features.
    # Use .get() to safely access keys and provide None if missing.
    sample = {
        'Product_Weight': product_data.get('Product_Weight'),
        'Product_Sugar_Content': product_data.get('Product_Sugar_Content'),
        'Product_Allocated_Area': product_data.get('Product_Allocated_Area'),
        'Product_Type': product_data.get('Product_Type') or product_data.get('Product_Type_Category'), # Handle Product_Type_Category
        'Product_MRP': product_data.get('Product_MRP'),
        'Store_Id': None,
        'Store_Establishment_Year': None, # Initialize as None, will be set below
        'Store_Size': product_data.get('Store_Size'),
        'Store_Location_City_Type': product_data.get('Store_Location_City_Type'), # Corrected typo
        'Store_Type': product_data.get('Store_Type')
    }

    # Impute missing 'Store_Id' values with 'OUT005'
    store_id = product_data.get('Store_Id')
    if store_id is None:
        sample['Store_Id'] = 'OUT005'
    else:
        sample['Store_Id'] = store_id

    # Handle Store_Establishment_Year and Store_Age_Years
    store_establishment_year = product_data.get('Store_Establishment_Year')
    store_age_years = product_data.get('Store_Age_Years')

    if store_establishment_year is not None:
        sample['Store_Establishment_Year'] = store_establishment_year
    elif store_age_years is not None:
        # Assuming Store_Age_Years is relative to current year
        sample['Store_Establishment_Year'] = 2026 - store_age_years
    else:
        # Impute remaining missing 'Store_Establishment_Year' values with 2009 (most frequent from EDA)
        sample['Store_Establishment_Year'] = 2009

    # Convert the extracted data into a Pandas DataFrame
    # Ensure the order of columns matches the training data features expected by the model
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float and round
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Sales (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/predictbatch')
def predict_batch_sales():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product details for multiple products
    and returns the predicted sales as a list in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Handle Store_Age_Years: convert to Store_Establishment_Year if present
    if 'Store_Age_Years' in input_data.columns:
        # If 'Store_Establishment_Year' column doesn't exist, create it from 'Store_Age_Years'
        if 'Store_Establishment_Year' not in input_data.columns:
            input_data['Store_Establishment_Year'] = 2026 - input_data['Store_Age_Years']
        else:
            # If 'Store_Establishment_Year' exists, but has NaNs, fill those NaNs with calculated values from 'Store_Age_Years'
            # Only where 'Store_Establishment_Year' is NaN, and 'Store_Age_Years' is not NaN
            mask_nan_establishment = input_data['Store_Establishment_Year'].isna()
            mask_not_nan_age = input_data['Store_Age_Years'].notna()
            input_data.loc[mask_nan_establishment & mask_not_nan_age, 'Store_Establishment_Year'] = \
                2026 - input_data.loc[mask_nan_establishment & mask_not_nan_age, 'Store_Age_Years']

        # Drop 'Store_Age_Years' as it's not a feature for the model
        input_data = input_data.drop(columns=['Store_Age_Years'])

    # Drop Product_Id_char if it exists, as it's not a feature for the model
    if 'Product_Id_char' in input_data.columns:
        input_data = input_data.drop(columns=['Product_Id_char'])

    # Rename Product_Type_Category to Product_Type if it exists, as this is the expected feature name
    if 'Product_Type_Category' in input_data.columns:
        input_data = input_data.rename(columns={'Product_Type_Category': 'Product_Type'})

    # Impute missing 'Store_Id' values with 'OUT005'
    # Ensure 'Store_Id' column exists before trying to fillna
    if 'Store_Id' not in input_data.columns:
        input_data['Store_Id'] = 'OUT005' # Assign default if column is entirely missing
    else:
        input_data['Store_Id'] = input_data['Store_Id'].fillna('OUT005')
    
    # Impute remaining missing 'Store_Establishment_Year' values with 2009
    input_data['Store_Establishment_Year'] = input_data['Store_Establishment_Year'].fillna(2009)

    # Make predictions for all products in the DataFrame
    predicted_sales_list = model.predict(input_data).tolist()

    # Round predictions to 2 decimal places
    predicted_sales_list = [round(float(sales), 2) for sales in predicted_sales_list]

    # Return the list of predictions as a JSON response
    return jsonify({'Predicted Sales (in dollars)': predicted_sales_list})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_predictor_api.run(debug=True)
