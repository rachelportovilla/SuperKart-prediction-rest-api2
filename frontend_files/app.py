import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Page title
st.title("SuperKart Sales Forecast App")
st.write(
    "Enter the product and store details below to predict the sales revenue."
)

# Collect product and store details
Product_Weight = st.number_input(
    "Product Weight (in kg)",
    min_value=0.0,
    value=12.66
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.027
)

Product_Type = st.selectbox(
    "Product Type",
    ["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
     "Snack Foods", "Household", "Meat", "Soft Drinks", "Breads",
     "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood",
     "Fruits and Vegetables"]
)

Product_MRP = st.number_input(
    "Product MRP (Maximum Retail Price)",
    min_value=0.0,
    value=117.08
)

Store_Id = st.selectbox(
    "Store ID",
    ["OUT004", "OUT003", "OUT001", "OUT002"]
)

Store_Establishment_Year = st.number_input(
    "Store Establishment Year",
    min_value=1900,
    max_value=2026,
    value=2009
)

Store_Size = st.selectbox(
    "Store Size",
    ["Medium", "High", "Small"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 2", "Tier 1", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type2", "Departmental Store", "Supermarket Type1", "Food Mart"]
)

# Create JSON payload for single prediction
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}

# Single Prediction
if st.button("Predict Sales", type="primary"):
    response = requests.post(
        f"{BACKEND_URL}/v1/single",
        json=product_data
    )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Sales: ${result['Predicted Sales (in dollars)']:.2f}")
    else:
        st.error(f"Error making prediction: {response.status_code} - {response.text}")

# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file for batch prediction",
    type=["csv"]
)

if uploaded_file is not None:
    if st.button("Predict for Batch", type="primary"):
        files = {'file': uploaded_file.getvalue() 
        # Reset file pointer to the beginning after reading if it's already been read
        if uploaded_file.tell() > 0 else uploaded_file.getvalue()
        }

        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files=files
        )

        if response.status_code == 200:
            results = response.json()
            st.success("Predictions completed successfully!")
            
            try:
                df_results = pd.DataFrame(results['Predicted Sales (in dollars)'], columns=['Predicted Sales'])
                st.dataframe(df_results, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to display results as a table: {e}")
                st.json(results)

        else:
            st.error(f"Error making batch prediction: {response.status_code} - {response.text}")
