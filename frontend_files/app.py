import os
import streamlit as st
import requests
import pandas as pd

# Set this to your backend's public URL (Hugging Face Space URL or Codespace forwarded URL).
# It can also be provided as an environment variable named BACKEND_URL when running the container,
# e.g. `docker run -e BACKEND_URL=https://your-username-superkart-backend.hf.space ...`
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

st.title("SuperKart — Sales Revenue Predictor")
st.write("Predict the expected sales revenue for a product-store combination.")
st.caption(f"Connected backend: {BACKEND_URL}")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Enter Product & Store Details")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
        mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
        product_id_char = st.selectbox("Product Category (Id prefix)", ["FD", "DR", "NC"])
    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Food Mart", "Supermarket Type1", "Supermarket Type2", "Departmental Store"])
        store_age = st.number_input("Store Age (Years)", min_value=0, value=15)
        product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if response.status_code == 200:
                st.success(f"Predicted Sales: {response.json()['predicted_sales']}")
            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend at {BACKEND_URL}: {e}")

with tab2:
    st.subheader("Upload a CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.dataframe(pd.read_csv(uploaded_file))
        uploaded_file.seek(0)
        if st.button("Run Batch Prediction"):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=60)
                if response.status_code == 200:
                    st.success("Batch prediction complete!")
                    st.json(response.json())
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
