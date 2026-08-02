import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load the serialized model pipeline (preprocessing + tuned Random Forest)
MODEL_PATH = "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

superkart_api = Flask(__name__)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@superkart_api.get("/")
def home():
    return {"message": "SuperKart Sales Prediction API is running."}


@superkart_api.post("/v1/predict")
def predict():
    """Online (single-record) inference endpoint."""
    payload = request.get_json()
    input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_sales": round(float(prediction), 2)})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """Batch inference endpoint - accepts a CSV file with the FEATURE_COLUMNS."""
    file = request.files["file"]
    input_df = pd.read_csv(file)
    input_df = input_df[FEATURE_COLUMNS]
    predictions = model.predict(input_df)
    result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    superkart_api.run(host="0.0.0.0", port=port)
