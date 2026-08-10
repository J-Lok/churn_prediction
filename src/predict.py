import os
import pandas as pd
import numpy as np
import joblib

# Load models and preprocessing artifacts
MODELS_DIR = "models"
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.joblib")
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_model.joblib")
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")

_models_loaded = False
_knn_model = None
_logreg_model = None
_mlp_model = None
_scaler = None
_feature_columns = None

def load_prediction_artifacts():
    global _knn_model, _logreg_model, _mlp_model, _scaler, _feature_columns, _models_loaded
    if not _models_loaded:
        if not os.path.exists(KNN_MODEL_PATH):
            raise FileNotFoundError(f"KNN model not found. Run training script first.")
        
        _knn_model = joblib.load(KNN_MODEL_PATH)
        _logreg_model = joblib.load(LOGREG_MODEL_PATH)
        _mlp_model = joblib.load(MLP_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _feature_columns = joblib.load(FEATURES_PATH)
        _models_loaded = True

def preprocess_single_input(customer_dict):
    """
    Transforms a single customer dictionary into the correct 40-feature vector.
    """
    load_prediction_artifacts()
    
    # 1. Clean and parse numerical values
    try:
        tenure = float(customer_dict.get("tenure", 0))
        monthly_charges = float(customer_dict.get("MonthlyCharges", 0))
        total_charges_raw = customer_dict.get("TotalCharges", 0)
        # Handle empty/whitespace TotalCharges (new customers might have tenure=0 and empty TotalCharges)
        if str(total_charges_raw).strip() == "":
            total_charges = monthly_charges * tenure if tenure > 0 else 0.0
        else:
            total_charges = float(total_charges_raw)
    except (ValueError, TypeError):
        tenure = 0.0
        monthly_charges = 0.0
        total_charges = 0.0

    # 2. Scale numeric variables
    # The scaler expects a 2D array or DataFrame of shape (n_samples, 3) for ["tenure", "MonthlyCharges", "TotalCharges"]
    df_mms_input = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=["tenure", "MonthlyCharges", "TotalCharges"])
    scaled_nums = _scaler.transform(df_mms_input)[0]
    scaled_tenure, scaled_monthly_charges, scaled_total_charges = scaled_nums
    
    # 3. Handle binary categorical mapping (Yes/No, Female/Male)
    gender = 1 if str(customer_dict.get("gender", "")).lower() == "female" else 0
    senior_citizen = 1 if int(customer_dict.get("SeniorCitizen", 0)) == 1 else 0
    partner = 1 if str(customer_dict.get("Partner", "")).lower() == "yes" else 0
    dependents = 1 if str(customer_dict.get("Dependents", "")).lower() == "yes" else 0
    phone_service = 1 if str(customer_dict.get("PhoneService", "")).lower() == "yes" else 0
    paperless_billing = 1 if str(customer_dict.get("PaperlessBilling", "")).lower() == "yes" else 0
    
    # 4. Prepare base prediction dictionary with 40 features set to 0
    x_dict = {col: 0 for col in _feature_columns}
    
    # Set scaled numerics & binary categoricals
    x_dict["tenure"] = scaled_tenure
    x_dict["MonthlyCharges"] = scaled_monthly_charges
    x_dict["TotalCharges"] = scaled_total_charges
    x_dict["gender"] = gender
    x_dict["SeniorCitizen"] = senior_citizen
    x_dict["Partner"] = partner
    x_dict["Dependents"] = dependents
    x_dict["PhoneService"] = phone_service
    x_dict["PaperlessBilling"] = paperless_billing
    
    # 5. Map multi-value categorical fields to dummy columns
    features_ohe = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaymentMethod"
    ]
    
    for field in features_ohe:
        val = str(customer_dict.get(field, "")).strip()
        # Clean PaymentMethod if it contains (automatic)
        if field == "PaymentMethod":
            val = val.replace(" (automatic)", "")
            
        dummy_col = f"{field}_{val}"
        if dummy_col in x_dict:
            x_dict[dummy_col] = 1
            
    # Create DataFrame to preserve column ordering exactly
    df_row = pd.DataFrame([x_dict], columns=_feature_columns)
    return df_row

def predict_churn(customer_dict):
    """
    Returns prediction results from all three models.
    """
    load_prediction_artifacts()
    df_row = preprocess_single_input(customer_dict)
    
    # Get predictions
    pred_knn = int(_knn_model.predict(df_row)[0])
    prob_knn = float(_knn_model.predict_proba(df_row)[0][1])
    
    pred_logreg = int(_logreg_model.predict(df_row)[0])
    prob_logreg = float(_logreg_model.predict_proba(df_row)[0][1])
    
    pred_mlp = int(_mlp_model.predict(df_row)[0])
    prob_mlp = float(_mlp_model.predict_proba(df_row)[0][1])
    
    return {
        "knn": {
            "prediction": pred_knn,
            "probability": prob_knn,
            "churn": pred_knn == 1
        },
        "logistic_regression": {
            "prediction": pred_logreg,
            "probability": prob_logreg,
            "churn": pred_logreg == 1
        },
        "neural_network": {
            "prediction": pred_mlp,
            "probability": prob_mlp,
            "churn": pred_mlp == 1
        }
    }

if __name__ == "__main__":
    # Test prediction
    test_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    print("Testing predictions with sample customer data:")
    results = predict_churn(test_customer)
    for model, res in results.items():
        print(f"{model.upper()}:")
        print(f"  Churn probability: {res['probability']:.4%}")
        print(f"  Predicts Churn:    {res['churn']}")
        print("-" * 30)
