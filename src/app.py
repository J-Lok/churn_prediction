import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, f1_score, accuracy_score, roc_auc_score, auc

from src.predict import predict_churn, preprocess_single_input, load_prediction_artifacts

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Request Model
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

# Global cache for metrics and coordinates
_cached_metrics = None

def compute_metrics_cache():
    """
    Computes performance curves, confusion matrices, and feature importances
    to serve to the frontend dashboard.
    """
    global _cached_metrics
    if _cached_metrics is not None:
        return _cached_metrics

    # Load resources
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        return {}

    # Read and preprocess test data to run evaluations
    df = pd.read_csv(data_path, index_col='customerID')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", "", regex=False)
    df.dropna(inplace=True)
    
    feature_le = ["Partner", "Dependents", "PhoneService", "Churn", "PaperlessBilling"]
    for feature in feature_le:
        df[feature] = df[feature].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    
    features_ohe = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaymentMethod"
    ]
    df_ohe = pd.get_dummies(df, columns=features_ohe, dtype=int)
    
    features_mms = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_mms = pd.DataFrame(df_ohe, columns=features_mms)
    df_remaining = df_ohe.drop(columns=features_mms)
    
    # Load scaling and models
    load_prediction_artifacts()
    from src.predict import _scaler, _knn_model, _logreg_model, _mlp_model, _feature_columns
    
    rescaled_features = _scaler.transform(df_mms)
    rescaled_feature_df = pd.DataFrame(
        rescaled_features, 
        columns=features_mms, 
        index=df_remaining.index
    )
    
    df_final = pd.concat([rescaled_feature_df, df_remaining], axis=1)
    
    X = df_final.drop(columns="Churn")
    y = df_final["Churn"]
    
    # Run split exactly as train.py to get the test partition
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Subsampling utility for curves to keep JSON response sizes small
    def subsample_curve(x, y, n_points=50):
        if len(x) <= n_points:
            return x.tolist(), y.tolist()
        indices = np.linspace(0, len(x) - 1, n_points, dtype=int)
        return x[indices].tolist(), y[indices].tolist()

    results = {}
    
    # Compute KNN metrics
    y_pred_knn = _knn_model.predict(X_test)
    y_prob_knn = _knn_model.predict_proba(X_test)[:, 1]
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    prec_knn, rec_knn, _ = precision_recall_curve(y_test, y_prob_knn)
    
    sub_fpr_knn, sub_tpr_knn = subsample_curve(fpr_knn, tpr_knn)
    sub_rec_knn, sub_prec_knn = subsample_curve(rec_knn, prec_knn)
    
    results["knn"] = {
        "accuracy": accuracy_score(y_test, y_pred_knn),
        "f1_score": f1_score(y_test, y_pred_knn),
        "roc_auc": roc_auc_score(y_test, y_prob_knn),
        "pr_auc": auc(rec_knn, prec_knn),
        "confusion_matrix": cm_knn.tolist(), # [[TN, FP], [FN, TP]]
        "roc_curve": {"fpr": sub_fpr_knn, "tpr": sub_tpr_knn},
        "pr_curve": {"recall": sub_rec_knn, "precision": sub_prec_knn}
    }
    
    # Compute Logistic Regression metrics
    y_pred_lr = _logreg_model.predict(X_test)
    y_prob_lr = _logreg_model.predict_proba(X_test)[:, 1]
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    
    sub_fpr_lr, sub_tpr_lr = subsample_curve(fpr_lr, tpr_lr)
    sub_rec_lr, sub_prec_lr = subsample_curve(rec_lr, prec_lr)
    
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "f1_score": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_prob_lr),
        "pr_auc": auc(rec_lr, prec_lr),
        "confusion_matrix": cm_lr.tolist(),
        "roc_curve": {"fpr": sub_fpr_lr, "tpr": sub_tpr_lr},
        "pr_curve": {"recall": sub_rec_lr, "precision": sub_prec_lr}
    }
    
    # Compute MLP Neural Network metrics
    y_pred_mlp = _mlp_model.predict(X_test)
    y_prob_mlp = _mlp_model.predict_proba(X_test)[:, 1]
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
    prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, y_prob_mlp)
    
    sub_fpr_mlp, sub_tpr_mlp = subsample_curve(fpr_mlp, tpr_mlp)
    sub_rec_mlp, sub_prec_mlp = subsample_curve(rec_mlp, prec_mlp)
    
    results["neural_network"] = {
        "accuracy": accuracy_score(y_test, y_pred_mlp),
        "f1_score": f1_score(y_test, y_pred_mlp),
        "roc_auc": roc_auc_score(y_test, y_prob_mlp),
        "pr_auc": auc(rec_mlp, prec_mlp),
        "confusion_matrix": cm_mlp.tolist(),
        "roc_curve": {"fpr": sub_fpr_mlp, "tpr": sub_tpr_mlp},
        "pr_curve": {"recall": sub_rec_mlp, "precision": sub_prec_mlp}
    }
    
    # Extract Feature Importance (from Logistic Regression weights)
    weights = _logreg_model.coef_[0]
    features_weights = list(zip(_feature_columns, weights))
    # Sort weights by absolute value
    features_weights_sorted = sorted(features_weights, key=lambda x: abs(x[1]), reverse=True)
    
    results["feature_importances"] = [
        {"feature": feat, "weight": float(w)} for feat, w in features_weights_sorted[:15]
    ]
    
    _cached_metrics = results
    return _cached_metrics

@app.on_event("startup")
def startup_event():
    # Pre-load prediction artifacts and pre-compute performance metrics cache
    try:
        load_prediction_artifacts()
        compute_metrics_cache()
        print("Models loaded and evaluation metrics pre-cached successfully.")
    except Exception as e:
        print(f"Error loading models on startup: {e}")

# API Route: Predict
@app.post("/api/predict")
def run_prediction(input_data: CustomerInput):
    try:
        customer_dict = input_data.dict()
        prediction_results = predict_churn(customer_dict)
        return prediction_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Route: Metrics
@app.get("/api/metrics")
def get_metrics():
    try:
        metrics = compute_metrics_cache()
        if not metrics:
            raise HTTPException(status_code=404, detail="Metrics cache not initialized. Ensure data is present.")
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serving Frontend Index
@app.get("/")
def read_root():
    static_index = os.path.join("src", "static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    raise HTTPException(status_code=404, detail="Index HTML not found in src/static/")

# Mount Static Files Directory
app.mount("/static", StaticFiles(directory=os.path.join("src", "static")), name="static")
