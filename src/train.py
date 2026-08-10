import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib

def main():
    print("Starting model training pipeline...")
    
    # 1. Load dataset
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path, index_col='customerID')
    print(f"Loaded dataset of shape: {df.shape}")
    
    # 2. Data Cleaning
    # Coerce TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Remove " (automatic)" from PaymentMethod to shorten labels
    df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", "", regex=False)
    
    # Drop missing values (which are rows with empty TotalCharges, representing 0.16% of rows)
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with missing values.")
    
    # 3. Feature Engineering - Label Encoding for binary features
    feature_le = ["Partner", "Dependents", "PhoneService", "Churn", "PaperlessBilling"]
    for feature in feature_le:
        df[feature] = df[feature].map({"Yes": 1, "No": 0})
    
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    
    # 4. Feature Engineering - One-Hot Encoding for categorical features with >2 categories
    features_ohe = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaymentMethod"
    ]
    df_ohe = pd.get_dummies(df, columns=features_ohe, dtype=int)
    
    # 5. Feature Engineering - Feature Scaling
    features_mms = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_mms = pd.DataFrame(df_ohe, columns=features_mms)
    df_remaining = df_ohe.drop(columns=features_mms)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(df_mms)
    
    rescaled_feature_df = pd.DataFrame(
        rescaled_features, 
        columns=features_mms, 
        index=df_remaining.index
    )
    
    # Reassemble dataframe with scaled features first
    df_final = pd.concat([rescaled_feature_df, df_remaining], axis=1)
    
    # Split into features X and target y
    X = df_final.drop(columns="Churn")
    y = df_final["Churn"]
    
    print(f"Final feature set shape: {X.shape}")
    
    # Save the order of features to align during prediction
    feature_columns = list(X.columns)
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 7. Model Training
    # Model A: K-Nearest Neighbors
    print("Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    
    # Model B: Logistic Regression
    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
    
    # Model C: Neural Network (MLPClassifier)
    print("Training Neural Network (MLPClassifier)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=1000, 
        activation='relu',
        solver='adam',
        early_stopping=True,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    
    # 8. Evaluation Summary
    print("\n" + "="*40)
    print("EVALUATION METRICS COMPARISON")
    print("="*40)
    for model_name, y_pred, y_prob in [
        ("K-Nearest Neighbors", y_pred_knn, y_prob_knn),
        ("Logistic Regression", y_pred_logreg, y_prob_logreg),
        ("Neural Network (MLP)", y_pred_mlp, y_prob_mlp)
    ]:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"{model_name}:")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC AUC   : {auc_score:.4f}")
        print("-" * 20)
        
    # 9. Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(knn, "models/knn_model.joblib")
    joblib.dump(logreg, "models/logreg_model.joblib")
    joblib.dump(mlp, "models/mlp_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_columns, "models/feature_columns.joblib")
    print("All models and preprocessors saved to the models/ directory.")

if __name__ == "__main__":
    main()
