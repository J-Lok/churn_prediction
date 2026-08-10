# ChurnRadar - Customer Churn Prediction ML Project

This project implements a complete Machine Learning and Data Analysis pipeline for Telco Customer Churn Prediction. It is based on the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` and follows the guidelines of the provided TP PDF instruction manual. 

It contains a detailed exploratory data analysis and model training Jupyter notebook, pipeline scripts to clean and automate model training/saving, and a premium web dashboard built with a FastAPI backend and a custom dark glassmorphic HTML/CSS/JS frontend.

## Project Structure

```text
churn/
├── notebooks/
│   └── churn_prediction.ipynb   # Complete step-by-step analysis & model training (in French)
├── src/
│   ├── static/
│   │   ├── index.html           # Dashboard UI Layout
│   │   ├── style.css            # Dark mode glassmorphic CSS styles
│   │   └── app.js               # Frontend Fetch APIs & Chart.js rendering
│   ├── train.py                 # Pipeline to preprocess data and train KNN, LogReg, and MLP models
│   ├── predict.py               # Preprocessing and predictions mapping for inference
│   └── app.py                   # FastAPI backend server
├── models/                      # Saved models, scaler, and column names (created during training)
│   ├── knn_model.joblib
│   ├── logreg_model.joblib
│   ├── mlp_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Customer dataset
├── requirements.txt             # Python packages
└── README.md                    # Project document
```

---

## Model Evaluation Results

After training the models on **80%** of the dataset and evaluating on the **20%** stratified test set, here are the comparative performance metrics:

| Model | Accuracy | F1-Score | ROC AUC | PR AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Principal) | **80.24%** | **60.51%** | **0.8348** | **0.6186** |
| **Neural Network (MLP)** | **79.67%** | **58.19%** | **0.8301** | **0.6151** |
| **K-Nearest Neighbors** | 75.05% | 53.63% | 0.7719 | 0.5282 |

### Key Takeaways
- The **Logistic Regression** and **Neural Network (MLP)** models outperform the baseline KNN model by a large margin (about 5% absolute accuracy and 7% F1-score increase).
- Logistic Regression is selected as the primary dashboard model due to its high interpretability and strong coefficient correlation scores.
- The MLP Neural Network represents the Deep Learning implementation (filling the missing section on page 21 of the PDF manual) and achieves comparable, robust results.

---

## Setup & Installation

### Prerequisite
Ensure you have Python 3.8+ installed.

### 1. Install Dependencies
Install all required libraries via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Train the Models
Run the training script to clean the data, train the three classifiers, output metrics, and save artifacts in the `models/` directory:
```bash
python src/train.py
```

### 3. Launch the Web Application
Start the FastAPI development server using `uvicorn`:
```bash
uvicorn src.app:app --reload --port 8000
```

### 4. Access the Dashboard
Open your web browser and navigate to:
```text
http://localhost:8000
```

---

## Dashboard Features

1. **Interactive Predictor Tab**: 
   - Enter details about a specific customer (demographics, services subscribed, contract types, billing preferences).
   - Click **Calculer le Risque d'Attrition** to receive instant churn risk probabilities from the three models.
   - Circular progress ring gauge adapts color in real-time (Green for safe, Orange for warning, Red for high risk).
   - Shows a customized breakdown of the **Key Influence Factors** (e.g. month-to-month contracts are positive risk factors, whereas online security and long term commitments act as retention/loyalty protectors).

2. **Model Performance Tab**:
   - Compares the performance of the three models with interactive **ROC Curves** and **Precision-Recall Curves** rendered with Chart.js.
   - Displays a comparative **Metrics Table** and dynamically generated **Confusion Matrices** (Vrais Positifs, Faux Positifs, Vrais Négatifs, Faux Négatifs).

3. **Dataset Insights Tab**:
   - Visualizes the top 12 global weights influencing customer churn in an interactive horizontal bar chart.
   - Explains the business implications of the features (e.g. Month-to-month contracts and Fiber Optic services are strong drivers of churn, whereas high tenure/loyalty serves as a major protective factor).
