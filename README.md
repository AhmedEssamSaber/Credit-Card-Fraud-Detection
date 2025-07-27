# 💳 Credit Card Fraud Detection

This project builds a machine learning pipeline to detect fraudulent credit card transactions using classical ML models like Logistic Regression, Random Forest, and XGBoost. It also includes experiment tracking with MLflow and detailed evaluation.

---

## 📂 Project Structure

```bash
credit-card-fraud-detection/
│
├── config.py                # Configuration settings for paths and parameters
├── preprocess.py            # Data preprocessing and feature engineering
├── train.py                 # Model training script (basic)
├── train_with_mlflow.py     # Model training with MLflow experiment tracking
│
├── models/                  # Folder containing model definitions
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── eval.py                  # Evaluation functions (metrics, plots, threshold tuning)
├── utils.py                 # Utility functions for saving/loading models and plots
│
├── results/                 # Saved models, plots, metrics (organized by model)
│   ├── LogisticRegression/
│   │   ├── model.pkl
│   │   └── plots/
│   │       ├── confusion_matrix.png
│   │       ├── roc_curve.png
│   │       └── precision_recall_curve.png
│   ├── RandomForest/
│   └── XGBoost/
│
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision, Recall, F1 Score
- ROC AUC Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Threshold Optimization

---

## 🧪 MLflow Tracking

```bash
# Start MLflow dashboard
mlflow ui
```

Open your browser at: [http://localhost:5000](http://localhost:5000)

---
## 📌 Sample Model Scores

| Model               | Accuracy | F1 Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 0.9994   | 0.81     | 0.95    |
| Random Forest       | 0.9995   | 0.837    | 0.98    |
| XGBoost             | 0.9996   | 0.862    | 0.99    |

---

## 🧠 Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

Install with:

```bash
pip install -r requirements.txt
```

