# 💳 Credit Card Fraud Detection

This project builds a machine learning pipeline to detect fraudulent credit card transactions using classical ML models like Logistic Regression, Random Forest, and XGBoost. It also includes experiment tracking with MLflow and detailed evaluation.

---
# Project structure
---
```bash
credit-card-fraud-detection/
│
├── Modiling/                            # Source code
│   ├── config.py                        # Configuration settings for paths and parameters
│   ├── preprocess.py                    # Data preprocessing and feature engineering
│   ├── train.py                         # Model training script (basic)
│   ├── mlfloww.py                       # Model training with MLflow experiment tracking
│   ├── eval.py                          # Evaluation functions (metrics, plots, threshold tuning)
│   ├── utils.py                         # Utility functions for saving/loading models and plots
│   │
│   └── models/                          # Folder containing model definitions
│       ├── logistic_regression.py
│       ├── random_forest.py
│       └── xgboost_model.py
│
├── datasets/                            # Folder containing datasets
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── EDA/                                 # Exploratory Data Analysis (notebooks or scripts)
│   ├── eda_notebook.ipynb
│
├── report/                              # Final report files (PDF/DOCX/LaTeX)
│   ├── credit_fraud_report.pdf
│
├── results/                             # Saved models, plots, metrics (organized by model)
│   ├── LogisticRegression/
│   │   ├── model.pkl
│   │   ├── confusion_matrix/
│   │   │   └── confusion_matrix.png
│   │   ├── roc_curve/
│   │   │   └── roc_curve.png
│   │   ├── precision_recall_curve/
│   │   │   └── precision_recall_curve.png
│   │   ├── precision_recall_threshold/
│   │   │   └── precision_recall_threshold.png
│   │   └── report/
│   │       └── report.txt
│   │
│   ├── RandomForest/
│   │   └── ...
│   │
│   └── XGBoost/
│       └── ...
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project overview (this file)
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
---
## 📌 Validation Model Scores 

| Model               | Accuracy | F1 Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 0.9994   | 0.80     | 0.98    |
| Random Forest       | 0.9995   | 0.851    | 0.98    |
| XGBoost             | 0.9996   | 0.871    | 0.99    |

---


Install requirements with:

```bash
pip install -r requirements.txt
```

