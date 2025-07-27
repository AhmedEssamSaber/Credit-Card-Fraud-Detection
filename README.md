# 💳 Credit Card Fraud Detection

This project is a comprehensive machine learning pipeline for detecting fraudulent credit card transactions using multiple supervised learning models including Logistic Regression, Random Forest, and XGBoost. It supports experiment tracking with MLflow, modular code structure, and various evaluation visualizations.

---

## 📁 Project Structure

credit-card-fraud-detection/
│
├── config.py                 # Configuration settings for paths and parameters
├── preprocess.py            # Data preprocessing and feature engineering
├── train.py                 # Model training script (basic)
├── train_with_mlflow.py     # Model training with MLflow experiment tracking
├── models/                  # Folder containing model definitions
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── eval.py                  # Evaluation functions (metrics, plots, threshold tuning)
├── utils.py                 # Utility functions for saving/loading models and plots
├── results/                 # Folder where all models, plots, and metrics are saved
├── requirements.txt         # Python dependencies
└── README.md                # This file

---

## 📊 Dataset

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Transactions: 284,807
- Fraudulent: 492 (~0.17%)
- Features: 30 (V1–V28 PCA, Time, Amount)
- Target: Class (0 → Legitimate, 1 → Fraudulent)

The dataset is pre-split into:
- train.csv
- val.csv
- test.csv

These paths must be configured in config.py.

---

## 🔧 How to Use

### 1. Clone the Repository

$ git clone https://github.com/your-username/credit-card-fraud-detection.git
$ cd credit-card-fraud-detection

### 2. Install Dependencies

$ pip install -r requirements.txt

### 3. Set File Paths in config.py

Example:

TRAIN_PATH = "data/train.csv"
VAL_PATH   = "data/val.csv"
TEST_PATH  = "data/test.csv"

---

## 🚀 Run Training

### Basic Training (All Models)

$ python train.py

### Training with MLflow Tracking

$ python train_with_mlflow.py

---

## 📈 Evaluation

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report
- Threshold Optimization using F1 Score

All plots and models are saved in:

results/<ModelName>/
├── model.pkl
├── metrics.txt
└── plots/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── precision_recall_curve.png

---

## 🧪 MLflow Tracking

To launch the MLflow dashboard:

$ mlflow ui

Open in browser: http://localhost:5000

---

## 📌 Sample Model Scores

| Model              | Accuracy | F1 Score | ROC AUC |
|-------------------|----------|----------|---------|
| Logistic Regression | 0.998   | 0.87     | 0.95    |
| Random Forest       | 0.999   | 0.91     | 0.98    |
| XGBoost             | 0.999   | 0.93     | 0.99    |

---

## ⚠️ Class Imbalance Handling

- Class weights (`class_weight='balanced'`)
- Stratified train/val/test splits
- Threshold tuning using precision-recall tradeoff

---

## ✅ Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- mlflow

---

## 👨‍💻 Author

Ahmed Essam  
AI Engineer | CS & IS Student  
[GitHub](https://github.com) | [LinkedIn](https://www.linkedin.com)

---

## 📃 License

This project is licensed under the MIT License.
