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

## 🧠 Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## ⚖️ Handling Imbalanced Data

- Stratified splitting
- Class weights (where supported)
- F1-based threshold tuning
- Evaluation using Precision-Recall curves

---

## 🧾 Requirements

```
Python 3.10+
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
imbalanced-learn
mlflow
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Ahmed Essam**  
AI Engineer & CS Student  
🔗 GitHub: [your link]  
🔗 LinkedIn: [your link]

---

## 📄 License

This project is licensed under the **MIT License**.
