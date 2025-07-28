from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(criterion="log_loss",n_estimators=100,class_weight={0:1,1:6}, random_state=42,max_features="log2",max_depth=9),
        "XGBoost": XGBClassifier(use_label_encoder=False,class_weight={0:1,1:10}, eval_metric="logloss", random_state=42,max_depth=9, n_estimators=100, learning_rate=0.1)
    }
