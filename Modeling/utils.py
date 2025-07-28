import os
import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from config import RESAMPLING_TECHNIQUE
import pickle

# Load datasets
def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

# Resampling  : optional
# def resample_data(X, y, technique=RESAMPLING_TECHNIQUE):
#     if technique == "undersample":
#         return RandomUnderSampler().fit_resample(X, y)
#     elif technique == "smote":
#         return SMOTE().fit_resample(X, y)
#     elif technique == "smotetomek":
#         return SMOTETomek().fit_resample(X, y)
#     return X, y

def create_model_dirs(model_name):
    base_dir = os.path.join("results", model_name.replace(" ", "_"))
    cm_dir = os.path.join(base_dir, "confusion_matrix")
    roc_dir = os.path.join(base_dir, "roc_curve")
    pr_dir = os.path.join(base_dir, "precision_recall_curve")
    prt_dir = os.path.join(base_dir, "precision_recall_threshold")

    for dir_path in [base_dir, cm_dir, roc_dir, pr_dir, prt_dir]:
        os.makedirs(dir_path, exist_ok=True)

    return base_dir, {
        "confusion_matrix": cm_dir,
        "roc_curve": roc_dir,
        "precision_recall_curve": pr_dir,
        "precision_recall_threshold": prt_dir,
    }


def save_model(model, threshold, scaler, model_save_name="model.pkl"):
    model_dict = {
        "model": model,
        "threshold": threshold if threshold is not None else "default",
        "model_name": model_save_name,
        "scaler": scaler
    }

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(CURRENT_DIR, "Saved Model")  

    os.makedirs(save_dir, exist_ok=True)

    if not model_save_name.endswith('.pkl'):
        model_save_name += '.pkl'

    model_path = os.path.join(save_dir, model_save_name)

    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)

    print(f"✅ {model_save_name} saved successfully in {model_path}")


def load_model(model_file_name):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(CURRENT_DIR, "Saved Model") 
    model_path = os.path.join(model_dir, model_file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ {model_path} does not exist.")

    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    model = model_dict["model"]
    threshold = model_dict.get("threshold", None)
    model_name = model_dict["model_name"].replace(".pkl", "")
    scaler = model_dict.get("scaler")

    print(f"✅ {model_name} model loaded successfully")
    return model, threshold, model_name, scaler
