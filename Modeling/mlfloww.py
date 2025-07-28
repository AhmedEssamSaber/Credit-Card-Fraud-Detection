import warnings
warnings.filterwarnings("ignore")

import os
import mlflow
import mlflow.sklearn

from config import TRAIN_PATH, VAL_PATH, TEST_PATH
from utils import load_data, load_model, create_model_dirs
from preprocess import convert_time_features
from eval import full_model_evaluation

mlflow.set_experiment("CreditCard_Fraud_Experiment")

def main():
    # Load and preprocess test data
    _, _, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)
    test_df = convert_time_features(test_df)
    X_test, y_test = test_df.drop("Class", axis=1), test_df["Class"]

    saved_models_dir = os.path.join(os.getcwd(), "Saved Model")

    for model_file in os.listdir(saved_models_dir):
        if not model_file.endswith(".pkl"):
            continue

        model_name = model_file.replace(".pkl", "")
        print(f"\n🚀 MLflow Test Run: {model_name}")

        with mlflow.start_run(run_name=f"{model_name}_Test"):
            mlflow.set_tag("phase", "test")
            mlflow.set_tag("model_name", model_name)

            # Load model + scaler + threshold
            model, threshold, _, scaler = load_model(model_file)
            X_test_scaled = scaler.transform(X_test)

            # Save dirs for plots and report
            base_dir, save_dirs = create_model_dirs(f"{model_name}_Test")

            # === Evaluate on Test ===
            test_results = full_model_evaluation(
                model,
                X_test_scaled,
                y_test,
                title=f"{model_name}_Test",
                save_dirs=save_dirs,
                optimal_threshold=threshold
            )

            # Log metrics
            for metric, value in test_results.items():
                mlflow.log_metric(f"test_{metric}", value)

            # Log artifacts
            for plot_type, path in save_dirs.items():
                for file in os.listdir(path):
                    mlflow.log_artifact(os.path.join(path, file))

            # Log classification report manually
            report_path = os.path.join(save_dirs["confusion_matrix"], f"{model_name}_Test_report.txt")
            with open(report_path, "w") as f:
                from sklearn.metrics import classification_report
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                y_preds = (y_probs >= threshold).astype(int)
                f.write(classification_report(y_test, y_preds))
            mlflow.log_artifact(report_path)

            print(f"✅ {model_name} test evaluation saved and logged to MLflow")

if __name__ == "__main__":
    main()