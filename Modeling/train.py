from config import TRAIN_PATH, VAL_PATH, TEST_PATH
from utils import load_data, resample_data, create_model_dirs, save_model
from preprocess import convert_time_features, scale_and_select_features
from models import get_models
from eval import full_model_evaluation
import warnings

warnings.filterwarnings("ignore")


def main():
    # Load data
    train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

    # Feature Engineering
    train_df = convert_time_features(train_df)
    val_df = convert_time_features(val_df)
    test_df = convert_time_features(test_df)

    # Separate features and labels
    X_train, y_train = train_df.drop("Class", axis=1), train_df["Class"]
    X_val, y_val = val_df.drop("Class", axis=1), val_df["Class"]

    # === Scale & Select Features ===
    X_train, scaler = scale_and_select_features(X_train)
    X_val = scaler.transform(X_val)  # Use the same scaler for validation

    # Resample train data
    #X_train_res, y_train_res = resample_data(X_train, y_train)  : optional

    # Train and evaluate
    models = get_models()

    for name, model in models.items():
        print(f"\n====== Training: {name} ======")
        model.fit(X_train, y_train)

        # Create directories to save results
        base_dir, save_dirs = create_model_dirs(name)

        # === Train Evaluation ===
        eval_results = full_model_evaluation(
            model,
            X_train,
            y_train,
            title=f"{name}_Train",
            save_dirs=save_dirs
        )

        # === Validation Evaluation with same threshold ===
        full_model_evaluation(
            model,
            X_val,
            y_val,
            title=f"{name}_Validation",
            save_dirs=save_dirs,
            optimal_threshold=eval_results['optimal_threshold']  
        )

        # Save model + threshold + scaler
        threshold = eval_results['optimal_threshold']
        save_model(model, threshold, scaler, model_save_name=name)


if __name__ == "__main__":
    main()
