import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    auc,
    accuracy_score,
    f1_score,
)
from tabulate import tabulate

def full_model_evaluation(model, x, y_true, title="Model", save_dirs=None, optimal_threshold=None):
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Model '{title}' does not support predict_proba.")

    probs = model.predict_proba(x)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))

    if optimal_threshold is None:  # First time (training)
        if len(thresholds) > 0:
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal_threshold = thresholds[optimal_idx]
            print(f"🔍 Calculated Optimal Threshold = {optimal_threshold:.4f}")
        else:
            optimal_threshold = 0.5
            print("⚠️ No thresholds found. Using default = 0.5")

    # Apply threshold
    y_pred_opt = (probs >= optimal_threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred_opt)
    f1_opt = f1_score(y_true, y_pred_opt)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, probs)

    # Print summary table
    table = [[title, f"{acc:.4f}", f"{f1_opt:.4f}", f"{pr_auc:.4f}", f"{roc_auc:.4f}", f"{optimal_threshold:.4f}"]]
    headers = ["Model", "Accuracy", "F1 Score", "PR AUC", "ROC AUC", "Optimal Threshold"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("\n🔹 Classification Report:")
    print(classification_report(y_true, y_pred_opt))

    # Confusion Matrix
    plt.figure()
    cm = confusion_matrix(y_true, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig = plt.gcf()
    if save_dirs:
        path = os.path.join(save_dirs["confusion_matrix"], f"{title}_confusion_matrix.png")
        fig.savefig(path)
        print(f"📊 Saved Confusion Matrix to {path}")
    plt.close(fig)

    # PR Curve
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {title}")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    if save_dirs:
        path = os.path.join(save_dirs["precision_recall_curve"], f"{title}_pr_curve.png")
        fig.savefig(path)
        print(f"📊 Saved PR Curve to {path}")
    plt.close(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    if save_dirs:
        path = os.path.join(save_dirs["roc_curve"], f"{title}_roc_curve.png")
        fig.savefig(path)
        print(f"📊 Saved ROC Curve to {path}")
    plt.close(fig)

    # Precision & Recall vs Threshold
    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision", marker='o')
    plt.plot(thresholds, recall[:-1], label="Recall", marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Threshold - {title}")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    if save_dirs:
        path = os.path.join(save_dirs["precision_recall_threshold"], f"{title}_precision_recall_threshold.png")
        fig.savefig(path)
        print(f"📊 Saved Precision & Recall vs Threshold Plot to {path}")
    plt.close(fig)

    return {
        'optimal_threshold': optimal_threshold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'f1_optimal': f1_opt,
    }

