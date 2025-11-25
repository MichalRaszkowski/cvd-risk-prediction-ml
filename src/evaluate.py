# src/evaluate.py

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss
)
import numpy as np

def evaluate_model(model, X, y, dataset_name="Test", threshold=0.5):
    '''
    Evaluates the model on given data and prints metrics - basic version.
    '''
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"\nMetrics for {dataset_name} (threshold={threshold})")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion matrix:\n", cm)


    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

#function below generated with help of chatgpt
def evaluate_model_extended(model, X, y, dataset_name="Test", threshold=0.5):
    '''
    Evaluates the model on given data and prints extended metrics.
    '''
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)  # TPR
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    npv = tn / (tn + fn)
    error_rate = (fp + fn) / (tp + tn + fp + fn)

    acc = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_proba)
    auc_pr = average_precision_score(y, y_proba)
    mcc = matthews_corrcoef(y, y_pred)
    gmean = np.sqrt(recall * specificity)
    brier = brier_score_loss(y, y_proba)

    bins = np.linspace(0, 1, 11)
    binids = np.digitize(y_proba, bins) - 1
    bin_acc = np.array([y[binids == i].mean() if np.any(binids == i) else 0 for i in range(len(bins))])
    bin_conf = np.array([y_proba[binids == i].mean() if np.any(binids == i) else 0 for i in range(len(bins))])
    bin_size = np.array([np.sum(binids == i) for i in range(len(bins))])
    weights = bin_size / np.sum(bin_size)
    ece = np.sum(weights * np.abs(bin_acc - bin_conf))
    mce = np.max(np.abs(bin_acc - bin_conf))

    cost_fn, cost_fp = 5, 1
    expected_cost = (cost_fn * fn + cost_fp * fp) / (tp + tn + fp + fn)

    print(f"\n=== Metrics for {dataset_name} (threshold={threshold}) ===")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision={precision:.3f}, Recall(TPR)={recall:.3f}, Specificity(TNR)={specificity:.3f}")
    print(f"FPR={fpr:.3f}, FNR={fnr:.3f}, NPV={npv:.3f}, Error Rate={error_rate:.3f}")
    print(f"Accuracy={acc:.3f}, Balanced Acc={balanced_acc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}, G-mean={gmean:.3f}")
    print(f"AUC ROC={auc_roc:.3f}, AUC PR={auc_pr:.3f}, Brier Score={brier:.3f}")
    print(f"ECE={ece:.3f}, MCE={mce:.3f}, Expected Cost={expected_cost:.3f}")

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision, "Recall": recall, "Specificity": specificity,
        "FPR": fpr, "FNR": fnr, "NPV": npv, "Error Rate": error_rate,
        "Accuracy": acc, "Balanced Acc": balanced_acc, "F1": f1,
        "MCC": mcc, "G-mean": gmean, "AUC PR": auc_pr, "AUC ROC": auc_roc,
        "Brier Score": brier, "ECE": ece, "MCE": mce, "Expected Cost": expected_cost
    }

