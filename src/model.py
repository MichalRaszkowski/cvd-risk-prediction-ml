# src/model.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name="dataset"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"Metrics for {dataset_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion matrix:\n", cm, "\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

def save_model(model, scaler=None, path_model="models/logistic_regression.pkl", path_scaler="models/scaler.pkl"):

    joblib.dump(model, path_model)
    if scaler is not None:
        joblib.dump(scaler, path_scaler)
    print(f"Model saved to {path_model}")
    if scaler:
        print(f"Scaler saved to {path_scaler}")
