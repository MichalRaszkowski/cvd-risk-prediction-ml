# src/logistic_regression.py

import joblib
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X, y, params):
    model = LogisticRegression(**params, max_iter=500)
    model.fit(X, y)
    return model


def save_model(model, scaler=None, path_model="models/logistic_regression.pkl", path_scaler="models/scaler.pkl"):

    joblib.dump(model, path_model)
    if scaler is not None:
        joblib.dump(scaler, path_scaler)
    print(f"Model saved to {path_model}")
    if scaler:
        print(f"Scaler saved to {path_scaler}")

