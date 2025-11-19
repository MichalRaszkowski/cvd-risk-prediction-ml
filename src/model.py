# src/model.py

import joblib
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, penalty='l2', C=0.1):
    model = LogisticRegression(
        max_iter=100,
        solver='liblinear',
        class_weight='balanced',
        penalty=penalty,
        C=C
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, scaler=None, path_model="models/logistic_regression.pkl", path_scaler="models/scaler.pkl"):

    joblib.dump(model, path_model)
    if scaler is not None:
        joblib.dump(scaler, path_scaler)
    print(f"Model saved to {path_model}")
    if scaler:
        print(f"Scaler saved to {path_scaler}")

