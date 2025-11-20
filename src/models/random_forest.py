# src/random_forest.py

from sklearn.ensemble import RandomForestClassifier
import joblib


def train_random_forest(X_train, y_train, params):
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model_rf(model, path="models/random_forest.pkl"):
    joblib.dump(model, path)
