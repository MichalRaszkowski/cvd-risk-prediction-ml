from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data, add_features
from src.models.logistic_regression import train_logistic_regression, save_model
from src.evaluate import evaluate_model_extended
from src.tuning import tune_hyperparameters, find_best_threshold
import numpy as np


def run_training_pipeline(data_path="data/heart.csv"):
    '''
     Runs the training pipeline for logistic regression.
    '''
    
    df = load_data(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train = add_features(X_train)
    X_val = add_features(X_val)
    X_test = add_features(X_test)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
        X_train, X_val, X_test
    )

    X_train_val_scaled = np.vstack((X_train_scaled, X_val_scaled))
    y_train_val = np.concatenate((y_train, y_val))

    best_params = tune_hyperparameters(X_train_val_scaled, y_train_val)
    
    model = train_logistic_regression(X_train_val_scaled, y_train_val, best_params)

    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    best_threshold, results = find_best_threshold(y_val, y_val_proba)
    print(f"Best threshold found: {best_threshold}")

    print("Final evaluation on all splits with best threshold:")

    metrics_train = evaluate_model_extended(model, X_train_scaled, y_train, "Train", threshold=best_threshold)
    metrics_val = evaluate_model_extended(model, X_val_scaled, y_val, "Validation", threshold=best_threshold)
    metrics_test = evaluate_model_extended(model, X_test_scaled, y_test, "Test", threshold=best_threshold)

    save_model(model, scaler)


    return {
        "model": model,
        "scaler": scaler,
        "threshold": best_threshold,
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test
        },
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "X_test_scaled": X_test_scaled,
        }
    }

