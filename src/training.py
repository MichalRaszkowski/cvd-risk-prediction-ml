from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data, add_features
from src.model import train_model, save_model
from src.evaluate import evaluate_model, evaluate_model_extended
from src.tuning import tune_hyperparameters, find_best_threshold


def run_training_pipeline(data_path="data/heart.csv"):
    df = load_data(data_path)
    df = add_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
        X_train, X_val, X_test
    )

    model = train_model(X_train_scaled, y_train)

    best_model = tune_hyperparameters(X_train_scaled, y_train)
    evaluate_model(best_model, X_val_scaled, y_val, "Validation (best model)")

    y_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    best_threshold, results = find_best_threshold(y_val, y_proba)

    metrics_train = evaluate_model_extended(best_model, X_train_scaled, y_train, "Train", threshold=best_threshold)
    metrics_val = evaluate_model_extended(best_model, X_val_scaled, y_val, "Validation", threshold=best_threshold)
    metrics_test = evaluate_model_extended(best_model, X_test_scaled, y_test, "Test", threshold=best_threshold)

    save_model(best_model, scaler)

    return {
        "model": best_model,
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
