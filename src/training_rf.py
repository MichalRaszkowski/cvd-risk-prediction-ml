from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data, add_features
from src.models.random_forest import train_random_forest, save_model_rf
from src.evaluate import evaluate_model, evaluate_model_extended
from src.tuning import tune_hyperparameters_rf
from src.visualize_rf import visualize_rf


def run_training_pipeline_rf(data_path="data/heart.csv"):
    df = load_data(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train = add_features(X_train)
    X_val = add_features(X_val)
    X_test = add_features(X_test)
    
    best_params = tune_hyperparameters_rf(X_train, y_train)

    model = train_random_forest(X_train, y_train, best_params)

    visualize_rf(model, X_train)

    evaluate_model(model, X_train, y_train, "Train RF")
    evaluate_model(model, X_val, y_val, "Validation RF")
    evaluate_model(model, X_test, y_test, "Test RF")

    save_model_rf(model)

    return model
