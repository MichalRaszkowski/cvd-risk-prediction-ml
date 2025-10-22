from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model, save_model

df = load_data("data/heart.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_val, X_test)

model = train_model(X_train_scaled, y_train)

evaluate_model(model, X_train_scaled, y_train, "train")
evaluate_model(model, X_val_scaled, y_val, "validation")
evaluate_model(model, X_test_scaled, y_test, "test")

save_model(model, scaler)