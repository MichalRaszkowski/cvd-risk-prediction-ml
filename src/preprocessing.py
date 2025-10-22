# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_val, X_test):
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_test = X_test.astype(float)

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
