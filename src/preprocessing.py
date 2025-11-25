# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE

def add_features(df):
    ''' 
    Adds engineered features to the DataFrame.
    '''
    df = df.copy()
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["bp_age_ratio"] = df["trestbps"] / df["age"]
    df["thalach_age_ratio"] = df["thalach"] / df["age"]
    
    df["hypertension"] = (df["trestbps"] > 130).astype(int)
    df["high_chol"] = (df["chol"] > 240).astype(int)

    return df

def preprocess_data(X_train, X_val, X_test):
    '''
    Scales the features using RobustScaler.
    '''
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
