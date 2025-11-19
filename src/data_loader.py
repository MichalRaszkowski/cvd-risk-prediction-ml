# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, shuffle=True, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
