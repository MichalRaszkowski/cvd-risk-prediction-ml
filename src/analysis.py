# src/analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix


def analyze_feature_importance(model, X_train):
    '''
    Analyzes and visualizes feature importance for a logistic regression model.
    model - trained logistic regression model
    X_train - training features as a DataFrame
    '''
    feature_names = X_train.columns
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Importance": np.abs(coefficients)
    }).sort_values(by="Absolute_Importance", ascending=False)

    print("\nTop features influencing heart disease prediction")
    print(importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Coefficient"])
    plt.xlabel("Coefficient value")
    plt.title("Feature importance (Logistic Regression)")
    plt.gca().invert_yaxis()
    plt.show()

    return importance_df


def analyze_errors(model, X_val, X_val_scaled, y_val, threshold=0.5):
    '''
    Analyzes prediction errors on the validation set.
    model - trained logistic regression model
    X_val - validation features as a DataFrame
    X_val_scaled - scaled validation features as a numpy array
    y_val - true validation target values as a Series
    threshold - classification threshold
    '''
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    errors_df = X_val.copy()
    errors_df["y_true"] = y_val.values
    errors_df["y_pred"] = y_pred
    errors_df["y_proba"] = y_proba

    def classify(row):
        if row["y_true"] == 1 and row["y_pred"] == 0:
            return "False Negative"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "False Positive"
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "True Positive"
        return "True Negative"

    errors_df["error_type"] = errors_df.apply(classify, axis=1)

    print(errors_df["error_type"].value_counts())

    return errors_df

def full_analysis(model, X_train, X_val, X_val_scaled, y_val, threshold=0.5):
    '''
    Performs full analysis: feature importance and error analysis.
    '''
    print("\nFEATURE IMPORTANCE")
    importance_df = analyze_feature_importance(model, X_train)

    print("\nERROR ANALYSIS")
    errors_df = analyze_errors(model, X_val, X_val_scaled, y_val, threshold)

    return {
        "importance": importance_df,
        "errors": errors_df,
    }
