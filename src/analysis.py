import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix


def analyze_feature_importance(model, X_train):
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

    print("\n=== Error Type Counts ===")
    print(errors_df["error_type"].value_counts())

    return errors_df


def analyze_thresholds(y_true, y_proba, start=0.1, stop=0.9, step=0.05):
    thresholds = np.arange(start, stop, step)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1 = f1_score(y_true, y_pred)
        results.append((t, f1, fn, fp))

    df = pd.DataFrame(results, columns=["Threshold", "F1", "FN", "FP"])

    print("\nThreshold Analysis")
    print(df)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Threshold"], df["F1"], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 score vs Threshold")
    plt.grid(True)
    plt.show()

    return df


def full_analysis(model, X_train, X_val, X_val_scaled, y_val, threshold=0.5):
    print("\nFEATURE IMPORTANCE")
    importance_df = analyze_feature_importance(model, X_train)

    print("\nERROR ANALYSIS")
    errors_df = analyze_errors(model, X_val, X_val_scaled, y_val, threshold)

    return {
        "importance": importance_df,
        "errors": errors_df,
    }
