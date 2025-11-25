# src/visualize_rf.py

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

''' 
Visualizes Random Forest model characteristics.
'''

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importance - Random Forest")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_oob_error(model):

    oob_errors = []
    for i in range(1, len(model.estimators_)+1):
        temp_model = type(model)(**model.get_params())
        temp_model.n_estimators = i
        temp_model.random_state = 42
        temp_model.oob_score = True
        temp_model.estimators_ = model.estimators_[:i]
        oob_errors.append(1 - model.oob_score_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(model.estimators_)+1), oob_errors, marker='o')
    plt.xlabel("Number of Trees")
    plt.ylabel("OOB Error")
    plt.title("OOB Error Across Trees")
    plt.show()


def plot_example_tree(model, feature_names, tree_index=0, max_depth=3):
    plt.figure(figsize=(20,10))
    plot_tree(
        model.estimators_[tree_index],
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10
    )
    plt.title(f"Example Tree #{tree_index}")
    plt.show()


def visualize_rf(model, X_train):
    plot_feature_importance(model, X_train.columns)
    plot_example_tree(model, X_train.columns)
    if hasattr(model, "oob_score_") and model.oob_score_:
        plot_oob_error(model)
