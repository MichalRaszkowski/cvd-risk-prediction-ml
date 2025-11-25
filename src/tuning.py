# src/tuning.py

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def tune_hyperparameters(X_train, y_train):
    '''
    Tunes hyperparameters for logistic regression using GridSearchCV.
    '''
    model = LogisticRegression(max_iter=3000, solver='liblinear')

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'class_weight': ['balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best F1 score:", grid.best_score_)

    return grid.best_params_

def find_best_threshold(y_true, y_proba, start=0.1, stop=0.9, step=0.05):
    '''
    Finds the best classification threshold based on F1 score.
    '''
    thresholds = np.arange(start, stop, step)
    f1_scores = {}

    best_thr = None
    best_f1 = -1

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        f1_scores[t] = f1

        print(f"threshold={t:.2f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    print(f"\nBest threshold: {best_thr:.2f} with F1={best_f1:.4f}")

    return best_thr, f1_scores


def tune_hyperparameters_rf(X_train, y_train):
    '''
    Tunes hyperparameters for Random Forest using GridSearchCV.
    '''
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3],
        "bootstrap": [True],
        "max_features": ["sqrt", "log2"]
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best RF Params:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)

    return grid.best_params_
