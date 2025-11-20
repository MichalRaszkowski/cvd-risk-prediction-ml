# Cardiovascular Disease (CVD) Risk Prediction

## Overview
This project focuses on predicting the risk of cardiovascular disease (CVD) using Machine Learning techniques. The goal is to classify patients into low or high-risk groups based on medical attributes such as age, chest pain, blood pressure, and more.

The solution includes a full ML pipeline: **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **Model Training (Logistic Regression)**, **Evaluation**, and an interactive **Web Interface** built with Streamlit.

## Key Features
* **Robust Preprocessing:** Handles scaling using `RobustScaler` and engineered features (e.g., `chol_age_ratio`).
* **Interpretable Model:** Uses Logistic Regression with balanced class weights to handle data distribution effectively.
* **Comprehensive Evaluation:** Used metrics such as Accuracy, F1-score, ROC-AUC, Specificity, Recall, Brier Score, and more.
* **Interactive Demo:** A user-friendly app to simulate predictions in real-time.

## Dataset & Analysis
The project uses a heart disease dataset (approx. 300 records) containing 14 attributes.
The dataset comes from the UCI Machine Learning Repository (Heart Disease dataset).

## Project Structure
```text
cvd-risk-prediction-ml/
├── data/
│   └── heart.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── notebooks/
│   └── data-analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── tuning.py
│   ├── evaluate.py
│   ├── analysis.py
│   ├── training.py
│   └── models/
│       ├── logistic_regression.py
│       └── random_forest.py
├── app.py
├── main.py
├── .gitignore
└── README.md
```

# Installation & Usage
1. Clone the repository
```
git clone https://github.com/MichalRaszkowski/cvd-risk-prediction-ml.git
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Train the Model
To retrain the model and generate new .pkl files:
```
python -m main
This will save the trained model and scaler into the models/ directory.
```
4. Run the Web App
```
streamlit run app.py
```
