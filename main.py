# main.py

from src.analysis import full_analysis
from src.training_lr import run_training_pipeline
from src.training_rf import run_training_pipeline_rf

def main():
    
    result = run_training_pipeline()
    
    #results_rf = run_training_pipeline_rf()
    
    analysis_results = full_analysis(
        model=result["model"],
        X_train=result["splits"]["X_train"],
        X_val=result["splits"]["X_val"],
        X_val_scaled=result["splits"]["X_val_scaled"],
        y_val=result["splits"]["y_val"],
        threshold=result["threshold"]
    )

if __name__ == "__main__":
    main()