import os
import pandas as pd

def generate_migration_tables(results_dir):
    print("📊 Generating Migration Thesis Tables...")
    os.makedirs(results_dir, exist_ok=True)

    # Table 1: Feature Engineering Lexicon
    feature_data = {
        "Feature Name": ["Month", "WeekOfYear", "Lag_1_Week", "Lag_2_Week", "Rolling_Mean_4W"],
        "Datatype": ["Integer", "Integer", "Integer", "Integer", "Float"],
        "Mathematical Operation": ["Datetime Extraction", "ISO Calendar Extraction", "t - 1 Shift", "t - 2 Shift", "4-Period Moving Average"],
        "Ecological Rationale": [
            "Captures macro-seasonality and broad migratory windows.",
            "Captures micro-seasonality and precise arrival timing.",
            "Models immediate arrival trends and flock momentum.",
            "Captures multi-week transit delays across the subcontinent.",
            "Smooths out variance and noise inherent in crowdsourced citizen-science data."
        ]
    }
    df_features = pd.DataFrame(feature_data)
    feature_path = os.path.join(results_dir, "table_1_feature_engineering.csv")
    df_features.to_csv(feature_path, index=False)
    print(f"✅ Saved Feature Engineering table to: {feature_path}")

    # Table 2: Hyperparameter Tuning Grid
    hyper_data = {
        "Algorithm": ["XGBoost"] * 5,
        "Parameter": ["objective", "learning_rate (eta)", "max_depth", "n_estimators", "tweedie_variance_power"],
        "Search Space": ["reg:squarederror, reg:tweedie", "[0.01, 0.05, 0.1]", "[3, 5, 7]", "[50, 100, 200]", "[1.1, 1.5, 1.9]"],
        "Final Selected Value": ["reg:tweedie", "0.05", "5", "100", "1.1"],
        "Selection Rationale": [
            "Optimized for zero-inflated, highly skewed count data.",
            "Lower learning rate prevented overshooting the global minimum.",
            "Prevented the model from overfitting to specific temporal outliers.",
            "Provided sufficient boosting rounds for convergence.",
            "Power of 1.1 best modeled the specific variance of the waterfowl cohort."
        ]
    }
    df_hyper = pd.DataFrame(hyper_data)
    hyper_path = os.path.join(results_dir, "table_2_hyperparameters.csv")
    df_hyper.to_csv(hyper_path, index=False)
    print(f"✅ Saved Hyperparameter Grid table to: {hyper_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIRECTORY = os.path.join(BASE_DIR, "results")
    generate_migration_tables(RESULTS_DIRECTORY)