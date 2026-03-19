import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def run_migration_ablation(file_path, results_dir, graphs_dir):
    print("🔬 Starting Migration Tracking Ablation Study...")
    
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    if len(df_cohort) < 50: df_cohort = df.copy()

    weekly_data = df_cohort.set_index('obsDt').resample('W')['howMany'].sum().reset_index()
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    
    # Feature Engineering
    df_xgb = weekly_data.copy()
    df_xgb['Month'] = df_xgb['obsDt'].dt.month
    df_xgb['WeekOfYear'] = df_xgb['obsDt'].dt.isocalendar().week.astype(int)
    df_xgb['Lag_1_Week'] = df_xgb['howMany'].shift(1)
    df_xgb['Lag_2_Week'] = df_xgb['howMany'].shift(2)
    df_xgb['Rolling_Mean_4W'] = df_xgb['howMany'].rolling(window=4).mean()
    df_xgb = df_xgb.dropna().reset_index(drop=True)
    
    train_size = int(len(df_xgb) * 0.8)
    train, test = df_xgb.iloc[:train_size], df_xgb.iloc[train_size:]
    target = 'howMany'
    
    results = {}
    
    # Model 1: Prophet Baseline (Hardcoded from your previous results for speed)
    results['1. Prophet Baseline'] = 107.00
    
    # Model 2: XGBoost (No Lags, Standard Objective)
    features_no_lags = ['Month', 'WeekOfYear']
    model_no_lags = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
    model_no_lags.fit(train[features_no_lags], train[target])
    results['2. XGBoost (No Lags)'] = calculate_mape(test[target], model_no_lags.predict(test[features_no_lags]))
    
    # Model 3: XGBoost (With Lags, Standard Objective)
    features_lags = ['Month', 'WeekOfYear', 'Lag_1_Week', 'Lag_2_Week', 'Rolling_Mean_4W']
    model_lags = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
    model_lags.fit(train[features_lags], train[target])
    results['3. XGBoost (With Lags)'] = calculate_mape(test[target], model_lags.predict(test[features_lags]))
    
    # Model 4: Final XGBoost (With Lags + Tweedie)
    model_final = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.1, n_estimators=100, learning_rate=0.05)
    model_final.fit(train[features_lags], train[target])
    results['4. Final Model (Lags + Tweedie)'] = calculate_mape(test[target], model_final.predict(test[features_lags]))
    
    # Plotting the Ablation Results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['lightgrey', 'salmon', 'lightcoral', 'teal'])
    plt.title('Ablation Study: Migration Forecasting Error (MAPE)', fontsize=14)
    plt.ylabel('Mean Absolute Percentage Error (%) - Lower is Better')
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "migration_ablation_study.png"), dpi=300)
    print("✅ Saved 'migration_ablation_study.png' to graphs folder.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    RESULTS_DIRECTORY = os.path.join(BASE_DIR, "results")
    
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    run_migration_ablation(CLEANED_DATA_PATH, RESULTS_DIRECTORY, GRAPHS_DIRECTORY)