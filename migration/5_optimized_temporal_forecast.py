import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def run_xgboost_optimization(file_path, graphs_dir):
    print("🚀 Starting Phase 3b: XGBoost Tweedie Optimization...")
    
    # 1. Load and Prep Data
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    
    if len(df_cohort) < 50: df_cohort = df.copy()

    # Resample to weekly
    weekly_data = df_cohort.set_index('obsDt').resample('W')['howMany'].sum().reset_index()
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    
    # 2. Advanced Feature Engineering for XGBoost
    print("⚙️ Engineering time-lag features...")
    df_xgb = weekly_data.copy()
    df_xgb['Month'] = df_xgb['obsDt'].dt.month
    df_xgb['WeekOfYear'] = df_xgb['obsDt'].dt.isocalendar().week.astype(int)
    
    # Create lag features (What was the population 1, 2, and 3 weeks ago?)
    df_xgb['Lag_1_Week'] = df_xgb['howMany'].shift(1)
    df_xgb['Lag_2_Week'] = df_xgb['howMany'].shift(2)
    df_xgb['Lag_3_Week'] = df_xgb['howMany'].shift(3)
    
    # Create rolling averages to smooth out noise
    df_xgb['Rolling_Mean_4W'] = df_xgb['howMany'].rolling(window=4).mean()
    
    # Drop NaNs created by lagging
    df_xgb = df_xgb.dropna().reset_index(drop=True)
    
    # 3. Train-Test Split (80/20 Chronological)
    train_size = int(len(df_xgb) * 0.8)
    train, test = df_xgb.iloc[:train_size], df_xgb.iloc[train_size:]
    
    features = ['Month', 'WeekOfYear', 'Lag_1_Week', 'Lag_2_Week', 'Lag_3_Week', 'Rolling_Mean_4W']
    target = 'howMany'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # 4. Train XGBoost with Tweedie Objective
    print("🧠 Training XGBoost Regressor (Tweedie Objective)...")
    # Tweedie variance power typically ranges from 1 to 2 for count data
    model = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    # 5. Predict and Evaluate
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = calculate_mape(y_test, predictions)
    
    print("\n====================================================")
    print(" 🏆 XGBOOST OPTIMIZATION RESULTS")
    print("====================================================")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}% (Compare this to Prophet's 107.00%)")
    
    # 6. Plot the Optimized Forecast
    plt.figure(figsize=(12, 5))
    plt.plot(df_xgb['obsDt'], df_xgb['howMany'], label="Actual Observations", color='teal', alpha=0.6)
    plt.plot(test['obsDt'], predictions, label="XGBoost Tweedie Prediction", color='red', linewidth=2)
    plt.title("Optimized Temporal Forecasting (XGBoost)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Bird Observations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "xgboost_optimized_forecast.png"), dpi=300)
    print("✅ Saved 'xgboost_optimized_forecast.png'")
    
    # 7. Extract XAI with SHAP
    print("\n🔍 Generating SHAP Explainability Graph...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Feature Importance: What drives a migration spike?", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "xgboost_shap_xai.png"), dpi=300)
    print("✅ Saved 'xgboost_shap_xai.png'")
    
if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    os.chdir(GRAPHS_DIRECTORY)
    
    run_xgboost_optimization(CLEANED_DATA_PATH, GRAPHS_DIRECTORY)
    
    os.chdir(BASE_DIR)