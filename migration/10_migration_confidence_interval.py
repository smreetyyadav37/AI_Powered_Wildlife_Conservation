import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def generate_forecast_with_intervals(file_path, output_dir):
    print("📈 Generating Time-Series Forecast with Confidence Intervals...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Standard Data Prep
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    if len(df_cohort) < 50: df_cohort = df.copy()

    weekly_data = df_cohort.set_index('obsDt').resample('W')['howMany'].sum().reset_index()
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    
    df_xgb = weekly_data.copy()
    df_xgb['Month'] = df_xgb['obsDt'].dt.month
    df_xgb['WeekOfYear'] = df_xgb['obsDt'].dt.isocalendar().week.astype(int)
    df_xgb['Lag_1_Week'] = df_xgb['howMany'].shift(1)
    df_xgb['Lag_2_Week'] = df_xgb['howMany'].shift(2)
    df_xgb['Rolling_Mean_4W'] = df_xgb['howMany'].rolling(window=4).mean()
    df_xgb = df_xgb.dropna().reset_index(drop=True)
    
    # Train/Test Split
    train_size = int(len(df_xgb) * 0.8)
    train_xgb, test_xgb = df_xgb.iloc[:train_size], df_xgb.iloc[train_size:]
    features = ['Month', 'WeekOfYear', 'Lag_1_Week', 'Lag_2_Week', 'Rolling_Mean_4W']
    
    # 2. Train Model and Predict
    model = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.1, n_estimators=100, learning_rate=0.05)
    model.fit(train_xgb[features], train_xgb['howMany'])
    
    predictions = model.predict(test_xgb[features])
    actuals = test_xgb['howMany'].values
    
    # 3. Calculate Historical Volatility for Confidence Intervals
    # We use the standard deviation of the training residuals to estimate future uncertainty
    train_preds = model.predict(train_xgb[features])
    residuals = train_xgb['howMany'] - train_preds
    std_dev = np.std(residuals)
    
    upper_bound = predictions + (1.96 * std_dev) # 95% Confidence Interval
    lower_bound = np.maximum(0, predictions - (1.96 * std_dev)) # Can't have negative birds

    # 4. Plotting
    plt.figure(figsize=(14, 6))
    
    # Plot Actuals vs Predictions
    plt.plot(test_xgb['obsDt'], actuals, label='Actual Observations', color='black', marker='.', linestyle='dashed', alpha=0.7)
    plt.plot(test_xgb['obsDt'], predictions, label='XGBoost Forecast', color='teal', linewidth=2)
    
    # Shaded Confidence Interval
    plt.fill_between(test_xgb['obsDt'], lower_bound, upper_bound, color='teal', alpha=0.2, label='95% Confidence Interval')
    
    plt.title("Waterfowl Migration Forecast: XGBoost-Tweedie with 95% Confidence Intervals", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Population Count", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "migration_confidence_intervals.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved strictly scientific forecast to: {save_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "Migration_Cleaned.csv")
    GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
    generate_forecast_with_intervals(DATA_PATH, GRAPHS_DIR)