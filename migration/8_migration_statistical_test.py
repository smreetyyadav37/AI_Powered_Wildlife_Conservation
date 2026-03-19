import os
import pandas as pd
import numpy as np
from scipy import stats
import xgboost as xgb
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def run_statistical_significance_test(file_path):
    print("🔬 Running Paired T-Test for Statistical Significance...")
    
    # 1. Prep Data (Same as ablation)
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    if len(df_cohort) < 50: df_cohort = df.copy()

    weekly_data = df_cohort.set_index('obsDt').resample('W')['howMany'].sum().reset_index()
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    
    # Features for XGBoost
    df_xgb = weekly_data.copy()
    df_xgb['Month'] = df_xgb['obsDt'].dt.month
    df_xgb['WeekOfYear'] = df_xgb['obsDt'].dt.isocalendar().week.astype(int)
    df_xgb['Lag_1_Week'] = df_xgb['howMany'].shift(1)
    df_xgb['Lag_2_Week'] = df_xgb['howMany'].shift(2)
    df_xgb['Rolling_Mean_4W'] = df_xgb['howMany'].rolling(window=4).mean()
    df_xgb = df_xgb.dropna().reset_index(drop=True)
    
    train_size = int(len(df_xgb) * 0.8)
    train_xgb, test_xgb = df_xgb.iloc[:train_size], df_xgb.iloc[train_size:]
    features_xgb = ['Month', 'WeekOfYear', 'Lag_1_Week', 'Lag_2_Week', 'Rolling_Mean_4W']
    
    # Features for Prophet
    df_prophet = df_xgb[['obsDt', 'howMany']].rename(columns={'obsDt': 'ds', 'howMany': 'y'})
    train_prophet, test_prophet = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

    # 2. Train Models
    print("Training Prophet (Baseline)...")
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train_prophet)
    future = m.make_future_dataframe(periods=len(test_prophet), freq='W')
    prophet_preds = m.predict(future)['yhat'].iloc[-len(test_prophet):].values

    print("Training XGBoost Tweedie (Final)...")
    model_xgb = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.1, n_estimators=100, learning_rate=0.05)
    model_xgb.fit(train_xgb[features_xgb], train_xgb['howMany'])
    xgb_preds = model_xgb.predict(test_xgb[features_xgb])

    actuals = test_xgb['howMany'].values

    # 3. Calculate Absolute Errors
    errors_prophet = np.abs(actuals - prophet_preds)
    errors_xgb = np.abs(actuals - xgb_preds)

    # 4. Perform Paired T-Test
    t_stat, p_value = stats.ttest_rel(errors_prophet, errors_xgb)

    print("\n====================================================")
    print(" 📊 STATISTICAL SIGNIFICANCE RESULTS (MIGRATION)")
    print("====================================================")
    print(f"Prophet Mean Absolute Error: {np.mean(errors_prophet):.2f}")
    print(f"XGBoost Mean Absolute Error: {np.mean(errors_xgb):.2f}")
    print("----------------------------------------------------")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.6f}")
    
    if p_value < 0.05:
        print("\n✅ CONCLUSION: The p-value is < 0.05.")
        print("The XGBoost model is STATISTICALLY SIGNIFICANTLY better than Prophet.")
    else:
        print("\n⚠️ CONCLUSION: The p-value is >= 0.05.")
        print("The difference in performance is not statistically significant.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CLEANED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "Migration_Cleaned.csv")
    run_statistical_significance_test(CLEANED_DATA_PATH)