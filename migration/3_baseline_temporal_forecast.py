import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap
import warnings
import os

warnings.filterwarnings('ignore')

def prepare_weekly_data(file_path):
    print("⏳ Loading and preparing time-series data...")
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    
    # 1. Cohort Logic (Matching EDA Phase)
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)]
    
    if len(df_cohort) < 100:
        print("⚠️ Cohort too sparse. Forecasting Overall Avian Biomass.")
        df_target = df.copy()
    else:
        print("🦆 Forecasting Migratory Waterfowl Cohort.")
        df_target = df_cohort.copy()

    # 2. Resample to strictly continuous Weekly intervals
    df_target.set_index('obsDt', inplace=True)
    weekly_data = df_target.resample('W')['howMany'].sum().reset_index()
    
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    
    weekly_data.rename(columns={'obsDt': 'ds', 'howMany': 'y'}, inplace=True)
    return weekly_data

def train_prophet_with_xai(df, graphs_dir):
    print("\n🚀 Training Prophet Model & Generating XAI Components...")
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    
    # Predict the next 12 weeks (approx 3 months)
    future = m.make_future_dataframe(periods=12, freq='W')
    forecast = m.predict(future)
    
    # 1. Plot standard forecast
    fig1 = m.plot(forecast)
    plt.title("Prophet Forecast: Bird Observation Volumes (12 Weeks Ahead)")
    plt.xlabel("Date")
    plt.ylabel("Observations")
    plt.savefig(os.path.join(graphs_dir, "prophet_forecast.png"), dpi=300, bbox_inches='tight')
    
    # 2. Extract XAI: Decompose the trend and yearly seasonality
    fig2 = m.plot_components(forecast)
    fig2.savefig(os.path.join(graphs_dir, "prophet_xai_components.png"), dpi=300, bbox_inches='tight')
    print("✅ Saved 'prophet_forecast.png' & 'prophet_xai_components.png'")

def train_lstm_forecaster(df, graphs_dir):
    print("\n🧠 Training LSTM Neural Network...")
    
    # 1. Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['y']].values)
    
    # 2. Create Sequences (Lookback window = 4 weeks to predict the 5th)
    lookback = 4
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    if len(X) < 10:
        print("⚠️ Sequence array too small for LSTM. More historical data required.")
        return
        
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 3. Build LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 4. Train Model
    print("Training in progress... (50 Epochs)")
    model.fit(X, y, epochs=50, batch_size=4, verbose=0)
    
    # 5. Evaluate Fit
    predictions = model.predict(X, verbose=0)
    preds_inversed = scaler.inverse_transform(predictions)
    actual_inversed = scaler.inverse_transform(y.reshape(-1, 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'].iloc[lookback:], actual_inversed, label='Actual Historical Data', color='teal')
    plt.plot(df['ds'].iloc[lookback:], preds_inversed, label='LSTM Fitted Prediction', color='darkorange', linestyle='--')
    plt.title("LSTM Model Fit: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Observation Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstm_model_fit.png", dpi=300)
    print("✅ Saved 'lstm_model_fit.png'")

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    weekly_data = prepare_weekly_data(CLEANED_DATA_PATH)
    
    os.chdir(GRAPHS_DIRECTORY)
    
    train_prophet_with_xai(weekly_data, GRAPHS_DIRECTORY)
    train_lstm_forecaster(weekly_data, GRAPHS_DIRECTORY)
    
    os.chdir(BASE_DIR)
    
  