import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
from prophet import Prophet
import hdbscan
import warnings

warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error safely."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0 
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def run_research_evaluations(file_path, results_dir):
    print("📊 Starting Phase 5: Rigorous Model Evaluation...")
    
    # 1. Load Data
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    
    if len(df_cohort) < 50:
        df_cohort = df.copy() # Fallback

    # TEMPORAL FORECASTING EVALUATION (Prophet)
    print("⏳ Calculating Temporal Metrics (80/20 Train-Test Split)...")
    weekly_data = df_cohort.set_index('obsDt').resample('W')['howMany'].sum().reset_index()
    weekly_data['howMany'] = weekly_data['howMany'].fillna(0)
    weekly_data.rename(columns={'obsDt': 'ds', 'howMany': 'y'}, inplace=True)

    # Split data to test prediction accuracy
    train_size = int(len(weekly_data) * 0.8)
    train, test = weekly_data.iloc[:train_size], weekly_data.iloc[train_size:]

    # Train model on 80%
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)
    
    # Predict the remaining 20%
    future = m.make_future_dataframe(periods=len(test), freq='W')
    forecast = m.predict(future)
    
    predictions = forecast['yhat'].iloc[-len(test):].values
    actuals = test['y'].values

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = calculate_mape(actuals, predictions)

    # SPATIAL CLUSTERING EVALUATION (HDBSCAN)
    print("📍 Calculating Spatial Metrics (Silhouette Score)...")
    coords = df_cohort[['lat', 'lng']].dropna().values
    coords_radians = np.radians(coords)

    hdb = hdbscan.HDBSCAN(min_cluster_size=15, metric='haversine', cluster_selection_method='eom')
    labels = hdb.fit_predict(coords_radians)

    # Calculate Silhouette score only on valid clusters (ignoring noise/in-flight points)
    valid_idx = labels != -1
    if len(set(labels[valid_idx])) > 1:
        sil_score = silhouette_score(coords_radians[valid_idx], labels[valid_idx], metric='haversine')
    else:
        sil_score = 0.0

    noise_ratio = list(labels).count(-1) / len(labels) * 100
    total_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # EXPORT FOR RESEARCH PAPER
    export_file = os.path.join(results_dir, "research_metrics.txt")
    print(f"\n📝 Exporting formatting results to '{export_file}'...")
    
    with open(export_file, "w") as f:
        f.write("====================================================\n")
        f.write("   WILDLIFE CONSERVATION AI: RESEARCH METRICS\n")
        f.write("====================================================\n\n")
        
        f.write("1. TEMPORAL FORECASTING ACCURACY (Prophet)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"RMSE (Root Mean Squared Error): {rmse:.2f} observations\n")
        f.write(f"MAE (Mean Absolute Error):      {mae:.2f} observations\n")
        f.write(f"MAPE (Mean Absolute % Error):   {mape:.2f}%\n")
        f.write("*Note: Evaluated on an 80/20 chronological split.\n\n")
        
        f.write("2. SPATIAL CLUSTERING COHESION (HDBSCAN)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Total Stopover Zones Identified: {total_clusters}\n")
        f.write(f"Silhouette Score (-1 to 1):      {sil_score:.4f}\n")
        f.write(f"Noise Ratio (In-Flight points):  {noise_ratio:.2f}%\n")
        f.write("*Note: Silhouette score calculated using Haversine distance.\n")

    print("✅ Success! Metrics for Research Paper generated.")

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    RESULTS_DIRECTORY = os.path.join(BASE_DIR, "results")
    
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    os.chdir(RESULTS_DIRECTORY)
    
    run_research_evaluations(CLEANED_DATA_PATH, RESULTS_DIRECTORY)
    
    os.chdir(BASE_DIR)