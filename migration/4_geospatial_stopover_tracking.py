import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from pykalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')

def load_and_prep_geospatial_data(file_path):
    print("🌍 Loading geospatial coordinate data...")
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    
    # 1. Isolate the exact same Waterfowl Cohort
    waterfowl_cohort = ['lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 'gadwal', 'eurwig', 'norsho', 'comteal']
    df_cohort = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    
    if len(df_cohort) < 50:
        df_target = df.copy() # Fallback
    else:
        df_target = df_cohort
        
    # Sort strictly by time so the Kalman Filter tracks chronological movement
    df_target = df_target.sort_values(by='obsDt').reset_index(drop=True)
    return df_target

def apply_hdbscan_stopover_clustering(df, graphs_dir):
    print("📍 Running HDBSCAN to identify migration stopovers...")
    
    # 1. Extract coordinates and convert to radians for Haversine (Earth surface) distance
    coords = df[['lat', 'lng']].dropna().values
    coords_radians = np.radians(coords)
    
    # 2. Configure HDBSCAN
    # min_cluster_size determines how many sightings constitute a "stopover"
    hdb = hdbscan.HDBSCAN(min_cluster_size=15, metric='haversine', cluster_selection_method='eom')
    df['Stopover_Cluster'] = hdb.fit_predict(coords_radians)
    
    # 3. Plot the Clusters (XAI Spatial Explainability)
    plt.figure(figsize=(10, 8))
    
    # Plot Noise (Cluster -1) in grey
    noise = df[df['Stopover_Cluster'] == -1]
    plt.scatter(noise['lng'], noise['lat'], color='lightgrey', s=10, alpha=0.5, label='In-Flight / Noise')
    
    # Plot actual Stopovers
    clusters = df[df['Stopover_Cluster'] != -1]
    sns.scatterplot(data=clusters, x='lng', y='lat', hue='Stopover_Cluster', palette='tab10', s=30, edgecolor='k')
    
    plt.title("HDBSCAN Migration Stopover Zones (Waterfowl Cohort)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Stopover ID")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "hdbscan_stopovers.png"), dpi=300)
    print("✅ Saved 'hdbscan_stopovers.png'")
    
    return df

def apply_kalman_trajectory_tracking(df, graphs_dir):
    print("✈️ Applying Kalman Filter for trajectory smoothing...")
    
    # 1. Extract chronological coordinates
    # We take a sample or average per day to keep the trajectory clean
    daily_coords = df.groupby(df['obsDt'].dt.date)[['lat', 'lng']].mean().dropna().values
    
    if len(daily_coords) < 10:
        print("⚠️ Not enough chronological coordinate days for a meaningful Kalman path.")
        return
        
    # 2. Initialize Kalman Filter
    kf = KalmanFilter(initial_state_mean=daily_coords[0], n_dim_obs=2)
    
    # Em-algorithm optimizes the filter parameters based on our specific data
    kf = kf.em(daily_coords, n_iter=10)
    
    # Smooth the trajectory
    smoothed_states, state_covariances = kf.smooth(daily_coords)
    
    # 3. Plot Raw vs Smoothed Trajectory
    plt.figure(figsize=(10, 8))
    
    # Plot raw daily averages
    plt.plot(daily_coords[:, 1], daily_coords[:, 0], 'o', color='teal', alpha=0.4, label='Raw Daily Sightings')
    
    # Plot smoothed Kalman path
    plt.plot(smoothed_states[:, 1], smoothed_states[:, 0], '-', color='darkorange', linewidth=3, label='Kalman Smoothed Flight Path')
    
    # Add start/end markers
    plt.plot(smoothed_states[0, 1], smoothed_states[0, 0], 'g*', markersize=15, label='Migration Start')
    plt.plot(smoothed_states[-1, 1], smoothed_states[-1, 0], 'rX', markersize=12, label='Current/End Location')

    plt.title("Kalman Filter: Predicted Migration Trajectory", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "kalman_trajectory.png"), dpi=300)
    print("✅ Saved 'kalman_trajectory.png'")

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    df_geo = load_and_prep_geospatial_data(CLEANED_DATA_PATH)
    
    os.chdir(GRAPHS_DIRECTORY)
    
    df_geo_clustered = apply_hdbscan_stopover_clustering(df_geo, GRAPHS_DIRECTORY)
    apply_kalman_trajectory_tracking(df_geo_clustered, GRAPHS_DIRECTORY)
    
    os.chdir(BASE_DIR)