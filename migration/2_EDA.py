import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMapWithTime
import warnings

warnings.filterwarnings('ignore')

def run_exploratory_data_analysis(file_path, graphs_dir):
    print("🚀 Starting Phase 2: Exploratory Data Analysis & Spatiotemporal Baselines...")
    
    # 1. Load the cleaned data
    df = pd.read_csv(file_path)
    df['obsDt'] = pd.to_datetime(df['obsDt'])
    
    # 2. Define our Migratory Waterfowl Cohort
    waterfowl_cohort = [
        'lewduc1', 'bahgoo', 'gargan', 'isbduc1', 'norpin', 
        'gadwal', 'eurwig', 'norsho', 'comteal'
    ]
    
    print("\n🦅 Isolating data for Migratory Waterfowl Cohort (Ducks & Geese)...")
    df_target = df[df['speciesCode'].isin(waterfowl_cohort)].copy()
    
    print(f"Cohort data points available: {len(df_target)}")
    
    # 3. Dynamic Fallback Strategy
    if len(df_target) < 100:
        print("⚠️ Warning: Cohort data is still too sparse for deep learning.")
        print("🔄 Pivoting to 'Overall Avian Observation Volume' to maintain model viability...")
        df_target = df.copy() # Fallback to using the entire dataset
        target_title = "Overall Bird Population"
    else:
        target_title = "Migratory Waterfowl Cohort"
        
    # 4. WEEKLY Time-Series Trend
    print("Generating Weekly Time-Series Trend...")
    weekly_trend = df_target.groupby(['Year', 'Week'])['howMany'].sum().reset_index()
    
    weekly_trend['Timeline'] = weekly_trend['Year'].astype(str) + "-W" + weekly_trend['Week'].astype(str).str.zfill(2)
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=weekly_trend, x='Timeline', y='howMany', marker='o', color='teal', linewidth=2)
    plt.xticks(rotation=45)
    
    # Showing only every 4th label to keep the x-axis clean
    for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if i % 4 != 0: label.set_visible(False)
        
    plt.title(f"Weekly Observation Volume of {target_title} (2023-Present)", fontsize=14)
    plt.xlabel("Timeline (Year-Week)", fontsize=12)
    plt.ylabel("Total Birds Observed", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "weekly_migration_trend.png"), dpi=300)
    print("✅ Saved 'weekly_migration_trend.png'")
    
    # 5. Spatiotemporal Heatmap Animation
    print("Generating Spatiotemporal Heatmap Animation...")
    df_target['YearMonth'] = df_target['obsDt'].dt.strftime('%Y-%m')
    months = sorted(df_target['YearMonth'].unique())
    
    heat_data = []
    for month in months:
        month_data = df_target[df_target['YearMonth'] == month]
        heat_data.append([[row['lat'], row['lng'], row['howMany']] for index, row in month_data.iterrows()])
    
    # Create base map centered on India
    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles='CartoDB Positron')
    
    HeatMapWithTime(
        heat_data, 
        index=months,
        auto_play=True, 
        radius=15, 
        max_opacity=0.8
    ).add_to(m)
    
    m.save(os.path.join(graphs_dir, "animated_migration_heatmap.html"))
    print("✅ Saved 'animated_migration_heatmap.html'")
    
if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    os.chdir(GRAPHS_DIRECTORY)
    
    run_exploratory_data_analysis(CLEANED_DATA_PATH, GRAPHS_DIRECTORY)
    
    os.chdir(BASE_DIR)