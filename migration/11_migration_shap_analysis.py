import os
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def generate_migration_shap(file_path, output_dir):
    print("🧠 Generating SHAP Game-Theoretic Analysis for XGBoost...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data Prep (Identical to your training pipeline)
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
    
    features = ['Month', 'WeekOfYear', 'Lag_1_Week', 'Lag_2_Week', 'Rolling_Mean_4W']
    X = df_xgb[features]
    y = df_xgb['howMany']

    # 2. Train the Final Model
    model = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.1, n_estimators=100, learning_rate=0.05)
    model.fit(X, y)

    # 3. Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 4. Plot 1: SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary: Feature Impact on Migration Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "migration_shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved 'migration_shap_summary.png'")

    # 5. Plot 2: SHAP Dependence Plot (Lag_1_Week vs Rolling Mean)
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("Lag_1_Week", shap_values, X, interaction_index="Rolling_Mean_4W", show=False)
    plt.title("SHAP Dependence: 1-Week Lag vs Rolling Mean", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "migration_shap_dependence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved 'migration_shap_dependence.png'")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "Migration_Cleaned.csv")
    GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
    generate_migration_shap(DATA_PATH, GRAPHS_DIR)