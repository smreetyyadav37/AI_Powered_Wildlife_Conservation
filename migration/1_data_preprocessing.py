import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def build_clean_migration_dataset(file_path, output_path):
    print("🚀 Starting Phase 1: Data Engineering & Feature Extraction...")
    
    # 1. Load the raw data directly from Excel
    print(f"Loading raw observations from {file_path}...")
    try:
        df = pd.read_excel(file_path, sheet_name="Observations")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}. Please check the exact path.")
        return None
    except ValueError:
        print(f"❌ Error: Sheet 'Observations' not found in the Excel file.")
        return None
        
    # 2. Data Cleaning & Validation
    print("Cleaning data and filtering invalid records...")
    if 'obsValid' in df.columns:
        df = df[df['obsValid'] == True]
    
    # Drop rows missing crucial kinematic or temporal data
    df = df.dropna(subset=['lat', 'lng', 'obsDt'])
    
    # Drop exact duplicates to prevent model bias
    df = df.drop_duplicates()
    
    # 3. Temporal Feature Engineering (Crucial for LSTM & Prophet XAI)
    print("Extracting temporal features...")
    df['obsDt'] = pd.to_datetime(df['obsDt'], errors='coerce')
    df = df.dropna(subset=['obsDt']) 
    
    df['Year'] = df['obsDt'].dt.year
    df['Month'] = df['obsDt'].dt.month
    df['Week'] = df['obsDt'].dt.isocalendar().week
    df['DayOfWeek'] = df['obsDt'].dt.dayofweek
    
    # Define Meteorological Seasons for India
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring' 
        elif month in [6, 7, 8, 9]: return 'Monsoon'
        else: return 'Autumn' 
        
    df['Season'] = df['Month'].apply(get_season)
    
    # 4. Geospatial Feature Engineering
    df['lat_rounded'] = df['lat'].round(3)
    df['lng_rounded'] = df['lng'].round(3)
    
    # 5. Handle Observation Volumes
    if 'howMany' in df.columns:
        df['howMany'] = pd.to_numeric(df['howMany'], errors='coerce').fillna(1)
    else:
        df['howMany'] = 1
        
    # 6. Final Feature Selection
    columns_to_keep = [
        'speciesCode', 'comName', 'obsDt', 'Year', 'Month', 'Week', 'DayOfWeek', 
        'Season', 'lat', 'lng', 'lat_rounded', 'lng_rounded', 'howMany'
    ]
    
    # Safely select columns that actually exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df_clean = df[existing_columns]
    
    # 7. Export to the single source of truth file
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Phase 1 Complete! Cleaned dataset saved to: {output_path}")
    print(f"📊 Total processed records ready for modeling: {len(df_clean)}")
    
    return df_clean

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIRECTORY = os.path.join(BASE_DIR, "dataset")
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    RAW_DATA_PATH = os.path.join(DATA_DIRECTORY, "India_Bird_Data.xlsx")
    CLEAN_DATA_PATH = os.path.join(DATA_DIRECTORY, "Migration_Cleaned.csv")
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    df_cleaned = build_clean_migration_dataset(RAW_DATA_PATH, CLEAN_DATA_PATH)
    
    if df_cleaned is not None:
        print("\nPreview of the engineered data:")
        print(df_cleaned.head())