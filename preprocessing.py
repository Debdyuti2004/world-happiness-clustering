# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    """
    Loads the World Happiness dataset and keeps selected columns.
    Uses the actual column names found in the file.
    """
    filepath = "data/world_happiness.csv"
    
    columns = ['Country or region', 'GDP per capita', 'Social support',
               'Healthy life expectancy', 'Freedom to make life choices',
               'Perceptions of corruption']
    
    df = pd.read_csv(filepath)
    df = df[columns]
    df = df.dropna()
    
    print("✅ Data loaded successfully!")
    print("Number of rows:", len(df))
    return df

def scale_features(df):
    """
    Scales numerical features using StandardScaler.
    Returns the scaled feature array and the scaler.
    """
    X = df.iloc[:, 1:].values  # exclude 'Country or region'
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features scaled successfully!")
    return X_scaled, scaler
