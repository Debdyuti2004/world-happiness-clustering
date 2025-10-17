import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    """
    Loads the World Happiness dataset, keeps the selected columns,
    and removes any missing values.
    """
    filepath = "data/world_happiness.csv"
    
    columns = [
        'Country or region', 
        'GDP per capita', 
        'Social support',
        'Healthy life expectancy', 
        'Freedom to make life choices',
        'Perceptions of corruption'
    ]
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Keep only relevant columns if they exist
    df = df[[col for col in columns if col in df.columns]]
    
    # Drop missing values
    df = df.dropna()
    
    print("✅ Data loaded successfully!")
    print(f"Number of rows: {len(df)}")
    return df


def scale_features(df):
    """
    Scales only numeric features using StandardScaler.
    Ignores non-numeric columns automatically.
    Returns the scaled feature array and the scaler.
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for scaling.")
    
    X = df[numeric_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✅ Features scaled successfully!")
    return X_scaled, scaler, numeric_cols  # return column names for reference

