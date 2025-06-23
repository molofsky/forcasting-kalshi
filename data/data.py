import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Load and prepare the data
def prepare_kalshi_data(csv_path):
    df = pd.read_csv(csv_path)
    market_question = df['Market'].iloc[0]
    df['market_question'] = market_question
    
    # Convert Forecast column from percentage to float (0-1)
    df['price'] = df['Forecast'].str.rstrip('%').astype(float) / 100
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and get last price of each day (closing price)
    daily_df = df.groupby('date').agg({
        'price': 'last',
        'market_question': 'first'
    }).reset_index()
    
    # Sort by date
    daily_df = daily_df.sort_values('date')
    
    return daily_df

def engineer_features(df):
    # Price changes
    df['price_change_1d'] = df['price'].diff()
    df['price_change_3d'] = df['price'] - df['price'].shift(3)
    df['price_change_7d'] = df['price'] - df['price'].shift(7)
    
    # Rolling statistics
    df['rolling_mean_7d'] = df['price'].rolling(window=7).mean()
    df['rolling_std_7d'] = df['price'].rolling(window=7).std()
    
    # Price momentum and acceleration
    df['momentum_1d'] = df['price_change_1d'] / df['price'].shift(1)  # Normalized price change
    df['momentum_3d'] = df['price_change_3d'] / df['price'].shift(3)
    df['momentum_7d'] = df['price_change_7d'] / df['price'].shift(7)
    
    # Time since start
    df['time_since_start'] = (df['date'] - df['date'].min()).dt.days
    # df['time_since_start'] = (df['date'] - df['date'].min()).dt.days.astype(int)
    
    # Price momentum and acceleration
    df['price_acceleration'] = df['price_change_1d'].diff()
    
    # Volatility features
    df['volatility_ratio'] = df['rolling_std_7d'] / df['rolling_mean_7d']
    
    # Trend features
    df['above_7d_mean'] = (df['price'] > df['rolling_mean_7d']).astype(int)
    df['distance_from_7d_mean'] = abs(df['price'] - df['rolling_mean_7d'])
    
    # Price extremes
    df['price_7d_high'] = df['price'].rolling(7).max()
    df['price_7d_low'] = df['price'].rolling(7).min()
    
    # Technical indicators
    df['mean_crossover'] = (df['price'] > df['rolling_mean_7d']).astype(int)
    df['price_range_ratio'] = (df['price_7d_high'] - df['price_7d_low']) / df['rolling_mean_7d']
    
    # Additional recommended features
    df['volatility_trend'] = df['rolling_std_7d'].diff()  # Change in volatility
    df['days_to_7d_high'] = df.groupby((df['price'] == df['price_7d_high']).cumsum())['price'].cumcount()
    
    # Calculate target variable (percentage price change)
    df['next_close'] = df['price'].shift(-1)  # Close price of the next day
    df['target'] = (df['next_close'] - df['price']) / df['price']  # (Next Close - Current Close) / Current Close
    
    # Drop the next_close column as it's no longer needed
    df.drop(columns=['next_close'], inplace=True)
    
    return df

def process_all_kalshi_data(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, 'kalshi-chart-data-*.csv'))
    
    if not csv_files:
        print(f"No Kalshi CSV files found in {directory_path}")
        return
    
    all_processed_data = []
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        try:
            # Load and prepare daily data
            df = prepare_kalshi_data(csv_file)
            
            # Add market name
            market_name = csv_file.split('kalshi-chart-data-')[-1].replace('.csv', '')
            df['market'] = market_name
            
            # Add features
            df = engineer_features(df)
            all_processed_data.append(df)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if all_processed_data:
        combined_df = pd.concat(all_processed_data, ignore_index=True)
        
        # Drop rows with NaN values
        combined_df = combined_df.dropna()
        
        # Select final features
        features = [
            'market', 'market_question', 'date', 'price',
            'price_change_1d', 'price_change_3d', 'price_change_7d',
            'momentum_1d', 'momentum_3d', 'momentum_7d',
            'time_since_start',
            'rolling_mean_7d', 'rolling_std_7d', 'price_acceleration',
            'volatility_ratio', 'above_7d_mean', 'distance_from_7d_mean',
            'price_7d_high', 'price_7d_low', 'mean_crossover',
            'price_range_ratio', 'volatility_trend', 'days_to_7d_high', 'target'
        ]
        
        final_df = combined_df[features]
        
        # Save combined data
        final_df.to_csv('kalshi.csv', index=False)
        
        print(f"\nProcessed {len(csv_files)} files")
        print(f"Total rows in combined dataset: {len(final_df)}")
        print(f"Saved combined data to: kalshi.csv")
        
        return final_df
    else:
        print("No data was successfully processed")
        return None

if __name__ == "__main__":
    directory = 'Kalshi_CSVs'
    processed_data = process_all_kalshi_data(directory)