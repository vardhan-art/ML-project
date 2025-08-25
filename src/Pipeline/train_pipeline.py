import pandas as pd
import numpy as np

# Data Cleaning and Preprocessing
def preprocess_data(df):
    try:
        # Rename columns to a consistent format
        df.rename(columns={
            'Timestamp': 'timestamp',
            'Energy_Consumption(kWh)': 'energy_consumption',
            'Peak_Hours': 'peak_hours',
            'Appliance_Usage(kWh)': 'appliance_usage',
            'Thermostat_Setting(°C)': 'thermostat_setting',
            'Smart_Lighting(kWh)': 'smart_lighting',
            'Temperature(°C)': 'temperature'
        }, inplace=True)
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # --- Create new features from existing data ---
        
        # Lag Features: Use consumption from the previous hour as a predictor
        df['energy_consumption_lag_1'] = df['energy_consumption'].shift(1)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical features for Hour and Day of Week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6.0)
        
        # Binary features
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_alert'] = df.apply(lambda row: 1 if row['temperature'] > 28 and row['peak_hours'] == 1 else 0, axis=1)

        # Interaction Features: combine features that might have a synergistic effect
        df['temp_occupancy_inter'] = df['temperature'] * df['occupancy']
        df['appliance_peak_inter'] = df['appliance_usage'] * df['peak_hours']
        df['smartlight_hour_inter'] = df['smart_lighting'] * df['hour']

        # Polynomial Features: to capture non-linear relationships
        df['temperature_sq'] = df['temperature'] ** 2
        df['appliance_sq'] = df['appliance_usage'] ** 2

        # Drop the first row with NaN from the lag feature
        df.dropna(inplace=True)
        
        return df

    except Exception as e:
        # Assuming you have a CustomException class
        raise CustomException(e, "Error during data preprocessing.")