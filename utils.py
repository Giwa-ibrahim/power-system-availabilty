"""
Utility functions for power availability prediction and allocation
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model


def load_data(data_path="cleaned_data/processed_data.csv"):
    """Load and prepare the dataset"""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_prediction_model(model_path="model_metrics/BiLSTM Hyperband.keras"):
    """Load the trained BiLSTM model"""
    return load_model(model_path)


def setup_preprocessing(df):
    """
    Setup label encoders and scalers for prediction
    
    Returns:
        tuple: (le_feeder, le_district, scaler_X, scaler_y)
    """
    le_feeder = LabelEncoder()
    le_district = LabelEncoder()
    
    # Encode categorical features
    data_copy = df.copy()
    data_copy['feeder_name'] = le_feeder.fit_transform(data_copy['feeder_name'])
    data_copy['district'] = le_district.fit_transform(data_copy['district'])
    
    # Prepare features (matching training configuration)
    features = ['feeder_name', 'district', 'consumption_mwh', 
                'lag1_avail', 'lag2_avail', 'lag3_avail', 'yesterday_full']
    
    X = data_copy[features]
    y = data_copy['availability_hrs']
    
    # Fit scalers on training portion (80/20 split)
    split_idx = int(len(X) * 0.8)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(X[:split_idx])
    scaler_y.fit(y[:split_idx].values.reshape(-1, 1))
    
    return le_feeder, le_district, scaler_X, scaler_y


def prepare_prediction_input(df, feeder_name, le_feeder, le_district, time_steps=7):
    """
    Prepare input features for prediction
    
    Args:
        df: DataFrame with historical data
        feeder_name: Name of the feeder to predict
        le_feeder: Fitted LabelEncoder for feeder names
        le_district: Fitted LabelEncoder for districts
        time_steps: Number of timesteps to use (default: 7)
    
    Returns:
        numpy array: Prepared features of shape (time_steps, 7)
    """
    feeder_history = df[df["feeder_name"] == feeder_name].tail(time_steps).copy()
    
    if len(feeder_history) < time_steps:
        raise ValueError(f"Insufficient data. Need {time_steps} days, got {len(feeder_history)}")
    
    # Encode categorical features
    feeder_encoded = le_feeder.transform(feeder_history['feeder_name'])
    district_encoded = le_district.transform(feeder_history['district'])
    
    # Stack features in correct order
    X_input = np.column_stack([
        feeder_encoded,
        district_encoded,
        feeder_history['consumption_mwh'].values,
        feeder_history['lag1_avail'].values,
        feeder_history['lag2_avail'].values,
        feeder_history['lag3_avail'].values,
        feeder_history['yesterday_full'].values
    ])
    
    return X_input


def predict_availability(model, X_input, scaler_X, scaler_y, time_steps=7):
    """
    Predict availability hours using the trained model
    
    Args:
        model: Trained Keras model
        X_input: Input features array
        scaler_X: Fitted MinMaxScaler for features
        scaler_y: Fitted MinMaxScaler for target
        time_steps: Number of timesteps (default: 7)
    
    Returns:
        int: Predicted availability hours (1-24)
    """
    # Scale and reshape
    X_scaled = scaler_X.transform(X_input)
    X_scaled = X_scaled.reshape(1, time_steps, 7)
    
    # Predict
    pred_scaled = model.predict(X_scaled, verbose=0)
    pred_value = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    # Clip and round
    return int(np.round(np.clip(pred_value, 1, 24)))


def allocate_energy(available_hours, total_supply, avg_consumption):
    """
    Allocate energy across feeder types based on availability hours
    
    Args:
        available_hours: Number of hours power is available
        total_supply: Total energy supply in MW
        avg_consumption: Series with average consumption by feeder type
    
    Returns:
        DataFrame: Hourly allocation for each feeder type (24 hours x feeder types)
    """
    # Define priority windows
    time_windows = [
        (0, 5, ["Healthcare", "Residential"]),
        (5, 9, ["Healthcare", "Residential", "Commercial"]),
        (9, 12, ["Healthcare", "Industrial", "Commercial"]),
        (12, 15, ["Healthcare", "Industrial"]),
        (15, 18, ["Healthcare", "Industrial", "Commercial"]),
        (18, 23, ["Healthcare", "Residential", "Commercial"]),
        (23, 24, ["Healthcare", "Residential"]),
    ]
    
    # Initialize allocation table
    all_feeders = sorted({f for _, _, feeders in time_windows for f in feeders})
    hourly_allocation = pd.DataFrame(0.0, index=range(24), columns=all_feeders)
    
    allocated_hours = 0
    
    # Allocate energy for available hours
    for start, end, active_feeders in time_windows:
        for hour in range(start, end):
            if allocated_hours >= available_hours:
                break
            
            # Healthcare priority: 40% if present
            if "Healthcare" in active_feeders:
                hourly_allocation.loc[hour, "Healthcare"] = 0.4 * total_supply
                remaining_supply = 0.6 * total_supply
            else:
                remaining_supply = total_supply
            
            # Proportional distribution for others
            others = [f for f in active_feeders if f != "Healthcare"]
            if others and remaining_supply > 0:
                others_consumption = avg_consumption[others]
                total_consumption = others_consumption.sum()
                
                for feeder in others:
                    weight = others_consumption[feeder] / total_consumption
                    hourly_allocation.loc[hour, feeder] = weight * remaining_supply
            
            allocated_hours += 1
    
    return hourly_allocation


def format_allocation_display(hourly_allocation):
    """Format allocation table for display with time labels"""
    display = hourly_allocation.copy()
    display.index = [f"{h:02d}:00" for h in display.index]
    display.index.name = "Time of Day"
    return display
