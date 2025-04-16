import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import sys

# Import config and model loader
from flood.config import STATE_MAPPING, MONTH_MAPPING, labelencoder_X_1, onehotencoder, sc_X
sys.path.append(os.path.abspath("./model"))
from flood.load import init

# Initialize model
model, graph = init()

def get_state_and_terrain(state_code):
    """Get full state name and terrain from state code"""
    if state_code in STATE_MAPPING:
        return STATE_MAPPING[state_code]["full_name"], STATE_MAPPING[state_code]["terrain"]
    return "Unknown State", "Unknown Terrain"

def get_month_details(month_num):
    """Get quarter and duration from month number"""
    if month_num in MONTH_MAPPING:
        return MONTH_MAPPING[month_num]["quarter"], MONTH_MAPPING[month_num]["duration"]
    return "Unknown Quarter", 3  # Default duration of 3 months

def normalize_precipitation(precipitation, duration):
    """Normalize precipitation over the duration"""
    return precipitation * 30 * duration

def get_historical_data(state, year, quarter, terrain):
    """Get historical flood data for years <= 2015"""
    dataset = pd.read_csv('data/flood_past.csv')
    dataset = dataset[dataset['YEAR'] > 1979]
    dataset = dataset.dropna()
    dataset = dataset.iloc[:, [0, 1, 3, 4, 6, 8]]
    
    sd = dataset['SUBDIVISION'] == state
    yr = dataset['YEAR'] == int(year)
    qr = dataset['QUARTER'] == quarter
    tr = dataset['TERRAIN'] == terrain
    
    filtered_data = dataset[sd & yr & qr & tr]
    
    if len(filtered_data) > 0:
        precipitation = float(filtered_data['PRECIPITATION'].iloc[0])
        severity = int(filtered_data['SEVERITY'].iloc[0])
        return precipitation, severity
    
    return 0.0, 0  # Default values if no data found

def get_rainfall_data(state_symbol, quarter):
    """Get rainfall data for years > 2015"""
    try:
        rainfall_data = pd.read_csv('data/flood_gen.csv')
        given_state = rainfall_data['STATE'] == state_symbol
        filtered_data = rainfall_data[given_state]
        
        if len(filtered_data) > 0:
            return filtered_data[quarter].iloc[0]
        return 0.0
    except Exception as e:
        print(f"Error getting rainfall data: {e}")
        return 0.0

def predict_flood_severity(state, precipitation):
    """Predict flood severity based on state and precipitation"""
    try:
        # Get state encoding
        state_encoded = labelencoder_X_1.transform([state])[0]
        print(f"State: {state}, encoded as: {state_encoded}")
        print(f"Precipitation: {precipitation}")
        
        # Create direct input array
        direct_x = np.zeros((1, 31), dtype=np.float64)
        direct_x[0, 0] = float(state_encoded)
        direct_x[0, -1] = float(precipitation)
        
        print(f"Direct input shape: {direct_x.shape}")
        print(f"Contains NaN: {np.isnan(direct_x).any()}")
        
        # Convert to TensorFlow tensor explicitly
        tf_input = tf.convert_to_tensor(direct_x, dtype=tf.float32)
        
        # Predict and get response
        out = model.predict(tf_input)
        response = np.argmax(out, axis=1)[0]
        print("predict response:", response)
        
        return int(response)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0  # Default severity if prediction fails

def create_encoder_and_scaler():
    """Create and fit encoders and scaler"""
    global labelencoder_X_1, onehotencoder, sc_X
    
    dataset = pd.read_csv('data/flood_past.csv')
    dataset = dataset[dataset['YEAR'] > 1980]
    dataset = dataset.dropna()
    
    # Make sure we have all unique state values
    all_states = dataset['SUBDIVISION'].unique()
    labelencoder_X_1.fit(all_states)
    
    X = dataset.iloc[:, [0, 4]].values
    y = dataset.iloc[:, 8].values
     
    X[:, 0] = labelencoder_X_1.transform(X[:, 0])
    
    # Create a reshaped version with only column 0 encoded
    X_encoded = X[:, 0].reshape(-1, 1)
    # Apply onehotencoder to just the first column
    encoded_features = onehotencoder.fit_transform(X_encoded)
    # Combine with the second column
    X = np.column_stack((encoded_features, X[:, 1]))
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Update this section for newer OneHotEncoder
    onehotencoder_2 = OneHotEncoder(sparse_output=False)
    y_train = np.reshape(y_train, (-1, 1))
    y_train = onehotencoder_2.fit_transform(y_train)
    
    X_train = sc_X.fit_transform(X_train)