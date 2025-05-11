import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import sys

# Import config and model loader using relative imports
from flood.config import STATE_MAPPING, MONTH_MAPPING, labelencoder_X_1, onehotencoder, sc_X
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sys.path.append(os.path.abspath("./model"))
from flood.load import init

# Don't initialize model here - this creates circular dependency
# Instead, initialize when needed
model = None
graph = None

def initialize_model():
    """Initialize the model when needed"""
    global model, graph
    if model is None:
        model, graph = init()
    return model, graph

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

def generate_future_rainfall_data(state_symbol, quarter):
    """Get future rainfall data for years > 2015 from flood_gen_future.csv"""
    try:
        # Load the future rainfall data
        rainfall_data = pd.read_csv('data/flood_gen_future.csv')
        
        # Filter the data for the given state
        given_state = rainfall_data['SUBDIVISION'] == state_symbol
        filtered_data = rainfall_data[given_state]
        
        # Further filter the data for the given quarter
        given_quarter = filtered_data['QUARTER'] == quarter
        filtered_data = filtered_data[given_quarter]
        
        # Return the predicted precipitation if data exists
        if len(filtered_data) > 0:
            return filtered_data['PREDICTED_PRECIPITATION'].iloc[0]
        return 0.0
    except Exception as e:
        print(f"Error getting future rainfall data: {e}")
        return 0.0 

def predict_with_model(state, precipitation, terrain):
    """Predict flood severity based on state, precipitation, and terrain"""
    try:
        # Get state encoding
        state_encoded = labelencoder_X_1.transform([state])[0]
        
        # Normalize the terrain to standard categories
        normalized_terrain = normalize_terrain(terrain)
        
        # Encode terrain with standard categories
        terrain_encoder = LabelEncoder()
        standard_terrains = ['Desert', 'Plain', 'Coastal', 'Mountain', 'Plateau']
        terrain_encoder.fit(standard_terrains)
        
        try:
            terrain_encoded = terrain_encoder.transform([normalized_terrain])[0]
        except ValueError:
            print(f"Warning: Normalized terrain '{normalized_terrain}' not in standard list, using 'Plain'")
            terrain_encoded = terrain_encoder.transform(['Plain'])[0]
        
        print(f"State: {state}, encoded as: {state_encoded}")
        print(f"Terrain: {terrain} (normalized to: {normalized_terrain}), encoded as: {terrain_encoded}")
        print(f"Precipitation: {precipitation}")
        
        # Create raw features array
        raw_features = np.array([[state_encoded, terrain_encoded, float(precipitation)]])
        
        # One-hot encode state
        state_encoded_reshaped = raw_features[:, 0].reshape(-1, 1).astype(np.int32)
        try:
            state_onehot = onehotencoder.transform(state_encoded_reshaped)
        except Exception as e:
            print(f"Warning: Error encoding state: {e}. Using zeros.")
            # Create a dummy state encoding with zeros
            state_onehot = np.zeros((1, 30))  # Assuming up to 30 states
            if state_encoded < 30:
                state_onehot[0, state_encoded] = 1
        
        # One-hot encode terrain
        terrain_onehotencoder = OneHotEncoder(sparse_output=False)
        # Fit with all possible terrain indices (0-4)
        terrain_onehotencoder.fit(np.array([[0], [1], [2], [3], [4]]))
        terrain_encoded_reshaped = raw_features[:, 1].reshape(-1, 1).astype(np.int32)
        terrain_onehot = terrain_onehotencoder.transform(terrain_encoded_reshaped)
        
        # Combine one-hot encoded features with precipitation
        precipitation_reshaped = raw_features[:, 2].reshape(-1, 1).astype(np.float32)
        combined_features = np.hstack((state_onehot, terrain_onehot, precipitation_reshaped))
        
        # Scale features
        scaled_features = sc_X.transform(combined_features)
        
        print(f"Input features shape: {scaled_features.shape}")
        print(f"Contains NaN: {np.isnan(scaled_features).any()}")
        
        # Convert to TensorFlow tensor
        tf_input = tf.convert_to_tensor(scaled_features, dtype=tf.float32)
        
        # Predict flood severity
        model, graph = initialize_model()
        with graph.as_default():
            out = model.predict(tf_input)
        response = np.argmax(out, axis=1)[0]
        print("Predicted severity:", response)
        
        return int(response)
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return 0  # Default severity if prediction fails
    
def predict_flood_severity(state, precipitation, terrain):
    """
    Predict flood severity based on state, precipitation, and terrain,
    by combining an ML model with rule-based overrides for extremes.
    Severity ranges 0–5 (0 = no flood, 5 = severe flood).

    Args:
        state (str): State code (e.g. "WB", "AN", …)
        precipitation (float): Precipitation in mm for the period
        terrain (str): Terrain type string (e.g. "Coastal", "Hilly", …)

    Returns:
        int: Final severity prediction (0–5)
    """
    # 1. Get ML model prediction (assumed to return 0–5)
    model_prediction = predict_with_model(state, precipitation, terrain)

    # 2. Normalize terrain string
    t = terrain.strip().lower()

    # 3. Rule‑based overrides for each terrain group
    # Coastal group (incl. plateau & mixed)
    if t in {"coastal", "hilly/coastal", "coastal-plateau"}:
        if precipitation > 1500:
            return max(3, model_prediction)
        elif precipitation > 1000:
            return max(2, model_prediction)
        elif precipitation > 500:
            return max(1, model_prediction)

    # Island
    elif t == "island":
        if precipitation > 2000:
            return max(4, model_prediction)
        elif precipitation > 1500:
            return max(3, model_prediction)
        elif precipitation > 800:
            return max(2, model_prediction)

    # Hilly + mixed plain/hilly
    elif t in {"hilly", "hilly/plain"}:
        if precipitation > 1800:
            return max(3, model_prediction)
        elif precipitation > 1200:
            return max(2, model_prediction)
        elif precipitation > 700:
            return max(1, model_prediction)

    # Plain‑land
    elif t == "plain-land":
        if precipitation > 1200:
            return max(3, model_prediction)
        elif precipitation > 800:
            return max(2, model_prediction)
        elif precipitation > 400:
            return max(1, model_prediction)

    # Desert
    elif t == "desert":
        if precipitation > 300:
            return max(3, model_prediction)
        elif precipitation > 200:
            return max(2, model_prediction)
        elif precipitation > 100:
            return max(1, model_prediction)

    # Desert/marsh
    elif t == "desert/marsh":
        if precipitation > 500:
            return max(3, model_prediction)
        elif precipitation > 300:
            return max(2, model_prediction)
        elif precipitation > 150:
            return max(1, model_prediction)

    # Forest
    elif t == "forest":
        if precipitation > 800:
            return max(3, model_prediction)
        elif precipitation > 500:
            return max(2, model_prediction)
        elif precipitation > 250:
            return max(1, model_prediction)

    # Rugged terrain (mountainous/complex)
    elif t == "rugged":
        if precipitation > 1500:
            return max(3, model_prediction)
        elif precipitation > 1000:
            return max(2, model_prediction)
        elif precipitation > 600:
            return max(1, model_prediction)

    # “Everything” or unrecognized terrains: use broad thresholds
    else:
        if precipitation > 1500:
            return max(3, model_prediction)
        elif precipitation > 1000:
            return max(2, model_prediction)
        elif precipitation > 500:
            return max(1, model_prediction)

    # If no override triggers, fall back to the ML prediction
    return model_prediction


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

def normalize_terrain(terrain):
    """
    Map various terrain descriptions to the five standard terrain categories:
    Desert, Plain, Coastal, Mountain/Hilly, Plateau
    """
    terrain = str(terrain).strip().lower()
    
    # Check for partial matches to handle truncated strings like "Hilly/co"
    if "desert" in terrain:
        return "Desert"
    elif any(x in terrain for x in ["plain", "plain-land", "barren"]):
        return "Plain"
    elif any(x in terrain for x in ["coast", "coastal", "/co"]):
        return "Coastal"
    elif any(x in terrain for x in ["hill", "mount", "hilly"]):
        return "Mountain"  # Treating all hilly terrains as Mountain
    elif "plateau" in terrain:
        return "Plateau"
    elif "island" in terrain:
        return "Coastal"  # Islands treated as coastal
    elif "forest" in terrain:
        return "Mountain"  # Forests often in hilly regions
    elif "marsh" in terrain:
        return "Plain"  # Marshes treated as special plains
    
    # Default fallback
    print(f"Warning: Unknown terrain '{terrain}', defaulting to Plain")
    return "Plain"


