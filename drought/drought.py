import sys
import os

# Add the parent directory to sys.path to access 'flood' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from flood.config import STATE_MAPPING, MONTH_MAPPING
from flood.flood import (
    get_state_and_terrain,
    get_month_details,
    get_historical_data,
    get_rainfall_data,
    predict_with_model
)

# Load ML model and label encoder
xgb_model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'model', 'xgb_drought_model.pkl'))
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'model', 'xgb_drought_encoders.pkl'))['SEVERITY']

# Terrain explanation mapping
terrain_analysis = {
    "Mountain": "Mountainous regions depend on orographic rainfall. Droughts may occur due to reduced monsoon winds or deforestation.",
    "Coastal": "Coastal areas typically get steady rain. A drought here suggests cyclonic inactivity or climate change effects.",
    "Plateau": "Plateaus are often in rain shadow zones. Droughts arise from consistently low rainfall trends.",
    "Plain": "Plains depend on monsoons and rivers. A drought may indicate monsoon failure or upstream water disputes."
}

Z_SCORE_THRESHOLD = 2.0

def get_normal_rainfall(state, month):
    try:
        years = range(2005, 2016)
        total_rainfall = 0
        count = 0

        for year in years:
            state_name, terrain = get_state_and_terrain(state)
            quarter, duration = get_month_details(month)
            rainfall, _ = get_historical_data(state_name, str(year), quarter, terrain)

            if rainfall > 0:
                total_rainfall += rainfall
                count += 1

        if count > 0:
            return total_rainfall / count
        return 0
    except Exception as e:
        print(f"Error getting normal rainfall: {str(e)}")
        return 0

def classify_drought_severity(current_rainfall, normal_rainfall):
    if normal_rainfall == 0:
        return "No Drought"

    deficiency = ((normal_rainfall - current_rainfall) / normal_rainfall) * 100

    if current_rainfall < 50:
        if deficiency < 60:
            return "Mild Drought"
        elif deficiency < 100:
            return "Moderate Drought"
        else:
            return "Severe Drought"
    else:
        if deficiency < 20:
            return "No Drought"
        elif deficiency < 60:
            return "Mild Drought"
        elif deficiency < 100:
            return "Moderate Drought"
        else:
            return "Severe Drought"

def detect_anomaly(current_rainfall, historical_rainfalls):
    if len(historical_rainfalls) == 0:
        return False
    mean = np.mean(historical_rainfalls)
    std = np.std(historical_rainfalls)
    z = (mean - current_rainfall) / std if std > 0 else 0
    return z > Z_SCORE_THRESHOLD

def explain_drought_terrain(terrain):
    return terrain_analysis.get(terrain, "No specific terrain explanation available.")

def classify_drought_ml(current_rainfall, normal_rainfall, month, state, terrain):
    try:
        terrain_map = {"Plain": 0, "Plateau": 1, "Coastal": 2, "Mountain": 3}
        terrain_code = terrain_map.get(terrain, 0)
        state_code = list(STATE_MAPPING.values()).index(state) if state in STATE_MAPPING.values() else 0

        features = np.array([[current_rainfall, normal_rainfall, MONTH_MAPPING[month], state_code, terrain_code]])
        prediction = xgb_model.predict(features)
        drought_class = label_encoder.inverse_transform(prediction)[0]
        return drought_class
    except Exception as e:
        print(f"ML classification failed: {str(e)}")
        return "Unknown"

def _detailed_predict_drought(state, year, month, flood_precipitation):
    try:
        state_name, terrain = get_state_and_terrain(state)
        quarter, _ = get_month_details(month)
        normal_rainfall = get_normal_rainfall(state, month)

        # Logging to verify inputs
        print(f"Predicting for State: {state_name}, Year: {year}, Month: {month}")
        print(f"Flood Precipitation: {flood_precipitation}, Normal Rainfall: {normal_rainfall}")

        history = []
        for yr in range(2005, 2016):
            try:
                rain, _ = get_historical_data(state_name, str(yr), quarter, terrain)
                history.append(rain)
            except:
                continue

        imd_class = classify_drought_severity(flood_precipitation, normal_rainfall)
        print(f"IMD Classification: {imd_class}")  # Check IMD classification
        
        # Using the ML classification
        ml_class = classify_drought_ml(flood_precipitation, normal_rainfall, month, state_name, terrain)
        print(f"ML Classification: {ml_class}")  # Check ML classification
        
        terrain_reason = explain_drought_terrain(terrain)
        anomaly_flag = detect_anomaly(flood_precipitation, history)

        # Returning the full results with all details
        return {
            
           "anomaly_detected": anomaly_flag
        }

    except Exception as e:
        print(f"Error in drought prediction: {str(e)}")
        return {
        
            "anomaly_detected": False

        }



def predict_drought(state, year, month, flood_precipitation):
    """
    Returns only 'High Risk' or 'Low Risk' for drought, based on anomaly detection.
    """
    full_result = _detailed_predict_drought(state, year, month, flood_precipitation)

    if full_result.get("anomaly_detected", False):
        return "High Risk"
    return "Low Risk"

