import pandas as pd
import numpy as np
from flood.config import STATE_MAPPING, MONTH_MAPPING
from flood.flood import (
    get_state_and_terrain,
    get_month_details,
    get_historical_data,
    get_rainfall_data,
    predict_with_model
)

def get_normal_rainfall(state, month):
    """
    Get normal rainfall for a state and month based on historical data.
    Returns the average rainfall in mm.
    """
    try:
        # Get historical data for the last 10 years
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
    """
    Classify drought severity based on IMD criteria and precipitation threshold:
    - If precipitation < 50mm: At least Mild Drought
    - Then check IMD deficiency criteria:
      - Normal: Within 19% of normal
      - Mild Drought: 20-59% deficiency
      - Moderate Drought: 60-99% deficiency
      - Severe Drought: 100% deficiency
    """
    if normal_rainfall == 0:
        return "No Drought"
        
    # First check precipitation threshold
    if current_rainfall < 50:
        # If below threshold, check IMD deficiency to determine severity
        deficiency = ((normal_rainfall - current_rainfall) / normal_rainfall) * 100
        
        if deficiency < 60:
            return "Mild Drought"
        elif deficiency < 100:
            return "Moderate Drought"
        else:
            return "Severe Drought"
    else:
        # If above threshold, use only IMD criteria
        deficiency = ((normal_rainfall - current_rainfall) / normal_rainfall) * 100
        
        if deficiency < 20:
            return "No Drought"
        elif deficiency < 60:
            return "Mild Drought"
        elif deficiency < 100:
            return "Moderate Drought"
        else:
            return "Severe Drought"

def predict_drought(state, year, month, flood_precipitation):
    """
    Predict drought conditions using flood.py's precipitation value.
    Returns drought severity based on simple threshold:
    - High Risk: precipitation < 50mm
    - Otherwise: No Drought
    """
    try:
        # Simple threshold check
        if flood_precipitation < 50:
            return "High Risk of Drought"
        return "No Drought"
        
    except Exception as e:
        print(f"Error in drought prediction: {str(e)}")
        return "No Drought"
