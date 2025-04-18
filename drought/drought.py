import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def get_historical_drought_data(state, year, month):
    """
    Get historical drought data for a specific state, year, and month.
    Returns precipitation and drought severity.
    """
    try:
        print(f"Getting historical data for {state}, {year}, {month}")
        # Read historical data
        df = pd.read_csv("data/flood_past.csv")
        print(f"Total records in dataset: {len(df)}")
        
        # Filter data for the specific state
        state_data = df[df['SUBDIVISION'] == state]
        print(f"Records for {state}: {len(state_data)}")
        
        if len(state_data) == 0:
            print(f"No data found for {state}")
            return None, "Insufficient Data"
            
        # Calculate average precipitation
        avg_precip = state_data['PRECIPITATION'].mean()
        print(f"Average precipitation for {state}: {avg_precip}")
        
        # Get precipitation for the specific year if available
        year_data = state_data[state_data['YEAR'] == int(year)]
        if len(year_data) > 0:
            precip = year_data['PRECIPITATION'].iloc[0]
            print(f"Found precipitation for {year}: {precip}")
        else:
            # If no data for specific year, use average
            precip = avg_precip
            print(f"Using average precipitation for {year}: {precip}")
            
        # Determine drought severity based on precipitation
        if precip < avg_precip * 0.5:
            severity = "Severe Drought"
        elif precip < avg_precip * 0.75:
            severity = "Moderate Drought"
        elif precip < avg_precip * 0.9:
            severity = "Mild Drought"
        else:
            severity = "No Drought"
            
        print(f"Calculated severity: {severity}")
        return precip, severity
        
    except Exception as e:
        print(f"Error in get_historical_drought_data: {str(e)}")
        raise Exception(f"Error getting historical drought data: {str(e)}")

def predict_drought(state, year, month):
    """
    Predict drought conditions for a given state, year, and month.
    Returns precipitation and drought severity.
    """
    try:
        # Read historical data
        df = pd.read_csv("data/flood_past.csv")
        
        # Filter data for the specific state
        state_data = df[df['SUBDIVISION'] == state]
        
        if len(state_data) == 0:
            # If no data for the state, use overall average
            avg_precip = df['PRECIPITATION'].mean()
            precip = avg_precip
        else:
            # Calculate average precipitation for the state
            avg_precip = state_data['PRECIPITATION'].mean()
            
            # Get precipitation for the specific year if available
            year_data = state_data[state_data['YEAR'] == int(year)]
            if len(year_data) > 0:
                precip = year_data['PRECIPITATION'].iloc[0]
            else:
                # If no data for specific year, use state average
                precip = avg_precip
        
        # Get terrain type for the state
        terrain = state_data['TERRAIN'].iloc[0] if len(state_data) > 0 else "Unknown"
        
        # Adjust thresholds based on terrain type
        if terrain.lower() in ['desert', 'desert/marsh']:
            # Lower thresholds for desert regions
            if precip < avg_precip * 0.3:
                severity = "Severe Drought"
            elif precip < avg_precip * 0.5:
                severity = "Moderate Drought"
            elif precip < avg_precip * 0.7:
                severity = "Mild Drought"
            else:
                severity = "No Drought"
        elif terrain.lower() in ['plain-land', 'coastal']:
            # Moderate thresholds for plain and coastal regions
            if precip < avg_precip * 0.4:
                severity = "Severe Drought"
            elif precip < avg_precip * 0.6:
                severity = "Moderate Drought"
            elif precip < avg_precip * 0.8:
                severity = "Mild Drought"
            else:
                severity = "No Drought"
        else:
            # Standard thresholds for other regions
            if precip < avg_precip * 0.5:
                severity = "Severe Drought"
            elif precip < avg_precip * 0.7:
                severity = "Moderate Drought"
            elif precip < avg_precip * 0.9:
                severity = "Mild Drought"
            else:
                severity = "No Drought"
            
        return precip, severity
        
    except Exception as e:
        # If any error occurs, return a default prediction
        return 100.0, "No Drought"  # Default to no drought with average precipitation