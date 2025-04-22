import joblib
import pandas as pd
import os

# Function to load the models and handle any file loading errors
states=['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 
 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 
 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

city_to_state = {
            'New Delhi': 'DL',
            'Delhi': 'DL',
            'Mumbai': 'MH',
            'Chennai': 'TN',
            'Kolkata': 'WB',
            'Bengaluru': 'KA',
            'Hyderabad': 'AD',
            'Ahmedabad': 'GJ',
            'Pune': 'MH',
            'Jaipur': 'RJ',
            'Lucknow': 'UP',
            'Kanpur': 'UP',
            'Nagpur': 'MH',
            'Indore': 'MP',
            'Thane': 'MH',
            'Bhopal': 'MP',
            'Visakhapatnam': 'AD',
            'Patna': 'BR',
            'Vadodara': 'GJ',
            'Ghaziabad': 'UP',
            'Ludhiana': 'PB',
            'Agra': 'UP',
            'Nashik': 'MH',
            'Faridabad': 'HR',
            'Meerut': 'UP',
            'Rajkot': 'GJ',
            'Kalyan': 'MH',
            'Vasai': 'MH',
            'Varanasi': 'UP',
            'Srinagar': 'JK',
            'Aurangabad': 'MH',
            'Dhanbad': 'JH',
            'Amritsar': 'PB',
            'Navi Mumbai': 'MH',
            'Allahabad': 'UP',
            'Ranchi': 'JH',
            'Howrah': 'WB',
            'Coimbatore': 'TN',
            'Jabalpur': 'MP',
            'Gwalior': 'MP',
            'Vijayawada': 'AD',
            'Jodhpur': 'RJ',
            'Madurai': 'TN',
            'Raipur': 'CG',
            'Kota': 'RJ',
            'Chandigarh': 'CH',
            'Guwahati': 'AS',
            'Solapur': 'MH',
            'Hubli': 'KA',
            'Mysore': 'KA',
            'Tiruchirappalli': 'TN',
            'Bareilly': 'UP',
            'Moradabad': 'UP',
            'Tiruppur': 'TN'
        }

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    except FileNotFoundError as e:
        print(f" {e}")
        raise  # Re-raise the exception to stop execution
    except Exception as e:
        print(f" Error loading model from {model_path}: {e}")
        raise

# Load models and scalers at the top of your script
try:
    rf_model = load_model('model/RF_predict_model_no_month.joblib')
    logreg_model = load_model('model/logreg_predict_model_no_month.joblib')
    scaler = load_model('model/Landslide_Scaler_no_month.joblib')
    le = load_model('model/State_Label_Encoder_no_month.joblib')
except Exception as e:
    print(" Critical error: Models could not be loaded. Exiting the application.")
    exit(1)  # Exit if models cannot be loaded

def predict_landslide(rainfall, state):
    """
    Predicts the landslide probability and gives the final prediction.
    
    Parameters:
    - rainfall (float): Amount of rainfall in mm.
    - state (str): The state name to encode and predict the landslide.
    
    Returns:
    - final_prob (float): Weighted combined probability.
    - final_prediction (int): Final prediction (1 for Landslide, 0 for No Landslide).
    """
    
    if not isinstance(rainfall, (int, float)) or rainfall < 0:
        print("Invalid rainfall value. Please enter a positive number for rainfall.")
        return None, None
    
    try:
        state_encoded = le.transform([state])[0]
    except ValueError:
        print(f" Invalid state name: '{state}'. Please provide a valid state.")
        return None, None
    

    try:
        input_data = pd.DataFrame([[rainfall, state_encoded]], columns=['RAINFALL', 'State_Encoded'])
        input_data_scaled = scaler.transform(input_data)
    except Exception as e:
        print(f" Error preparing or scaling input data: {e}")
        return None, None
    
    try:
        rf_prob = rf_model.predict_proba(input_data_scaled)[0][1]  
        logreg_prob = logreg_model.predict_proba(input_data_scaled)[0][1]
    except Exception as e:
        print(f" Error during model prediction: {e}")
        return None, None

    final_prob = 0.6 * rf_prob + 0.4 * logreg_prob
    
    
    return final_prob