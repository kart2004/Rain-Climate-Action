from flask import Flask, redirect, url_for, render_template, request, session
import requests
import urllib
from datetime import datetime
import numpy as np
import tensorflow as tf
import sys
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Import flood-related functionality
from flood.flood import (
    get_state_and_terrain, 
    get_month_details, 
    normalize_precipitation,
    get_historical_data,
    get_rainfall_data,
    predict_flood_severity,
    create_encoder_and_scaler
)
from erosion.erosion import predict_r_factor
# Import drought-related functionality
from drought.drought import (
    predict_drought
)

from landslide.model import predict_landslide,states,city_to_state
# Import configuration
from flood.config import STATE_MAPPING, BING_API_KEY, OPENWEATHER_API_KEY

#Import groundwater-related functionality
from groundwater.groundwateranalysis import (
    initialize_groundwater_data,
    analyze_groundwater_risk,
    get_risk_level
)

# Enable eager execution for TensorFlow
tf.compat.v1.enable_eager_execution()

# Initialize Flask app

app = Flask(__name__)
app.secret_key = 'my secret and not your secret'

# Route for About Us page
@app.route('/aboutus')
def about():
    return render_template("about.html")

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Route to handle login and redirect to prediction page
@app.route('/grantaccess', methods=['POST'])
def grantaccess():
    try:
        location = request.form.get('location')
        date = request.form.get('date')
        
        # Map common cities to their state codes
        city_to_state = {
            'New Delhi': 'DL',
            'Delhi': 'DL',
            'Mumbai': 'MH',
            'Chennai': 'TN',
            'Kolkata': 'WB',
            'Bengaluru': 'KA',
            'Bangalore': 'KA',
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
        
        # Get state code from city name
        state_code = city_to_state.get(location)
        if not state_code:
            # If city not found in mapping, use the location as state code
            state_code = location
        
        try:
            # Try to parse date in different formats
            try:
                date_obj = datetime.strptime(date, '%B %Y')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    date_obj = datetime.strptime(date, '%d/%m/%y')
            
            year = date_obj.year
            month = date_obj.month
            
        except ValueError as e:
            return render_template('error.html', error=f"Invalid date format: {str(e)}")
        
        # Get state details and month details
        state_name, terrain = get_state_and_terrain(state_code)
        quarter, duration = get_month_details(month)
        
        # Get precipitation based on year
        if year <= 2015:
            precipitation, severity = get_historical_data(state_name, year, quarter, terrain)
        else:
            precipitation = get_rainfall_data(state_code, quarter)
            
        if precipitation == 0.0:
            # If no data found, use the generated data
            precipitation = get_rainfall_data(state_code, quarter)
        
        # Normalize precipitation for the duration
        precipitation = normalize_precipitation(precipitation, duration)
        
        # Store the precipitation value in session
        session['precipitation'] = precipitation
        
        # Determine drought severity
        if precipitation < 50:
            severity = "High Risk of Drought"
            summary = "Warning: High risk of drought detected. Immediate water conservation measures required."
        else:
            severity = "No Drought Risk"
            summary = "No significant drought risk detected. Continue normal water usage with conservation practices."

        groundwater_data = analyze_groundwater_risk(state_code)
            
        return render_template('drought_predict.html',
                             location=location,
                             date=date,
                             precipitation=precipitation,
                             severity=severity,
                             summary=summary,
                             groundwater_risk=groundwater_data['risk_level'],
                             groundwater_percentage=groundwater_data['risk_percentage'],
                             groundwater_recharge=groundwater_data['recharge_rate'],
                             groundwater_summary=groundwater_data['summary'])
        
    except Exception as e:
        return render_template('error.html', error=f"Sorry, something went wrong: {str(e)}")

# Location route to get location data from the Bing API
@app.route('/location', methods=['GET', 'POST'])
def jsonlocation():
    if request.method == 'POST':
        year = request.form.get('year')
        location = request.form.get('location')
        date = request.form.get('date')
    else:
        year = request.args.get('year')
        location = request.args.get('location')
        date = request.args.get('date')
    
    # Validate required parameters
    if not all([year, location, date]):
        return render_template('error.html', error="Missing required parameters. Please provide location, date, and year.")
    
    url = 'http://dev.virtualearth.net/REST/v1/Locations?'
    key = BING_API_KEY
    cr = 'IN'
    results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
    response = requests.get(results)
    parser = response.json()
    auth = parser['statusDescription']
    if auth == 'OK':
        if 'adminDistrict' not in parser['resourceSets'][0]['resources'][0]['address']:
            return render_template('error.html', error="Location does not exist in India! Please try again!")
        state = parser['resourceSets'][0]['resources'][0]['address']['adminDistrict']
        lat = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][0]
        lon = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][1]
        city = parser['resourceSets'][0]['resources'][0]['address']['locality']
        
        # Ensure all required parameters are present before redirect
        if not all([city, state, date, lat, lon, year]):
            return render_template('error.html', error="Failed to retrieve complete location data. Please try again.")
            
        return redirect(url_for('jsonweather', 
            city=city, 
            state=state, 
            date=date, 
            latitude=lat, 
            longitude=lon, 
            year=year
        ))
    else:
        return render_template('error.html', error=f"Status: {auth}! Server issue! Please try again later!")

# Weather data retrieval route using OpenWeather API
@app.route('/weather')
def jsonweather():
    year = request.args.get('year')
    city = request.args.get('city')
    state = request.args.get('state')
    date = request.args.get('date')
    lat = request.args.get('latitude')
    lon = request.args.get('longitude')
    
    # Validate required parameters
    if not all([year, city, state, date, lat, lon]):
        return render_template('error.html', error="Missing required parameters from location data")
    
    url = 'https://api.openweathermap.org/data/2.5/forecast?'
    key = OPENWEATHER_API_KEY
    mode = 'json'
    count = 32
    results = url + urllib.parse.urlencode(({'lat': lat, 'lon': lon, 'appid': key, 'mode': mode, 'cnt': count}))
    response = requests.get(results)
    
    parser = response.json()
    auth = parser['cod']
    totalprecipitation = 0.00
    if auth == '200':
        for each in parser['list']:
            if 'rain' not in each:
                continue
            else:
                rain = each['rain']
                if '3h' in rain:
                    prec = rain['3h']
                    totalprecipitation += prec
    else:
        return 'Status: %s! Server issue! Please try again later!' % auth
    
    totalprecipitation = totalprecipitation/4
    return redirect(url_for('interim', 
        city=city, 
        state=state, 
        date=date, 
        precipitation=round(totalprecipitation, 2), 
        year=year,
        latitude=lat,
        longitude=lon
    ))

# Route to process interim data for flood prediction
@app.route('/interim', methods=['GET', 'POST'])
def interim():
    city = request.args.get('city')
    state = request.args.get('state')
    date = request.args.get('date')
    precip_str = request.args.get('precipitation')
    year = request.args.get('year')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    
    # Validate required parameters
    if not all([city, state, date, precip_str, year, latitude, longitude]):
        return render_template('error.html', error="Missing required parameters from weather data")
    
    try:
        precipitation = float(precip_str)
    except ValueError:
        return render_template('error.html', error="Invalid precipitation value")

    # Calculate month - handle both date formats
    try:
        # Try parsing with the expected format first
        month = datetime.strptime(date, '%d/%m/%y').strftime('%m')
    except ValueError:
        try:
            # If that fails, try parsing with the alternative format
            month = datetime.strptime(date, '%Y-%m-%d').strftime('%m')
        except ValueError:
            return render_template('error.html', error="Invalid date format. Please use either DD/MM/YY or YYYY-MM-DD format")
    
    actualMonth = month
    state_symbol = state

    # Get month details and state details
    quarter, duration = get_month_details(month)
    state_name, terrain = get_state_and_terrain(state)

    return redirect(url_for(
        'predict', 
        city=city, 
        state=state_name, 
        month=quarter, 
        precipitation=precipitation, 
        duration=duration, 
        terrain=terrain, 
        year=year, 
        actualMonth=actualMonth, 
        state_symbol=state_symbol,
        latitude=latitude,
        longitude=lon
    ), code=307)

# Flood prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get parameters from either query string or form data
    if request.method == 'POST':
        state = request.form.get('state')
        precip_str = request.form.get('precipitation')
        year = request.form.get('year')
        month = request.form.get('month')
        terrain = request.form.get('terrain')
        actualMonth = request.form.get('actualMonth')
        state_symbol = request.form.get('state_symbol')
        city = request.form.get('city')
        duration = request.form.get('duration')
        from_summary = request.form.get('from_summary', 'false')  # Add this to track source
    else:
        state = request.args.get('state')
        precip_str = request.args.get('precipitation')
        year = request.args.get('year')
        month = request.args.get('month')
        terrain = request.args.get('terrain')
        actualMonth = request.args.get('actualMonth')
        state_symbol = request.args.get('state_symbol')
        city = request.args.get('city')
        duration = request.args.get('duration')
        from_summary = request.args.get('from_summary', 'false')
    
    # Validate required parameters
    if not all([state, precip_str, year]):
        return render_template('error.html', error="Missing required parameters. Please ensure all fields are filled correctly.")
    
    try:
        precipitation = float(precip_str)
        year = int(year)
    except ValueError:
        return render_template('error.html', error="Invalid precipitation or year value")

    # Optional parameters
    if duration is not None:
        try:
            duration = float(duration)
        except ValueError:
            return render_template('error.html', error="Invalid duration value")

    # Only normalize precipitation if not coming from summary page
    if from_summary != 'true':
        # Normalize precipitation over the duration
        precipitation = normalize_precipitation(precipitation, duration)

    # For years <= 2015, use historical data
    if year <= 2015:
        historical_precip, severity = get_historical_data(state, year, month, terrain)
        
        # Only use historical data if not coming from summary
        if from_summary != 'true':
            precipitation = historical_precip

        return render_template(
            'flood_predict.html', 
            severity=str(severity), 
            city=city, 
            state=state, 
            month=month, 
            duration=duration, 
            precipitation=round(precipitation, 2), 
            terrain=terrain, 
            year=year,
            rf = predict_r_factor(state, year, precipitation)  
        )
    # For years > 2015, predict using model
    else:
        # Only recalculate precipitation if not coming from summary
        if from_summary != 'true':
            # Get rainfall data
            precipitation = get_rainfall_data(state_symbol, month)

        try:
            # Predict severity
            severity = predict_flood_severity(state, precipitation, terrain)

            return render_template(
                'flood_predict.html', 
                severity=str(severity), 
                city=city, 
                state=state, 
                month=month, 
                duration=duration, 
                precipitation=round(precipitation, 2), 
                terrain=terrain, 
                year=year,
                rf = predict_r_factor(state, year, precipitation)  
            )
        except Exception as e:
            return render_template('error.html', error=str(e), city=city, state=state)

# Route for Drought Prediction page
from drought.drought import predict_drought

@app.route('/drought', methods=['GET', 'POST'])
def drought():
    if request.method == 'POST':
        try:
            date = request.form.get('date')
            location = request.form.get('location')
            year = request.form.get('year')
            precipitation = float(request.form.get('precipitation'))

            # Call simplified drought prediction logic
            result = predict_drought(location, year, date.split("-")[1], precipitation)
            anomaly = result.get("anomaly_detected", False)

            return render_template('drought_predict.html', 
                                   anomaly=anomaly,
                                   location=location,
                                   precipitation=precipitation,
                                   date=date,
                                   year=year)
        except Exception as e:
            return render_template('drought_predict.html', 
                                   error=f"An error occurred: {str(e)}")
    else:
        # For GET requests, display the drought prediction page
        location = request.args.get('location')
        date = request.args.get('date')
        year = request.args.get('year')
        precipitation = float(request.args.get('precipitation', 0))  # Default to 0 if not present

        # Default anomaly value
        anomaly = False

        return render_template('drought_predict.html', 
                               anomaly=anomaly,
                               location=location,
                               date=date,
                               year=year,
                               precipitation=precipitation)





# Add this new route after your existing routes
@app.route('/landslide',methods=['GET','POST'])
def landslide():
    if request.method=='POST':
        location=request.form.get('location')
        rainfall=request.form.get('precipitation')
        final_prediction=request.form.get('final_prediction')
        month=request.form.get('month')
    else:
        location=request.args.get('location')
        rainfall=request.args.get('precipitation')
        final_prediction=request.args.get('final_prediction')
        month=request.args.get('month')
    if not all([location, rainfall,final_prediction,month]):
        return render_template('error.html', error="Missing required parameters from initial form")
    
    #print(f'Recevied {location} ,{rainfall} and {final_prediction} and {month}')
    try:
        return render_template('landslide_predict.html',location=location,final_prediction=final_prediction,precipitation=rainfall)
    except Exception as e:
        return render_template('error.html',error=str(e))   

# Results summary page showing both predictions
@app.route('/summary_results', methods=['POST'])
def summary_results():
    try:
        # Get form data
        location = request.form.get('location')
        date = request.form.get('date')
        
        # Validate required fields
        if not all([location, date]):
            return render_template('error.html', error="Please provide both location and date")
        
        # Process date
        try:
            year = date[:4]
            formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%y')
            month_name = datetime.strptime(date, '%Y-%m-%d').strftime('%B')
            month_num = datetime.strptime(date, '%Y-%m-%d').strftime('%m')
        except ValueError:
            return render_template('error.html', error="Invalid date format. Please use YYYY-MM-DD format")
        
        # Get location data from Bing Maps API
        url = 'http://dev.virtualearth.net/REST/v1/Locations?'
        key = BING_API_KEY
        cr = 'IN'
        results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
        response = requests.get(results)
        parser = response.json()
        
        if parser['statusDescription'] != 'OK':
            return render_template('error.html', error="Could not retrieve location data. Please try again.")
            
        if 'adminDistrict' not in parser['resourceSets'][0]['resources'][0]['address']:
            return render_template('error.html', error="Location does not exist in India! Please try again!")
            
        # Get state name from Bing Maps response
        state = parser['resourceSets'][0]['resources'][0]['address']['adminDistrict']
        city = parser['resourceSets'][0]['resources'][0]['address']['locality']
        
        # Find state code from state name
        state_code = None
        for code, details in STATE_MAPPING.items():
            if details["full_name"].lower() == state.lower() or code.lower() == state.lower():
                state_code = code
                break
                
        if not state_code:
            return render_template('error.html', error=f"State '{state}' not found in our database. Please try another location.")
        
        # Get state details
        quarter, duration = get_month_details(month_num)
        state_name, terrain = get_state_and_terrain(state_code)
        
        # Get flood prediction data
        flood_precipitation = get_rainfall_data(state_code, quarter)
        flood_severity = predict_flood_severity(state_name, flood_precipitation, terrain)
        
        # Get drought prediction data using flood's precipitation
        drought_severity = predict_drought(location, year, month_name, flood_precipitation)
        
        # Create summary text based on severity
        flood_severity_text = ""
        flood_summary = ""
        flood_severity_percentage = 0
        
        if flood_severity == 0:
            flood_severity_text = "No Flood Risk"
            flood_summary = "Based on our analysis, there are negligible chances of a flood occurring in your area."
            flood_severity_percentage = 10
        elif flood_severity == 1:
            flood_severity_text = "Mild Flood Risk"
            flood_summary = "There are mild chances of a flood occurring. Be careful while going outdoors."
            flood_severity_percentage = 30
        elif flood_severity == 2:
            flood_severity_text = "Moderate Flood Risk"
            flood_summary = "There are high chances of a flood occurring. Going outdoors is not advisable."
            flood_severity_percentage = 50
        elif flood_severity == 3:
            flood_severity_text = "High Flood Risk"
            flood_summary = "There are very high chances of a flood occurring. Prepare for heavy water logging."
            flood_severity_percentage = 70
        elif flood_severity == 4:
            flood_severity_text = "Severe Flood Risk"
            flood_summary = "There are extremely high chances of a flood occurring. Take quick action to protect yourself."
            flood_severity_percentage = 85
        elif flood_severity == 5:
            flood_severity_text = "Extreme Flood Risk"
            flood_summary = "There are incredibly high chances of a severe flood occurring. Prepare for a strong wave of destruction."
            flood_severity_percentage = 100
        
        # Create drought summary text
        drought_summary = ""
        drought_severity_percentage = 0
        
        if drought_severity == "No Drought":
            drought_summary = "Based on our analysis, there is no drought predicted for your location."
            drought_severity_percentage = 10
        elif drought_severity == "Mild Drought":
            drought_summary = "A mild drought is predicted for your area. Basic water conservation is recommended."
            drought_severity_percentage = 40
        elif drought_severity == "Moderate Drought":
            drought_summary = "A moderate drought is expected. Consider implementing water conservation measures."
            drought_severity_percentage = 70
        elif drought_severity == "Severe Drought":
            drought_summary = "A severe drought is predicted. Immediate water conservation actions are needed."
            drought_severity_percentage = 100
        

        landslide_summary = ""
        # Initialize landslide_location with the original location to prevent UnboundLocalError
        landslide_location = location  

        if location not in states:
            state_code = city_to_state.get(location)
            if state_code:  # Check if state_code exists
                state_name, terrain = get_state_and_terrain(state_code)
                if state_name.title() in states:
                    landslide_location = state_name.strip().title()
                else:
                    # Search for partial matches
                    found_match = False
                    for state in states:
                        if state in state_name.strip().title():
                            landslide_location = state
                            found_match = True
                            break  # Exit loop once found
                    
                    # If no match found, keep the original value
                    if not found_match:
                        landslide_location = location
            else:
                # If city not found in mapping, keep original location
                landslide_location = location

        # Now landslide_location is guaranteed to be defined
        try:
            landslide_probability = predict_landslide(flood_precipitation, landslide_location.lower(),month_num)
            
            # Handle case where landslide_probability is a tuple instead of a number
            if isinstance(landslide_probability, tuple):
                landslide_probability = 0
                
            if landslide_probability >= 0.6:
                final_prediction = "Landslide risk exist"
                landslide_summary = "Based on our analysis your area is under threat of a severe landslide, evacuative measures are suggested"
            elif landslide_probability > 0 and landslide_probability < 0.3:
                final_prediction = "No landslide"
                landslide_summary = "Based on our analysis there is no landslide risk for your area, you can relax and enjoy the weather"
            elif landslide_probability >= 0.3 and landslide_probability < 0.6:
                final_prediction = "Moderate risk"
                landslide_summary = "Based on our analysis there is a moderate risk of landslide, be prepared"
            else:
                final_prediction = "Insufficient data"
                landslide_summary = "Due to insufficient data, analysis cannot be made in your region"
        except Exception as e:
            # Gracefully handle errors in landslide prediction
            print(f"Error in landslide prediction: {str(e)}")
            final_prediction = "Prediction unavailable"
            landslide_summary = "Unable to calculate landslide risk due to a technical issue"
            landslide_probability = 0

        try:
            groundwater_data = analyze_groundwater_risk(state_code)
        except Exception as e:
            print(f"Error in groundwater analysis: {str(e)}")
            groundwater_data = {
                'risk_level': "Analysis Unavailable",
                'risk_percentage': 0,
                'recharge_rate': 0,
                'summary': "Unable to analyze groundwater risk due to technical issue"
            }

        return render_template(
            'prediction_summary.html',
            location=city,  # Use city name from Bing Maps
            state=state,    # Add state name
            month=month_name,
            year=year,
            date=formatted_date,
            terrain=terrain,
            flood_precipitation=round(flood_precipitation, 2),
            flood_severity=flood_severity,
            flood_severity_text=flood_severity_text,
            flood_summary=flood_summary,
            flood_severity_percentage=flood_severity_percentage,
            drought_precipitation=round(flood_precipitation, 2),  # Use same precipitation as flood
            drought_severity=drought_severity,
            drought_summary=drought_summary,
            drought_severity_percentage=drought_severity_percentage,
            duration=duration,
            actual_month=month_num,
            state_code=state_code,
            landslide_probability=landslide_probability,
            landslide_prediction=final_prediction,
            landslide_summary=landslide_summary,
            groundwater_risk=groundwater_data['risk_level'],
            groundwater_percentage=groundwater_data['risk_percentage'],
            groundwater_recharge=groundwater_data['recharge_rate'],
            groundwater_summary=groundwater_data['summary']
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=f"An error occurred: {str(e)}")

@app.route('/long_term_predict', methods=['GET', 'POST'])
def long_term_predict():
    if request.method == 'POST':
        try:
            location = request.form.get('location')
            
            # Use Bing Maps API to validate and get location data (same as summary_results)
            url = 'http://dev.virtualearth.net/REST/v1/Locations?'
            key = BING_API_KEY
            cr = 'IN'
            results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
            response = requests.get(results)
            parser = response.json()
            
            if parser['statusDescription'] != 'OK':
                return render_template('error.html', error="Could not retrieve location data. Please try again.")
                
            if 'adminDistrict' not in parser['resourceSets'][0]['resources'][0]['address']:
                return render_template('error.html', error="Location does not exist in India! Please try again!")
                
            # Get state name from Bing Maps response
            state = parser['resourceSets'][0]['resources'][0]['address']['adminDistrict']
            city = parser['resourceSets'][0]['resources'][0]['address']['locality']
            
            # Find state code from state name (same as summary_results)
            state_code = None
            for code, details in STATE_MAPPING.items():
                if details["full_name"].lower() == state.lower() or code.lower() == state.lower():
                    state_code = code
                    break
                    
            if not state_code:
                return render_template('error.html', error=f"State '{state}' not found in our database. Please try another location.")
            
            # Get current year
            current_year = datetime.now().year
            
            # Get state details for terrain
            state_name, terrain = get_state_and_terrain(state_code)
            
            # Map Bing Maps state names to landslide model state names
            def map_state_for_landslide(bing_state):
                state_mapping = {
                    'Delhi': 'Delhi',
                    'Maharashtra': 'Maharashtra',
                    'Tamil Nadu': 'Tamil Nadu',
                    'West Bengal': 'West Bengal',
                    'Karnataka': 'Karnataka',
                    'Andhra Pradesh': 'Andhra Pradesh',
                    'Gujarat': 'Gujarat',
                    'Rajasthan': 'Rajasthan',
                    'Uttar Pradesh': 'Uttar Pradesh',
                    'Madhya Pradesh': 'Madhya Pradesh',
                    'Bihar': 'Bihar',
                    'Punjab': 'Punjab',
                    'Haryana': 'Haryana',
                    'Jammu and Kashmir': 'Jammu and Kashmir',
                    'Jharkhand': 'Jharkhand',
                    'Kerala': 'Kerala',
                    'Assam': 'Assam',
                    'Chhattisgarh': 'Chhattisgarh',
                    'Chandigarh': 'Chandigarh',
                    'Odisha': 'Odisha',
                    'Uttarakhand': 'Uttarakhand',
                    'Telangana': 'Telangana',
                    'Himachal Pradesh': 'Himachal Pradesh',
                    'Tripura': 'Tripura',
                    'Manipur': 'Manipur',
                    'Meghalaya': 'Meghalaya',
                    'Nagaland': 'Nagaland',
                    'Arunachal Pradesh': 'Arunachal Pradesh',
                    'Mizoram': 'Mizoram',
                    'Sikkim': 'Sikkim',
                    'Goa': 'Goa',
                    'Lakshadweep': 'Lakshadweep',
                    'Andaman and Nicobar Islands': 'Andaman and Nicobar Islands',
                    'Dadra and Nagar Haveli': 'Gujarat',
                    'Daman and Diu': 'Gujarat',
                    'Puducherry': 'Tamil Nadu'
                }
                return state_mapping.get(bing_state, bing_state)
            
            landslide_state = map_state_for_landslide(state)
            
            # Generate predictions for next 5 years using the exact same rainfall prediction logic
            predictions = []
            for year in range(current_year + 1, current_year + 6):
                # Initialize year_predictions dictionary
                year_predictions = {
                    'year': year,
                    'rainfall': {},
                    'disasters': {}
                }
                
                # Use the EXACT same rainfall prediction logic as summary_results
                # Get month details (using May as default month for consistency)
                month_num = "05"  # May
                quarter, duration = get_month_details(month_num)
                month_name = "May"
                
                # Get state details (exact same as summary_results)
                state_name, terrain = get_state_and_terrain(state_code)
                
                # Get the EXACT same rainfall prediction as summary_results
                flood_precipitation = get_rainfall_data(state_code, quarter)
                
                # Use the EXACT same flood prediction logic as summary_results
                flood_severity = predict_flood_severity(state_name, flood_precipitation, terrain)
                
                # Use the EXACT same drought prediction logic as summary_results (using same rainfall)
                drought_severity = predict_drought(city, str(year), month_name, flood_precipitation)
                
                # Use the EXACT same landslide prediction logic as summary_results (using same rainfall)
                landslide_location = city
                if city not in states:
                    state_code_for_landslide = city_to_state.get(city)
                    if state_code_for_landslide:
                        state_name_for_landslide, _ = get_state_and_terrain(state_code_for_landslide)
                        if state_name_for_landslide.title() in states:
                            landslide_location = state_name_for_landslide.strip().title()
                        else:
                            found_match = False
                            for state_in_list in states:
                                if state_in_list in state_name_for_landslide.strip().title():
                                    landslide_location = state_in_list
                                    found_match = True
                                    break
                            if not found_match:
                                landslide_location = city
                    else:
                        landslide_location = city
                
                try:
                    landslide_probability = predict_landslide(flood_precipitation, landslide_location.lower(), month_num)
                    if isinstance(landslide_probability, tuple):
                        landslide_probability = 0
                except Exception as e:
                    print(f"Error in landslide prediction: {e}")
                    landslide_probability = 0
                
                # Get groundwater data (exact same as summary_results)
                try:
                    groundwater_data = analyze_groundwater_risk(state_code)
                except Exception as e:
                    print(f"Error in groundwater analysis: {e}")
                    groundwater_data = {
                        'risk_level': "Analysis Unavailable",
                        'risk_percentage': 0,
                        'recharge_rate': 0,
                        'summary': "Unable to analyze groundwater risk due to technical issue"
                    }
                
                # Store rainfall data (use the same rainfall prediction for all quarters for consistency)
                quarters = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
                total_rainfall = 0
                for q in quarters:
                    rainfall = get_rainfall_data(state_code, q)
                    if rainfall is None or not isinstance(rainfall, (int, float)) or rainfall < 0:
                        rainfall = 0.0
                    year_predictions['rainfall'][q] = rainfall
                    total_rainfall += rainfall
                
                if total_rainfall is None or not isinstance(total_rainfall, (int, float)) or total_rainfall < 0:
                    total_rainfall = 0.0
                year_predictions['rainfall']['total'] = total_rainfall
                
                # Use the EXACT same probability conversion as summary_results
                # Flood probability (convert severity to percentage, then to 0-1)
                if flood_severity == 0:
                    flood_prob = 0.1
                elif flood_severity == 1:
                    flood_prob = 0.3
                elif flood_severity == 2:
                    flood_prob = 0.5
                elif flood_severity == 3:
                    flood_prob = 0.7
                elif flood_severity == 4:
                    flood_prob = 0.85
                elif flood_severity == 5:
                    flood_prob = 1.0
                else:
                    flood_prob = 0.0
                
                # Drought probability (convert severity text to percentage, then to 0-1)
                if drought_severity == "No Drought":
                    drought_prob = 0.1
                elif drought_severity == "Mild Drought":
                    drought_prob = 0.4
                elif drought_severity == "Moderate Drought":
                    drought_prob = 0.7
                elif drought_severity == "Severe Drought":
                    drought_prob = 1.0
                else:
                    drought_prob = 0.0
                
                # Landslide probability (use raw probability from model)
                landslide_prob = landslide_probability if landslide_probability is not None else 0.0
                
                # Erosion probability (calculate from R-factor using same rainfall)
                try:
                    erosion_r_factor = predict_r_factor(state, year, flood_precipitation)
                    if erosion_r_factor is None or not isinstance(erosion_r_factor, (int, float)):
                        erosion_prob = 0.0
                    else:
                        erosion_prob = min(erosion_r_factor / 1000, 1.0)
                except Exception as e:
                    print(f"Error in erosion prediction: {e}")
                    erosion_prob = 0.0
                
                # Groundwater probability (use percentage from analysis)
                groundwater_prob = groundwater_data['risk_percentage'] / 100.0 if groundwater_data['risk_percentage'] is not None else 0.0
                
                # Store all disaster probabilities
                year_predictions['disasters'] = {
                    'flood': flood_prob,
                    'drought': drought_prob,
                    'landslide': landslide_prob,
                    'erosion': erosion_prob,
                    'groundwater': groundwater_prob
                }
                
                # Final validation: ensure all probabilities are valid numbers
                for disaster_type, prob in year_predictions['disasters'].items():
                    if prob is None or not isinstance(prob, (int, float)) or prob < 0:
                        year_predictions['disasters'][disaster_type] = 0.0
                    elif prob > 1.0:
                        year_predictions['disasters'][disaster_type] = 1.0
                
                predictions.append(year_predictions)
            
            return render_template('long_term_predict.html', 
                                 location=city, 
                                 state=state,
                                 predictions=predictions,
                                 current_year=current_year)
        
        except Exception as e:
            return render_template('error.html', error=f"Error generating long-term predictions: {str(e)}")
    
    return render_template('long_term_form.html')

# Main entry point to run the Flask app
if __name__ == "__main__":
    create_encoder_and_scaler()  # Optional: initialize encoders/scalers for flood prediction
    initialize_groundwater_data()
    app.run(debug=True)