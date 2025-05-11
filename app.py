from flask import Flask, redirect, url_for, render_template, request
import requests
import urllib
from datetime import datetime
import numpy as np
import tensorflow as tf
import sys
import os
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
            
        return render_template('drought_predict.html',
                             location=location,
                             date=date,
                             precipitation=precipitation,
                             severity=severity,
                             summary=summary)
                             
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
    else:
        location=request.args.get('location')
        rainfall=request.args.get('precipitation')
        final_prediction=request.args.get('final_prediction')
    
    if not all([location, rainfall,final_prediction]):
        return render_template('error.html', error="Missing required parameters from initial form")
    
    # print(f'Recevied {location} ,{rainfall} and {final_prediction}')
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
            landslide_probability = predict_landslide(flood_precipitation, landslide_location.strip())
            
            # Handle case where landslide_probability is a tuple instead of a number
            if isinstance(landslide_probability, tuple):
                landslide_probability = 0
                
            if landslide_probability >= 0.7:
                final_prediction = "Landslide risk exist"
                landslide_summary = "Based on our analysis your area is under threat of a severe landslide, evacuative measures are suggested"
            elif landslide_probability > 0 and landslide_probability < 0.3:
                final_prediction = "No landslide"
                landslide_summary = "Based on our analysis there is no landslide risk for your area, you can relax and enjoy the weather"
            elif landslide_probability >= 0.3 and landslide_probability < 0.5:
                final_prediction = "Mild risk"
                landslide_summary = "Based on our analysis there is a mild risk of landslide, stay updated with the latest news"
            elif landslide_probability >= 0.5 and landslide_probability < 0.7:
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
            landslide_summary=landslide_summary
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=f"An error occurred: {str(e)}")

# Main entry point to run the Flask app
if __name__ == "__main__":
    create_encoder_and_scaler()  # Optional: initialize encoders/scalers for flood prediction
    app.run(debug=True)