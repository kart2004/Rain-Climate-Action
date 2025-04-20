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

# Import drought-related functionality
from drought.drought import (
    predict_drought
)

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
@app.route('/grantaccess', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        try:
            # Get form data
            location = request.form.get('location')
            date = request.form.get('date')
            prediction_type = request.form.get('prediction_type')
            
            # Validate required fields
            if not all([location, date]):
                return render_template('error.html', error="Please provide both location and date")
            
            # Process date
            try:
                year = date[:4]
                datetemp = datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%y')
            except ValueError:
                return render_template('error.html', error="Invalid date format. Please use YYYY-MM-DD format")
            
            # Handle different prediction types
            if prediction_type == 'drought':
                return redirect(url_for('drought', 
                    location=location.strip(), 
                    date=datetemp, 
                    year=year
                ))
            else:  # flood prediction
                return redirect(url_for('jsonlocation', 
                    location=location.strip(), 
                    date=datetemp, 
                    year=year
                ))
        except Exception as e:
            return render_template('error.html', error=f"An error occurred: {str(e)}")
    else:
        return redirect(url_for('index'))

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
        longitude=longitude
    ), code=307)

# Flood prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    state = request.args.get('state')
    precip_str = request.args.get('precipitation')
    year = request.args.get('year')
    
    # Validate required parameters
    if not all([state, precip_str, year]):
        return render_template('error.html', error="Missing required parameters. Please ensure all fields are filled correctly.")
    
    try:
        precipitation = float(precip_str)
        year = int(year)
    except ValueError:
        return render_template('error.html', error="Invalid precipitation or year value")

    month = request.args.get('month')
    terrain = request.args.get('terrain')
    actualMonth = request.args.get('actualMonth')
    state_symbol = request.args.get('state_symbol')

    # Optional parameters
    city = request.args.get('city')
    duration = request.args.get('duration')
    
    # Validate duration
    if duration is not None:
        try:
            duration = float(duration)
        except ValueError:
            return render_template('error.html', error="Invalid duration value")

    # Normalize precipitation over the duration
    precipitation = normalize_precipitation(precipitation, duration)

    # For years <= 2015, use historical data
    if year <= 2015:
        precipitation, severity = get_historical_data(state, year, month, terrain)

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
            rf = int(precipitation / 2)
        )
    # For years > 2015, predict using model
    else:
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
                rf = precipitation/2
            )
        except Exception as e:
            return render_template('error.html', error=str(e), city=city, state=state)

# Route for Drought Prediction page
@app.route('/drought')
def drought():
    # Get parameters from the initial form
    location = request.args.get('location')
    date = request.args.get('date')
    year = request.args.get('year')
    
    if not all([location, date, year]):
        return render_template('error.html', error="Missing required parameters from initial form")
    
    try:
        # Process date to get month name
        month = datetime.strptime(date, '%d/%m/%y').strftime('%B')
        
        # Predict drought
        precipitation, severity = predict_drought(location, year, month)
        
        # Now use a single template for all cases
        return render_template(
            "drought_predict.html",
            state=location,
            year=year,
            month=month,
            precipitation=round(precipitation, 2) if severity != "Insufficient Data" else 0,
            severity=severity
        )
            
    except Exception as e:
        return render_template('error.html', error=str(e))

# Main entry point to run the Flask app
if __name__ == "__main__":
    create_encoder_and_scaler()  # Optional: initialize encoders/scalers for flood prediction
    app.run(debug=True)
