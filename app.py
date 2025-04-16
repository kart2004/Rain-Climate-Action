# Flask Libraries
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

# Import configuration
from flood.config import STATE_MAPPING, BING_API_KEY, OPENWEATHER_API_KEY

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'my secret and not your secret'

@app.route('/aboutus')
def about():
    return render_template("about.html")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/grantaccess', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        year = request.form['date'][:4]
        datetemp = datetime.strptime(request.form['date'], '%Y-%m-%d').strftime('%d/%m/%y')
        return redirect(url_for('jsonlocation', location=request.form['location'], date=datetemp, year=year))
    else:
        return redirect(url_for('index'))

@app.route('/location')
def jsonlocation():
    year = request.args.get('year')
    location = request.args.get('location')
    date = request.args.get('date')
    url = 'http://dev.virtualearth.net/REST/v1/Locations?'
    key = BING_API_KEY
    cr = 'IN'
    results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
    response = requests.get(results)
    parser = response.json()
    print("jsonlocation response\n", parser)
    auth = parser['statusDescription']
    if auth == 'OK':
        if 'adminDistrict' not in parser['resourceSets'][0]['resources'][0]['address']:
            return 'Location does not exist in India! Please try again!'
        state = parser['resourceSets'][0]['resources'][0]['address']['adminDistrict']
        lat = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][0]
        lon = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][1]
        city = parser['resourceSets'][0]['resources'][0]['address']['locality']
        return redirect(url_for('jsonweather', city=city, state=state, date=date, latitude=lat, longitude=lon, year=year))
    else:
        return 'Status: %s! Server issue! Please try again later!' % auth

@app.route('/weather')
def jsonweather():
    year = request.args.get('year')
    city = request.args.get('city')
    state = request.args.get('state')
    date = request.args.get('date')
    lat = request.args.get('latitude')
    lon = request.args.get('longitude')
    url = 'https://api.openweathermap.org/data/2.5/forecast?'
    key = OPENWEATHER_API_KEY
    mode = 'json'
    count = 32
    results = url + urllib.parse.urlencode(({'lat': lat, 'lon': lon, 'appid': key, 'mode': mode, 'cnt': count}))
    response = requests.get(results)
    
    parser = response.json()
    print("jsonweather response\n", parser)
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
    return redirect(url_for('interim', city=city, state=state, date=date, precipitation=round(totalprecipitation, 2), year=year))

@app.route('/interim', methods=['GET', 'POST'])
def interim():
    city = request.args.get('city')
    state = request.args.get('state')
    state_symbol = state
    date = request.args.get('date')
    precipitation = float(request.args.get('precipitation'))
    
    # Calculate month
    month = datetime.strptime(date, '%d/%m/%y').strftime('%m')
    actualMonth = month
    year = request.args.get('year')
    
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
        state_symbol=state_symbol
    ), code=307)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    state = request.args.get('state')
    precipitation = float(request.args.get('precipitation'))
    month = request.args.get('month')
    terrain = request.args.get('terrain')
    year = request.args.get('year')
    actualMonth = int(request.args.get('actualMonth'))
    state_symbol = request.args.get('state_symbol')
    
    # Optional parameters
    city = request.args.get('city')
    duration = float(request.args.get('duration'))
    
    # Normalize precipitation over the duration
    precipitation = normalize_precipitation(precipitation, duration)
    
    print(f"Looking for rainfall data: State={state_symbol}, Month={month}, Year={year}")

    # For years <= 2015, use historical data
    if int(year) <= 2015:
        precipitation, severity = get_historical_data(state, year, month, terrain)
        print("predict response: ", severity)
        
        return render_template(
            'flood_predict.html', 
            severity=str(severity), 
            city=city, 
            state=state, 
            month=month, 
            duration=duration, 
            precipitation=round(precipitation, 2), 
            terrain=terrain, 
            year=year
        )
    # For years > 2015, predict using model
    else:
        # Get rainfall data
        precipitation = get_rainfall_data(state_symbol, month)
        
        try:
            # Predict severity
            severity = predict_flood_severity(state, precipitation)
            
            return render_template(
                'flood_predict.html', 
                severity=str(severity), 
                city=city, 
                state=state, 
                month=month, 
                duration=duration, 
                precipitation=round(precipitation, 2), 
                terrain=terrain, 
                year=year
            )
        except Exception as e:
            print(f"Error in prediction: {e}")
            return render_template('error.html', error=str(e), city=city, state=state)

# Add new routes for other components here
# @app.route('/drought')
# def drought():
#     return render_template("drought.html")

if __name__ == "__main__":
    create_encoder_and_scaler()
    app.run()