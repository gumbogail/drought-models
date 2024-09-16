import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import sqlite3
import requests

# Initialize TensorFlow models
drought_occurrence_model = tf.keras.models.load_model('drought_occurrence_model.keras')
drought_severity_model = tf.keras.models.load_model('drought_severity_model.keras')

# SQLite database connection
conn = sqlite3.connect('weather_data.db')
cursor = conn.cursor()

def fetch_historical_rainfall():
    """Fetches historical rainfall data from the URL"""
    historical_data = pd.read_csv("https://raw.githubusercontent.com/gumbogail/FarmersGuide/testing123/newnewdataset.csv")
    return historical_data

def calculate_spi_and_lta(current_rainfall, historical_rainfall):
    """Calculates LTA, standard deviation, rainfall anomaly, and SPI"""
    lta = np.mean(historical_rainfall)
    std = np.std(historical_rainfall)
    rainfall_anomaly = current_rainfall - lta
    spi = rainfall_anomaly / std
    return lta, std, rainfall_anomaly, spi

def process_and_store_data(data, latitude, longitude):
    """Processes data and stores predictions in the SQLite database"""
    try:
        # Extract historical rainfall data
        historical_data = fetch_historical_rainfall()
        historical_rainfall = historical_data['totalprecip_mm'].values
        historical_rainfall = historical_rainfall[-60:]

        # Extract current weather data from the API response
        current_rainfall = data['forecast']['forecastday'][0]['day']['totalprecip_mm']

        # Calculate LTA, STD, Anomaly, SPI
        lta, std, rainfall_anomaly, spi = calculate_spi_and_lta(current_rainfall, historical_rainfall)

        # Predict drought occurrence and severity
        occurrence_predictions = drought_occurrence_model.predict(np.array([[spi, latitude, longitude]]))
        severity_predictions = drought_severity_model.predict(np.array([[spi, latitude, longitude]]))

        drought_occurrence = int(occurrence_predictions[0][0] > 0.5)
        drought_severity = np.argmax(severity_predictions[0])

        # Current date and time
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        year = datetime.now().year
        month = datetime.now().month

        # Store data in SQLite database
        cursor.execute('''
            INSERT INTO WeatherData (date, rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, current_rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year, latitude, longitude))
        conn.commit()

    except Exception as e:
        print(f"Error during data processing: {e}")

def fetch_weather_data(latitude, longitude):
    """Fetches weather data from WeatherAPI"""
    try:
        api_url = f"http://api.weatherapi.com/v1/history.json?key=3a0d4c4e3e304fdd92e190019243007&q={latitude},{longitude}&dt={datetime.now().strftime('%Y-%m-%d')}"
        response = requests.get(api_url)
        weather_data = response.json()

        # Call the function to process and store the data
        process_and_store_data(weather_data, latitude, longitude)

    except Exception as e:
        print(f"Error fetching data from WeatherAPI: {e}")

# Example usage with hardcoded latitude and longitude (update as needed)
fetch_weather_data(18.4930, -22.9671)

# Check database
def check_database():
    """Check the SQLite database to see if data has been inserted"""
    try:
        with sqlite3.connect('weather_data.db') as conn:
            cursor = conn.execute("SELECT * FROM WeatherData")
            result = cursor.fetchall()
            print("Database content:")
            print(result)
    except Exception as e:
        print(f"Error reading database: {e}")

# Call this function to check the database content
check_database()
