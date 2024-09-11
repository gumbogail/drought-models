import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import sqlite3
import requests

# Restrict TensorFlow to only use the CPU
tf.config.set_visible_devices([], 'GPU')

# SQLite database connection
conn = sqlite3.connect('weather_data.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS WeatherData (
  id INTEGER PRIMARY KEY,
  date TEXT,
  rainfall REAL,
  lta REAL,
  std REAL,
  rainfall_anomaly REAL,
  spi REAL,
  drought_occurrence INTEGER,
  drought_severity INTEGER,
  month INTEGER,
  year INTEGER
)
''')

# URL of the historical rainfall dataset
historical_data_url = "https://raw.githubusercontent.com/gumbogail/FarmersGuide/testing123/newnewdataset.csv"

def fetch_historical_rainfall():
    """Fetches historical rainfall data from the URL"""
    historical_data = pd.read_csv(historical_data_url)
    return historical_data

def calculate_spi_and_lta(current_rainfall, historical_rainfall):
    """Calculates LTA, standard deviation, rainfall anomaly, and SPI"""
    lta = np.mean(historical_rainfall)
    std = np.std(historical_rainfall)
    rainfall_anomaly = current_rainfall - lta
    spi = rainfall_anomaly / std
    return lta, std, rainfall_anomaly, spi


# Load TensorFlow models (replace with your actual model paths)
# Make sure to use the raw string (r'path') or double backslashes to avoid escape sequences in file paths
drought_occurrence_model = tf.keras.models.load_model(r'C:\Users\abiga\OneDrive\Desktop\drought models\drought_occurrence_model.keras')
drought_severity_model = tf.keras.models.load_model(r'C:\Users\abiga\OneDrive\Desktop\drought models\drought_severity_model.keras')


def process_and_store_data(data):
    """Processes data and stores predictions in the SQLite database"""
    try:
        # Extract historical rainfall data
        historical_data = fetch_historical_rainfall()
        historical_rainfall = historical_data['totalprecip_mm'].values  # Adjust column name if needed
        historical_rainfall = historical_rainfall[-60:]  # Use the most recent 5 years of data for LTA and SPI

        # Extract current weather data from the API response
        current_rainfall = data['forecast']['forecastday'][0]['day']['totalprecip_mm']

        # Calculate LTA, STD, Anomaly, SPI
        lta, std, rainfall_anomaly, spi = calculate_spi_and_lta(current_rainfall, historical_rainfall)

        # Predict drought occurrence and severity
        occurrence_predictions = drought_occurrence_model.predict(np.array([[spi]]))  # Adjust input shape as needed
        severity_predictions = drought_severity_model.predict(np.array([[spi]]))  # Adjust input shape as needed

        drought_occurrence = int(occurrence_predictions[0][0] > 0.5)
        drought_severity = np.argmax(severity_predictions[0])

        # Current date and time
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        year = datetime.now().year
        month = datetime.now().month

        # Debugging: Print the values to be inserted
        print(f"Inserting data: {date}, {current_rainfall}, {lta}, {std}, {rainfall_anomaly}, {spi}, {drought_occurrence}, {drought_severity}, {month}, {year}")

        # Store data in SQLite database
        cursor.execute('''
            INSERT INTO WeatherData (date, rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, current_rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year))
        conn.commit()

    except Exception as e:
        print(f"Error during data processing: {e}")

# Fetch weather data from WeatherAPI
def fetch_weather_data():
    """Fetches weather data from WeatherAPI"""
    try:
        # Use your provided API key and Namibia as location
        api_url = f"http://api.weatherapi.com/v1/history.json?key=3a0d4c4e3e304fdd92e190019243007&q=namibia&dt={datetime.now().strftime('%Y-%m-%d')}"
        response = requests.get(api_url)
        weather_data = response.json()

        # Debugging: Print the API response to check its structure
        print("WeatherAPI response:")
        print(weather_data)

        # Call the function to process and store the data
        process_and_store_data(weather_data)

    except Exception as e:
        print(f"Error fetching data from WeatherAPI: {e}")

# Example of running the fetch function (you can schedule this to run daily)
fetch_weather_data()

# Manually check if data is inserted
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

# Manual test to insert data (for debugging purposes)
def insert_test_data():
    """Manually insert test data into the SQLite database"""
    test_data = ("2024-09-10 12:00:00", 20.5, 15.0, 3.2, 5.5, 1.72, 1, 2, 9, 2024)
    try:
        cursor.execute('''
            INSERT INTO WeatherData (date, rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', test_data)
        conn.commit()
        print("Test data inserted successfully.")
    except Exception as e:
        print(f"Error inserting test data: {e}")

# Call this function to manually insert test data
insert_test_data()
