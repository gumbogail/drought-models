from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import sqlite3
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Initialize FastAPI app
app = FastAPI()

# Load your models
drought_occurrence_model = tf.keras.models.load_model('drought_occurrence_model.keras')
drought_severity_model = tf.keras.models.load_model('drought_severity_model.keras')

class WeatherData(BaseModel):
    date: str
    year: int
    month: int
    drought_occurrence: int
    drought_severity: int

@app.get("/weather_data", response_model=WeatherData)
async def get_weather_data():
    try:
        with sqlite3.connect('weather_data.db') as conn:
            cursor = conn.execute("SELECT date, year, month, drought_occurrence, drought_severity FROM WeatherData ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()

        if result:
            data = WeatherData(
                date=result[0],
                year=result[1],
                month=result[2],
                drought_occurrence=result[3],
                drought_severity=result[4]
            )
            return data
        else:
            raise HTTPException(status_code=404, detail="No data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictionData(BaseModel):
    month: int
    year: int
    drought_occurrence: float
    drought_severity: int

@app.get("/predict_next_three_months", response_model=List[PredictionData])
async def predict_next_three_months(latitude: float, longitude: float):
    try:
        # Get the most recent weather data from the database
        with sqlite3.connect('weather_data.db') as conn:
            cursor = conn.execute("SELECT year, month FROM WeatherData ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="No historical data available for predictions")

        current_year = result[0]
        current_month = result[1]

        predictions = []
        for i in range(1, 4):
            # Handle month and year progression using relativedelta
            future_date = datetime(current_year, current_month, 1) + relativedelta(months=i)
            future_year = future_date.year
            future_month = future_date.month

            # Fetch real input features for prediction from the database
            with sqlite3.connect('weather_data.db') as conn:
                cursor = conn.execute(f"""
                    SELECT rainfall, lta, std, rainfall_anomaly, spi, drought_occurrence, drought_severity, month, year
                    FROM WeatherData
                    WHERE year = {current_year} AND month = {current_month}
                    ORDER BY id DESC LIMIT 1
                """)
                feature_result = cursor.fetchone()

            if not feature_result:
                raise HTTPException(status_code=404, detail="No feature data available")

            # Create input array of features including latitude and longitude
            input_data = np.array([list(feature_result) + [latitude, longitude]]).astype(np.float32)

            # Ensure the input shape is (1, 13) if you added 2 more features
            if input_data.shape[1] != 11:
                raise ValueError(f"Expected input shape (1, 11), but got {input_data.shape}")

            # Predict drought occurrence and severity
            occurrence_prediction = drought_occurrence_model.predict(input_data)
            severity_prediction = drought_severity_model.predict(input_data)

            # Extract predictions
            drought_occurrence = float(occurrence_prediction[0][0])
            drought_severity = int(np.argmax(severity_prediction[0]))

            # Create PredictionData object
            prediction = PredictionData(
                month=future_month,
                year=future_year,
                drought_occurrence=drought_occurrence,
                drought_severity=drought_severity
            )
            predictions.append(prediction)

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
