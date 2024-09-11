
# import json
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# import sqlite3

# app = FastAPI()

# class WeatherData(BaseModel):
#     date: str
#     year: int
#     month: int
#     drought_occurrence: int
#     drought_severity: int

# @app.get("/weather_data", response_model=WeatherData)
# async def get_weather_data():
#     try:
#         with sqlite3.connect('weather_data.db') as conn:
#             cursor = conn.execute("SELECT date, year, month, drought_occurrence, drought_severity FROM WeatherData ORDER BY id DESC LIMIT 1")
#             result = cursor.fetchone()

#         if result:
#             data = WeatherData(
#                 date=result[0],
#                 year=result[1],
#                 month=result[2],
#                 drought_occurrence=result[3],
#                 drought_severity=result[4]
#             )
#             return data
#         else:
#             raise HTTPException(status_code=404, detail="No data available")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     app.run(debug=True)

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import sqlite3
import numpy as np
import tensorflow as tf

# Load your models (replace with actual paths to your models)
drought_occurrence_model = tf.keras.models.load_model('path_to_occurrence_model')
drought_severity_model = tf.keras.models.load_model('path_to_severity_model')

app = FastAPI()

class WeatherData(BaseModel):
    date: str
    year: int
    month: int
    drought_occurrence: int
    drought_severity: int

class PredictionData(BaseModel):
    month: int
    year: int
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

@app.get("/predict_next_three_months", response_model=list[PredictionData])
async def predict_next_three_months():
    try:
        # Get the most recent weather data from the database
        with sqlite3.connect('weather_data.db') as conn:
            cursor = conn.execute("SELECT year, month FROM WeatherData ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="No historical data available for predictions")

        # Extract the current year and month
        current_year = result[0]
        current_month = result[1]

        # Generate predictions for the next three months
        predictions = []
        for i in range(1, 4):
            # Calculate the month and year for the prediction
            future_date = datetime(current_year, current_month, 1) + timedelta(days=30*i)
            future_year = future_date.year
            future_month = future_date.month

            # Dummy input data for predictions - replace with actual input data processing
            # You may need to adjust this depending on the input your model expects
            input_data = np.random.rand(1, 10)  # Assuming your model takes an input of shape (1, 10)

            # Predict drought occurrence and severity
            occurrence_prediction = drought_occurrence_model.predict(input_data)
            severity_prediction = drought_severity_model.predict(input_data)

            # Convert predictions to appropriate output
            drought_occurrence = int(occurrence_prediction[0] > 0.5)
            drought_severity = np.argmax(severity_prediction[0])

            # Store prediction data
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
    app.run(debug=True)
