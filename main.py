
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sqlite3

app = FastAPI()

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


if __name__ == "__main__":
    app.run(debug=True)