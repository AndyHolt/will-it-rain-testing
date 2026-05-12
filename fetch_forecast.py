from math import comb
import openmeteo_requests
from openmeteo_sdk import Model
import pandas as pd
import requests_cache
from retry_requests import retry
from pydantic_settings import BaseSettings, SettingsConfigDict

# Read environment variables from .env file
class Settings(BaseSettings):
    LATITUDE: float
    LONGITUDE: float

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


def model_to_name(code: int):
    """convert Model enumeration to name"""
    for name, value in Model.Model.__dict__.items():
        if value == code:
            return name
    return None


# Setup API client with cache and retries
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# N.B. order of variables in hourly must match below for correct assignment
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]
models = [
    "best_match",
    "ecmwf_ifs",
    # best_match for specified location is identical to ukmo_uk... except that it also includes preciptation probability  column
    # "ukmo_uk_deterministic_2km",
]
params = {
    "latitude": settings.LATITUDE,
    "longitude": settings.LONGITUDE,
    "start_date": "2026-04-25",
    "end_date": "2026-05-09",
    "hourly": variables,
    "models": models,
}
responses = openmeteo.weather_api(url, params=params)

combined_dataframe = None

for response in responses:
    lat_direction = "N" if response.Latitude() > 0 else "S"
    long_direction = "E" if response.Longitude() > 0 else "W"

    print(f"Coordinates: {response.Latitude()}°{lat_direction} {response.Longitude()}°{long_direction}")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    model_id = response.Model()
    model_name = model_to_name(model_id)
    print(f"Model: {model_name}")

    # process hourly data
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq = pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    for i, variable_name in enumerate(variables):
        hourly_data[f"{model_name}__{variable_name}"] = hourly.Variables(i).ValuesAsNumpy()

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    if combined_dataframe is None:
        combined_dataframe = hourly_dataframe
    else:
        combined_dataframe = combined_dataframe.merge(hourly_dataframe, on="date", how="outer")

print("\nHourly data\n", combined_dataframe)

combined_dataframe.to_csv("forecast.csv", index=False, encoding="utf-8")
