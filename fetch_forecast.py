import openmeteo_requests
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


# Setup API client with cache and retries
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# N.B. order of variables in hourly must match below for correct assignment
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": settings.LATITUDE,
    "longitude": settings.LONGITUDE,
    "start_date": "2026-04-25",
    "end_date": "2026-05-09",
    "hourly": [
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
    ],
}
responses = openmeteo.weather_api(url, params=params)

# process first location & model
response = responses[0]

lat_direction = "N" if response.Latitude() > 0 else "S"
long_direction = "E" if response.Longitude() > 0 else "W"

print(f"Coordinates: {response.Latitude()}°{lat_direction} {response.Longitude()}°{long_direction}")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")


# process hourly data
hourly = response.Hourly()

hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
hourly_rain = hourly.Variables(5).ValuesAsNumpy()
hourly_showers = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq = pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["precipitation_probability"] = hourly_precipitation_probability
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["showers"] = hourly_showers
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)
