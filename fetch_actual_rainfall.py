import csv
from datetime import datetime

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def date_range_query_string(start: datetime | None, end: datetime | None) -> str:
    match start, end:
        case datetime() as s, None:
            return f"datetime={format_datetime(s)}.."
        case None, datetime() as e:
            return f"datetime=..{format_datetime(e)}"
        case datetime() as s, datetime() as e:
            return f"datetime={format_datetime(s)}/{format_datetime(e)}"
        case _:
            raise TypeError("expected at least one datetime argument")

# Get COSMOS site location code from .env file
class Settings(BaseSettings):
    COSMOS_UK_SITE_CODE: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

BASE_URL = "https://cosmos-api.ceh.ac.uk"
start_date = datetime(2023, 5, 12)
query_date_range = date_range_query_string(start_date, None)
param_names = ["precip"]
params_string = ",".join(param_names)
param_constraints = f"parameter-name={params_string}"

url = f"{BASE_URL}/collections/30M/locations/{settings.COSMOS_UK_SITE_CODE}?{query_date_range}&{param_constraints}"

response = requests.get(url)
response_json = response.json()

site_data = response_json["coverages"][0]

time_values = site_data["domain"]["axes"]["t"]["values"]
param_values = {param_name: param_data["values"] for param_name, param_data in site_data["ranges"].items()}

pluvio_readings = param_values["precip"]

rain_readings = [
    {"time": t, "pluvio": p}
    for t, p in zip(time_values, pluvio_readings, strict=True)
]

with open("actual_rainfall.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rain_readings[0].keys())
    writer.writeheader()
    writer.writerows(rain_readings)
