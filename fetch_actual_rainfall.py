from datetime import datetime
import io
import json
import zipfile

import requests
import pandas as pd
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
start_date = datetime(2026, 5, 9)
query_date_range = date_range_query_string(start_date, None)

url = f"{BASE_URL}/collections/30M/locations/{settings.COSMOS_UK_SITE_CODE}?{query_date_range}"

response = requests.get(url)
response_json = response.json()

site_data = response_json["coverages"][0]

site_id = site_data["dct:identifier"]

time_values = pd.DatetimeIndex(site_data["domain"]["axes"]["t"]["values"])
param_values = {param_name: param_data["values"] for param_name, param_data in site_data["ranges"].items()}

print(param_values)

pluvio_readings = param_values["precip"]
rainE_readings = param_values["precip_raine"]
tipping_readings = param_values["precip_tipping"]

rain_readings = [
    {"time": t, "pluvio": p, "rainE": r, "tipping": tp}
    for t, p, r, tp in zip(time_values, pluvio_readings, rainE_readings, tipping_readings, strict=True)
]

for row in rain_readings:
    print(f"{row['time']}\t{row['pluvio']}\t{row['rainE']}\t{row['tipping']}")
