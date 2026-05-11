from datetime import datetime
import io
import json
import zipfile

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get COSMOS site location code from .env file
class Settings(BaseSettings):
    COSMOS_UK_SITE_CODE: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

BASE_URL = "https://cosmos-api.ceh.ac.uk"
url = f"{BASE_URL}/collections/30M/locations/{settings.COSMOS_UK_SITE_CODE}"
