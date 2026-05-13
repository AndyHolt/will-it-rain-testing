from datetime import datetime, timezone

import joblib
import openmeteo_requests
import pandas as pd
import requests_cache
from openmeteo_sdk import Model
from pydantic_settings import BaseSettings, SettingsConfigDict
from retry_requests import retry


class Settings(BaseSettings):
    LATITUDE: float
    LONGITUDE: float

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


def model_to_name(code: int):
    for name, value in Model.Model.__dict__.items():
        if value == code:
            return name
    return None


VARIABLES = [
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
WEATHER_MODELS = ["best_match", "ecmwf_ifs"]

bundle = joblib.load("model_bundle.joblib")
gbm_model = bundle["model"]
calibrator = bundle["calibrator"]
threshold = bundle["threshold"]
feature_cols = bundle["feature_cols"]
lag_hours = bundle["lag_hours"]
sparse_columns = bundle["sparse_columns"]

# Short cache (10 min) — we want fresh data at inference time.
cache_session = requests_cache.CachedSession(".cache", expire_after=600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Live forecast endpoint: gives current and recent forecasts from the latest
# model runs. past_hours covers the lag window; forecast_hours is small since
# we only use the anchor row's features (the lead-time-stratified forecasts
# from T+1..T+4 are not used as features, per the lead-time discussion).
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": settings.LATITUDE,
    "longitude": settings.LONGITUDE,
    "past_hours": 6,
    "forecast_hours": 4,
    "hourly": VARIABLES,
    "models": WEATHER_MODELS,
}
responses = openmeteo.weather_api(url, params=params)

combined = None
for response in responses:
    model_name = model_to_name(response.Model())
    hourly = response.Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, variable_name in enumerate(VARIABLES):
        data[f"{model_name}__{variable_name}"] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    combined = df if combined is None else combined.merge(df, on="date", how="outer")

combined = combined.set_index("date").sort_index()
combined = combined.drop(columns=[c for c in sparse_columns if c in combined.columns])

# Build features identically to training: anchor row + lag1/2/3 + hour + month.
features = combined.copy()
for lag in lag_hours:
    features = features.join(combined.shift(lag).add_suffix(f"__lag{lag}h"))
features["hour_of_day"] = features.index.hour
features["month"] = features.index.month

# Anchor on the current hour (the hour containing "now"). The prediction
# window [T, T+4h] then covers from up to 1h ago through ~3h ahead, which
# for a 4-hour-ahead "will it rain" question correctly includes "right now".
now = datetime.now(tz=timezone.utc)
current_hour = now.replace(minute=0, second=0, microsecond=0)
anchor_cols = list(combined.columns)
candidates = features.dropna(subset=anchor_cols, how="any")
eligible = candidates[candidates.index >= current_hour]
if eligible.empty:
    raise RuntimeError("No forecast hours available with populated features for the current hour.")

anchor_time = eligible.index[0]
X_now = eligible.loc[[anchor_time], feature_cols]

prob_raw = float(gbm_model.predict_proba(X_now)[:, 1][0])
prob_calibrated = float(calibrator.transform([prob_raw])[0])
will_rain = prob_calibrated >= threshold

window_end = anchor_time + pd.Timedelta(hours=4)
# astimezone() with no arg converts to the system local timezone.
now_local = now.astimezone()
anchor_local = anchor_time.to_pydatetime().astimezone()
window_end_local = window_end.to_pydatetime().astimezone()
tz_name = now_local.tzname()
print(f"Now:                  {now_local.replace(microsecond=0)} ({tz_name})")
print(f"Anchor (T):           {anchor_local} ({tz_name})   [{anchor_time} UTC]")
print(f"Prediction window:    {anchor_local}  ->  {window_end_local} ({tz_name})")
print(f"Raw probability:      {prob_raw:.3f}")
print(f"Calibrated probability: {prob_calibrated:.3f}")
print(f"Threshold:            {threshold:.3f}")
print()
print(f"Prediction: {'RAIN likely' if will_rain else 'likely DRY'} in the next 4 hours")
