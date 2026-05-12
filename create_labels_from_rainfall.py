import pandas as pd

WINDOW_HOURS = 4
RAIN_THRESHOLD_MM = 0.1

rainfall = pd.read_csv(
    "actual_rainfall.csv",
    parse_dates=["time"],
    index_col="time",
)

# Sum each pair of 30-min readings into hourly totals (mm/hour)
hourly_rainfall = rainfall["pluvio"].resample("1h").sum()

# For each hourly anchor T, find the max hourly total over [T, T+WINDOW_HOURS)
forward_max = pd.concat(
    [hourly_rainfall.shift(-i) for i in range(WINDOW_HOURS)],
    axis=1,
).max(axis=1)

labels = pd.DataFrame({
    "max_hourly_mm": forward_max,
    "will_rain": forward_max >= RAIN_THRESHOLD_MM,
}).dropna()

# Last WINDOW_HOURS-1 rows have an incomplete forward window
labels = labels.iloc[: -(WINDOW_HOURS - 1)] if WINDOW_HOURS > 1 else labels

print(labels)
print(f"\nRows: {len(labels)}")
print(f"Range: {labels.index.min()} to {labels.index.max()}")
print(f"Positive (rain) rate: {labels['will_rain'].mean():.1%}")

labels.to_csv("labels.csv")
