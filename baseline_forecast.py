import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
PRECIP_THRESHOLD_MM = 0.1
PRECIP_PROB_THRESHOLD = 50.0

labels = pd.read_csv("labels.csv", parse_dates=["time"], index_col="time")
forecast = pd.read_csv("forecast.csv", parse_dates=["date"], index_col="date")

joined = labels.join(forecast, how="inner")

# Same temporal 70/15/15 split as baseline_persistence.py
n = len(joined)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
test = joined.iloc[val_end:]

print(f"Test rows: {len(test)} ({test.index[0]} to {test.index[-1]})")
print(f"Actual positive rate: {test['will_rain'].mean():.1%}\n")

y_true = test["will_rain"].astype(bool)


def evaluate(name: str, y_pred: pd.Series) -> None:
    print(f"=== {name} ===")
    print(f"Predicted positive rate: {y_pred.mean():.1%}")
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    print(pd.DataFrame(
        cm,
        index=["actual_dry", "actual_rain"],
        columns=["pred_dry", "pred_rain"],
    ))
    print(classification_report(y_true, y_pred, target_names=["dry", "rain"], digits=3))


evaluate(
    f"best_match precipitation @ T  >=  {PRECIP_THRESHOLD_MM}mm",
    test["best_match__precipitation"] >= PRECIP_THRESHOLD_MM,
)

evaluate(
    f"best_match precipitation_probability @ T  >=  {PRECIP_PROB_THRESHOLD}%",
    test["best_match__precipitation_probability"] >= PRECIP_PROB_THRESHOLD,
)
