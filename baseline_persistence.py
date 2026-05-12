import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_HOURS = 4
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test = remaining 0.15

labels = pd.read_csv("labels.csv", parse_dates=["time"], index_col="time")

# Persistence prediction: assume next 4h matches the previous 4h.
# The label at row T-4h already encodes "did it rain in [T-4h, T)",
# so shifting forward by WINDOW_HOURS rows aligns it as the prediction for T.
predicted = labels["will_rain"].shift(WINDOW_HOURS)

# Temporal 70/15/15 split. Baseline only needs the test slice for fair
# comparison with a trained model, but we compute the split boundaries
# here so the same logic can be reused in the model script.
n = len(labels)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

test_slice = labels.iloc[val_end:]
print(f"Total rows: {n}")
print(f"Train: rows 0..{train_end - 1} ({labels.index[0]} to {labels.index[train_end - 1]})")
print(f"Val:   rows {train_end}..{val_end - 1} ({labels.index[train_end]} to {labels.index[val_end - 1]})")
print(f"Test:  rows {val_end}..{n - 1} ({labels.index[val_end]} to {labels.index[n - 1]})")
print()

eval_df = pd.DataFrame({
    "actual": labels["will_rain"],
    "predicted": predicted,
}).loc[test_slice.index].dropna()

y_true = eval_df["actual"].astype(bool)
y_pred = eval_df["predicted"].astype(bool)

print(f"Evaluated rows: {len(eval_df)}")
print(f"Actual positive rate: {y_true.mean():.1%}")
print(f"Predicted positive rate: {y_pred.mean():.1%}")

print("\nConfusion matrix (rows = actual, cols = predicted):")
cm = confusion_matrix(y_true, y_pred, labels=[False, True])
print(pd.DataFrame(
    cm,
    index=["actual_dry", "actual_rain"],
    columns=["pred_dry", "pred_rain"],
))

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["dry", "rain"], digits=3))
