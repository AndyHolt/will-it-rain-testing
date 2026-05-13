import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
LAG_HOURS = [1, 2, 3]
RANDOM_SEED = 42

# Columns only available for a fraction of the historical range. Including them
# would either force dropping older data or create a distribution shift across
# train/val/test (the feature is mostly NaN early, mostly present late).
SPARSE_COLUMNS = [
    "best_match__precipitation_probability",
    "ecmwf_ifs__precipitation_probability",
    "ecmwf_ifs__showers",
]

labels = pd.read_csv("labels.csv", parse_dates=["time"], index_col="time")
forecast = pd.read_csv("forecast.csv", parse_dates=["date"], index_col="date")
forecast = forecast.drop(columns=SPARSE_COLUMNS)

# Build feature frame. Following the lead-time discussion, we use only the
# forecast at the anchor T plus lagged values from T-1..T-3 (which at inference
# would have similar lead-time profile to the historical archive). No forward-
# window features (T+1..T+4) — those would create train/inference mismatch.
features = forecast.copy()
for lag in LAG_HOURS:
    lagged = forecast.shift(lag).add_suffix(f"__lag{lag}h")
    features = features.join(lagged)

features["hour_of_day"] = features.index.hour
features["month"] = features.index.month

# LightGBM handles NaN features natively, so only drop rows where the
# label is missing. Several forecast columns (precipitation_probability,
# ecmwf_ifs__showers) are sparse in the historical archive — dropping
# them would throw away most pre-2025 data.
dataset = features.join(labels["will_rain"], how="inner")
dataset = dataset[dataset["will_rain"].notna()]

n = len(dataset)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

train = dataset.iloc[:train_end]
val = dataset.iloc[train_end:val_end]
test = dataset.iloc[val_end:]

feature_cols = [c for c in dataset.columns if c != "will_rain"]

X_train, y_train = train[feature_cols], train["will_rain"].astype(int)
X_val, y_val = val[feature_cols], val["will_rain"].astype(int)
X_test, y_test = test[feature_cols], test["will_rain"].astype(int)

print(f"Train: {len(train)} rows ({train.index[0]} to {train.index[-1]})")
print(f"Val:   {len(val)} rows ({val.index[0]} to {val.index[-1]})")
print(f"Test:  {len(test)} rows ({test.index[0]} to {test.index[-1]})")
print(f"Features: {len(feature_cols)}")
print(f"Train rain rate: {y_train.mean():.1%}  Val: {y_val.mean():.1%}  Test: {y_test.mean():.1%}\n")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_SEED,
    verbose=-1,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
print(f"Best iteration: {model.best_iteration_}\n")

# Tune classification threshold on the validation set, then apply to test.
val_probs = model.predict_proba(X_val)[:, 1]
candidate_thresholds = np.linspace(0.05, 0.95, 91)
val_f1s = [f1_score(y_val, val_probs >= t) for t in candidate_thresholds]
best_threshold = candidate_thresholds[int(np.argmax(val_f1s))]
print(f"Best threshold (max F1 on val): {best_threshold:.2f}  (val F1 = {max(val_f1s):.3f})\n")

test_probs = model.predict_proba(X_test)[:, 1]
y_pred = test_probs >= best_threshold

print("=== Test set results ===")
print(f"Predicted positive rate: {y_pred.mean():.1%}  (actual {y_test.mean():.1%})")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(pd.DataFrame(
    cm,
    index=["actual_dry", "actual_rain"],
    columns=["pred_dry", "pred_rain"],
))
print(classification_report(y_test, y_pred, target_names=["dry", "rain"], digits=3))

# Diagnostic: what is the achievable F1 on test if the threshold were tuned
# directly on test? This is an OPTIMISTIC ceiling — never use this number as a
# real estimate. It only tells us whether the model has signal at all.
oracle_f1s = [(t, f1_score(y_test, test_probs >= t)) for t in candidate_thresholds]
oracle_best_t, oracle_best_f1 = max(oracle_f1s, key=lambda x: x[1])
print(f"DIAGNOSTIC: best achievable test F1 (threshold tuned on test): {oracle_best_f1:.3f} at t={oracle_best_t:.2f}\n")

print("Top 15 feature importances:")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance.head(15).to_string())
