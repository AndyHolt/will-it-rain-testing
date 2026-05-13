# Data & model development for local rain prediction

## Problem and opportunity

I often want to know if it's going to rain. Is it worth hanging washing out to
dry? Should I go for a run soon, or wait until later or tomorrow? Should we put
the kids' waterproofs on before we go out for a walk?

Hyperlocal forecasts are pretty good now, so why not just use those? The saying
goes that a man who wears two watches never knows the time. And the problem is I
tend to look at Apple weather's rain forecast, while my wife tends to look at
the BBC weather forecast, and they regularly don't agree.

What I've noticed in comparing the Apple and BBC forecasts is that Apple tends
to be over-optimistic, indicating it will be dry when a shower is coming, while
the BBC tends to be pessimistic, often indicating rain when it stays dry.

This suggests that there's something more than just imperfect forecasting going
on, and the systematic errors could be improved using local data. Weather models
operate in relatively large areas. Tuning to specific local data offers
the possibility of improving the local forecast.

I don't have a home weather station to gather data from our house. But there's
something potentially even better: we happen to live a few hundred metres from a
research grade weather monitoring station which publishes open data, including
rainfall.

## Getting the data

Unfortunately, neither Apple weather nor BBC publish their historical forecasts,
so I can't easily get a good amount of historical data for training a model from
the forecasts I've noticed systematic bias.

But Open-Meteo publishes historical data from a number of major weather models,
which I believe are used by both BBC and Apple to generate their own forecasts.

The ideal data set to use would be "Previous Model Runs", but this has limited
data availability, back to a few months.

So I'm starting with the "Historical Forecast" data from a couple of models,
which gives multiple years of data, plenty for an initial validation.

The big limitation of the "Historical Forecasts" vs "Previous Model Runs" is
that "Historical Forecasts" gives the shortest-term forecast at every hour,
essentially giving the "next hour" forecast at all points. So if including the
whole four-hour window of interest in training data, this is sneaking in a
forecast for hour T+4h that won't be known for another 3 hours.

This is the big advantage that "Previous Model Runs" provides, allowing training
on actual forecasts for (T+1h, T+2h, T+3h, T+4h), reflecting the data available
at inference time.

But for simplicity of features, timescale available, and to test the overall
framework, it's still worth starting out with the simplest version.

If reasonably successful, I can start regularly ingesting "Previous Model Runs"
data, along with potentially the BBC and Apple Weather forecasts to use those in
the future when I've built up enough historical data to train. (That requires
enough data for training, especially enough examples of actual rainfall, which
is a relatively sparse feature. And it also needs to account for different
seasons to get the full range of features.)

## Generating labels

I want to work in 4 hour windows for determining if it's going to rain. That
feels like a reasonable time period to consider the range of useful windows,
between various activities like going for a run, a family walk, a bike ride,
mowing the lawn, hanging out washing, or taking the right gear for an outing.

I'm also taking 0.1mm rainfall as the boundary. A few minutes of *very* light
drizzle is tolerable, and provides a bit of buffer around iffy periods, but also
keeps the threshold low, so it's predicting *all* rain, not just moderate rain.

After applying these thresholds, I've got a 22.1% positive rain rate, which is
balanced enough for training.

## Model training and selection

Because this is time series forecasting, data needs to be split for
training/validation temporally, not randomly. Otherwise the test data being
evaluated will be strongly correlated with the actual test data which surrounds
it, giving an inflated validation result, and must poorer performance in
inference. So first 70% of data will be used for training, next 15% for
validation, and final 15% for testing.

This split has a significant downside: the final 15% used for testing is
seasonally isolated, covering only a segment of winter/spring. It also will give
a poor data set if the underlying weather models have been updated, such that
all training data reflects an older weather model.

### Baseline persistence

Naive baseline for model comparison: if it rained in period (T-4h, T), it will
also rain in (T, T+4h). If dry in (T-4h, T), it will be dry in (T, T+4h).

Evaluation results:

```
Evaluated rows: 3946
Actual positive rate: 27.0%
Predicted positive rate: 27.0%

Confusion matrix (rows = actual, cols = predicted):
             pred_dry  pred_rain
actual_dry       2489        393
actual_rain       393        671

Classification report:
              precision    recall  f1-score   support

         dry      0.864     0.864     0.864      2882
        rain      0.631     0.631     0.631      1064

    accuracy                          0.801      3946
   macro avg      0.747     0.747     0.747      3946
weighted avg      0.801     0.801     0.801      3946
```

### Baseline forecast

Next baseline: use Open-Meteo's "Best match" forecast. This validates that the
forecast data provides more signal than noise for predicting rain locally. It
also gives a more challenging baseline for training models. If there are genuine
systematic discrepancies between various forecasts and the local weather, a
model trained across the different forecasts should have better prediction than
a single forecast.

One major caveat on this baseline is the "Historical Forecasts" data strongly
favours forecasts 1 hour ahead, and we can't use forecasts for all four hours of
the window of interest, since those forecasts aren't yet available at the
beginning of the window. So we're using the metric "will it rain in the next
hour?" as a proxy for "will it rain in the next four hours?". This will
systematically under-report.

Two different versions have been run, one using the predicted precipitation, and
the other using precipitation probability.

Evaluation results:

```
=== best_match precipitation @ T  >=  0.1mm ===
Predicted positive rate: 24.4%
             pred_dry  pred_rain
actual_dry       2621        261
actual_rain       361        703
              precision    recall  f1-score   support

         dry      0.879     0.909     0.894      2882
        rain      0.729     0.661     0.693      1064

    accuracy                          0.842      3946
   macro avg      0.804     0.785     0.794      3946
weighted avg      0.839     0.842     0.840      3946

=== best_match precipitation_probability @ T  >=  50.0% ===
Predicted positive rate: 27.3%
             pred_dry  pred_rain
actual_dry       2554        328
actual_rain       315        749
              precision    recall  f1-score   support

         dry      0.890     0.886     0.888      2882
        rain      0.695     0.704     0.700      1064

    accuracy                          0.837      3946
   macro avg      0.793     0.795     0.794      3946
weighted avg      0.838     0.837     0.837      3946

```

### LightGBM model

Since this is a tabular classification problem, gradient boosting using LightGBM
is the obvious choice, especially given the relatively small pool of training
data which favours gradient boosting over a neural net.

The first run of training was disappointing, but had an obvious issue: a much
smaller training set was used than expected. This was because `NaN` values were
being dropped, and all data from before June 2025 had `NaN` for
`precipitation_probability` values: a key feature for training. Including the
whole training set also gave poor performance, since the training data is
largely training on rows which lack this key feature, then tested on features
which include it.

A second ill-effect of the limited data is that the partitioning of
training/validation/test data is much more strongly seasonal.

There's nothing much we can do about the missing data problem in the short term,
so we've got to live with the data we've got and keep moving forwards. But we
can take some measures to mitigate the seasonality of the data. (At this point,
I'm leaning more heavily on Claude Code to guide improvements, for two reasons.
Firstly, I'm not an expert in time series forecasting or gradient boosting, and
I don't need to be because, secondly, we're very much in the exploration and
prototyping phase, so we can try lots of things quickly and see what works well.
The goal here is not to have a perfect method and model, but to quickly get to a
usable model and get it deployed. Model improvement can come later, as I built
up more familiarity with best practices in gradient boosting and time series
forecasting.)

The first step in improving on the seasonal mismatch between training (dryer)
and testing (wetter) data is to use a threshold that matches the positive rate
on the validation data (how many wet periods) rather than maximising F1 in the
validation data, since the distributions are not aligned. This improves F1 of
test data from 0.588 to 0.672: a substantial improvement and brings our trained
model above continuity baseline. It's not as good as the forecast baseline
(without `precipitation_probability`), but pretty close. The issue now is that
the validation dataset in which we're matching wetness percentage doesn't match
the test dataset: validation is wetter than test. So the next improvement is to
add isotonic calibration.

After adding isotonic calibration, the results are looking very encouraging:
```
=== Test set results ===
Predicted positive rate: 32.0%  (actual 27.0%)
             pred_dry  pred_rain
actual_dry       2486        396
actual_rain       196        868
              precision    recall  f1-score   support

         dry      0.927     0.863     0.894      2882
        rain      0.687     0.816     0.746      1064

    accuracy                          0.850      3946
   macro avg      0.807     0.839     0.820      3946
weighted avg      0.862     0.850     0.854      3946
```
F-1 score is now well improved over the persistence baseline, and also beating the
precipitation depth and probability forecast baselines. The overall methodological
limitations still apply: we're using only the first hour forecast to predict
rain over the whole 4 hour window. But with the current available data, this
gives enough model validation to use this as a first version.

### Model performance progression

| Model                                         | Test F1 (rain) |
|-----------------------------------------------+----------------|
| Persistence                                   |          0.631 |
| Forecast: precipitation @ T >= 0.1mm          |          0.693 |
| Forecast: precipitation_probability @ T >= 50% |          0.700 |
| LightGBM with F1 max threshold                |          0.588 |
| LightGBM with rank based threshold            |          0.672 |
| LightGBM + isotonic calibration               |          0.746 |
