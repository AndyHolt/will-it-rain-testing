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
