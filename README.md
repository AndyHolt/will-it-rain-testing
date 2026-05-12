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
