from __future__ import annotations

import numpy as np
import pandas as pd

from src.quant_signal import (
    align_series,
    granger_causality,
    lagged_correlations,
    simulate_klines,
)


def test_lagged_correlation_recovers_known_lag():
    index = pd.date_range("2026-01-01", periods=180, freq="h", tz="UTC")
    sentiment = pd.Series(np.sin(np.arange(len(index)) / 4.0) * 0.8)
    price = simulate_klines(index, sentiment=sentiment, seed=11)
    sentiment_frame = pd.DataFrame(
        {
            "timestamp": index,
            "sentiment": sentiment,
            "negative_ratio": (sentiment < 0).astype(float),
            "volume": 10,
        }
    )
    merged = align_series(sentiment_frame, price)
    lags = lagged_correlations(merged, max_lag=6)
    best = lags.loc[lags["correlation"].abs().idxmax()]
    assert int(best["lag"]) == 1
    assert float(best["correlation"]) > 0.5


def test_granger_detects_sentiment_lead():
    index = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    sentiment = pd.Series(np.sin(np.arange(len(index)) / 5.0) * 0.9)
    price = simulate_klines(index, sentiment=sentiment, seed=3, sigma=0.008)
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "sentiment": sentiment,
            "log_return": price["log_return"].to_numpy(),
        }
    )
    result = granger_causality(frame, max_lag=6)
    assert result["sufficient_data"] is True
    assert result["method"] == "numpy-ols-f-test"
    assert result["best_lag"] == 1
    assert result["min_p_value"] < 0.05
    assert result["significant"] is True


def test_granger_insufficient_data_is_safe():
    frame = pd.DataFrame({"log_return": [0.01, -0.02, 0.0], "sentiment": [0.1, -0.2, 0.0]})
    result = granger_causality(frame, max_lag=8)
    assert result["sufficient_data"] is False
    assert result["significant"] is False
