"""Sentiment ↔ price lead-lag analysis and Granger causality — dimension 5.

Two data modes:

* ``live``      Binance public klines (no API key required)
* ``simulated`` deterministic GBM whose drift is *optionally* coupled to lagged
                sentiment, so the notebook is reproducible offline. The coupling
                is disclosed in the figure title and README — it validates the
                method, not a market claim.

Granger test orientation: H0 = sentiment does **not** Granger-cause returns.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def load_sentiment_series(
    path: str | Path,
    entity: str | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Aggregate processed comments into a sentiment time series."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entity and entity not in (row.get("entities") or []):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No processed comments found in {path} (entity={entity})")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [row["created_at"] for row in rows], utc=True, format="mixed"
            ),
            "sentiment": [float(row.get("sentiment", 0.0)) for row in rows],
            "negative": [1.0 if row.get("negative") else 0.0 for row in rows],
        }
    )
    rule = {"1h": "1h", "30m": "30min", "15m": "15min", "4h": "4h", "1d": "1D"}.get(
        freq, freq
    )
    series = (
        frame.set_index("timestamp")
        .resample(rule)
        .agg(
            sentiment=("sentiment", "mean"),
            negative_ratio=("negative", "mean"),
            volume=("sentiment", "size"),
        )
        .dropna(subset=["sentiment"])
        .reset_index()
    )
    return series


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: datetime | None = None,
    end: datetime | None = None,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch public Binance klines with pagination (max 1000 rows per call)."""

    import httpx

    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=7))
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []

    with httpx.Client(timeout=timeout) as client:
        while cursor < end_ms:
            response = client.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + 1
            if len(batch) < 1000:
                break

    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def simulate_klines(
    index: pd.DatetimeIndex,
    sentiment: pd.Series | None = None,
    seed: int = 7,
    start_price: float = 65_000.0,
    sigma: float = 0.012,
    beta: float = 0.03,
) -> pd.DataFrame:
    """Deterministic GBM; when ``sentiment`` is given, drift uses t-1 sentiment.

    This coupling is synthetic and exists only to make the offline notebook
    demonstrate that the estimator recovers a known lead-lag structure.
    """

    rng = np.random.default_rng(seed)
    n = len(index)
    shock = rng.normal(0.0, sigma, n)
    drift = np.zeros(n)
    if sentiment is not None and len(sentiment) == n:
        lagged = sentiment.fillna(0.0).to_numpy()
        drift[1:] = beta * lagged[:-1]
    log_returns = drift + shock
    close = start_price * np.exp(np.cumsum(log_returns))
    open_price = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(mean=4.0, sigma=0.4, size=n)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    frame["log_return"] = np.log(frame["close"]).diff().fillna(0.0)
    return frame


def align_series(sentiment: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """As-of alignment on hourly buckets + log returns."""

    sent = sentiment.sort_values("timestamp").copy()
    prices = price.sort_values("timestamp").copy()
    if "log_return" not in prices:
        prices["log_return"] = np.log(prices["close"]).diff()
    merged = pd.merge_asof(
        sent,
        prices[["timestamp", "close", "log_return"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("90min"),
    ).dropna(subset=["close", "log_return"])
    return merged.reset_index(drop=True)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Continued-fraction implementation (Numerical Recipes), no SciPy needed."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    max_iter, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(x: float, a: float, b: float) -> float:
    from math import exp, lgamma, log

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    front = exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _regularized_incomplete_beta(x, a, b) / a
    return 1.0 - front * _regularized_incomplete_beta(1.0 - x, b, a) / b


def _student_t_sf(t: float, df: int) -> float:
    """P(T > |t|) for a two-sided test."""

    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return _betainc(x, df / 2.0, 0.5)


def _f_sf(f_stat: float, df1: int, df2: int) -> float:
    """P(F > f_stat) = I_{df2/(df2+df1*f)}(df2/2, df1/2)."""

    if f_stat <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    return _betainc(x, df2 / 2.0, df1 / 2.0)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((x_centered**2).sum() * (y_centered**2).sum()))
    if denom == 0:
        return 0.0, 1.0
    r = float((x_centered * y_centered).sum() / denom)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / (1.0 - r * r))
    return r, float(min(1.0, _student_t_sf(abs(float(t_stat)), n - 2) * 2.0))


def _ols_rss(y: np.ndarray, design: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    return float((residuals**2).sum())


def _adf_pvalue(series: np.ndarray, max_lag: int = 4) -> float:
    """Augmented Dickey-Fuller t-test (constant, no trend), normal approximation."""

    series = np.asarray(series, dtype=float)
    n = len(series)
    lag = min(max_lag, max(1, n // 5))
    if n < lag + 8:
        return 1.0
    delta = np.diff(series)
    rows, targets = [], []
    for t in range(lag, len(delta)):
        row = [1.0, series[t]]
        row.extend(delta[t - j] for j in range(1, lag + 1))
        rows.append(row)
        targets.append(delta[t])
    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    dof = max(1, len(target) - design.shape[1])
    sigma2 = float((residuals**2).sum()) / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        return 1.0
    if se == 0:
        return 1.0
    t_stat = abs(float(coefficients[1]) / se)
    return float(min(1.0, _student_t_sf(t_stat, dof) * 2.0))


def lagged_correlations(frame: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Correlation between sentiment at t and return at t+lag."""

    rows: list[dict[str, Any]] = []
    for lag in range(0, max_lag + 1):
        future_return = frame["log_return"].shift(-lag)
        pair = pd.concat([frame["sentiment"], future_return], axis=1).dropna()
        if len(pair) < 8 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
            corr, pvalue = 0.0, 1.0
        else:
            corr, pvalue = _pearson(
                pair.iloc[:, 0].to_numpy(dtype=float),
                pair.iloc[:, 1].to_numpy(dtype=float),
            )
        rows.append(
            {
                "lag": lag,
                "correlation": round(float(corr), 4),
                "p_value": round(float(pvalue), 4),
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows)



def granger_causality(frame: pd.DataFrame, max_lag: int = 8) -> dict[str, Any]:
    """Granger F-test, H0: sentiment does not Granger-cause returns.

    Restricted:  y_t = c + Σ a_i y_{t-i} + e_t
    Unrestricted: y_t = c + Σ a_i y_{t-i} + Σ b_i x_{t-i} + e_t
    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))
    """

    data = frame[["log_return", "sentiment"]].dropna().to_numpy(dtype=float)
    n = len(data)
    effective_max = min(max_lag, max(1, (n - 6) // 2))
    if len(data) < max(20, max_lag * 3):
        return {
            "sufficient_data": False,
            "n": len(data),
            "max_lag": max_lag,
            "p_values": {},
            "best_lag": None,
            "min_p_value": None,
            "significant": False,
            "method": "numpy-ols-f-test",
        }

    y, x = data[:, 0], data[:, 1]
    p_values: dict[int, float] = {}
    for lag in range(1, effective_max + 1):
        rows, targets = [], []
        for t in range(lag, n):
            row = [1.0]
            row.extend(y[t - j] for j in range(1, lag + 1))
            row.extend(x[t - j] for j in range(1, lag + 1))
            rows.append(row)
            targets.append(y[t])
        design = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        rss_unrestricted = _ols_rss(target, design)
        rss_restricted = _ols_rss(target, design[:, : lag + 1])
        df1 = lag
        df2 = len(target) - design.shape[1]
        if df2 <= 0 or rss_unrestricted <= 0:
            p_values[lag] = 1.0
            continue
        f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
        p_values[lag] = round(float(_f_sf(max(f_stat, 0.0), df1, df2)), 6)

    best_lag = min(p_values, key=p_values.get)
    return {
        "sufficient_data": True,
        "n": len(data),
        "max_lag": effective_max,
        "p_values": p_values,
        "best_lag": int(best_lag),
        "min_p_value": p_values[best_lag],
        "significant": p_values[best_lag] < 0.05,
        "adf_pvalue_sentiment": round(_adf_pvalue(data[:, 1]), 4),
        "adf_pvalue_returns": round(_adf_pvalue(data[:, 0]), 4),
        "method": "numpy-ols-f-test",
        "interpretation": (
            f"在 lag={best_lag} 时 p={p_values[best_lag]:.4f}，"
            + (
                "拒绝原假设：情感显著格兰杰领先于价格收益。"
                if p_values[best_lag] < 0.05
                else "不能拒绝原假设：未发现显著格兰杰因果关系。"
            )
        ),
    }



def plot_report(
    frame: pd.DataFrame,
    lags: pd.DataFrame,
    granger: dict[str, Any],
    output: Path,
    *,
    entity: str,
    symbol: str,
    simulated: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Sentiment vs Price — {entity or 'ALL'} / {symbol}"
        + (" (simulated klines)" if simulated else " (Binance klines)"),
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(frame["timestamp"], frame["sentiment"], color="#d62728", label="Sentiment")
    ax.set_ylabel("Sentiment [-1, 1]", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax.twinx()
    ax2.plot(frame["timestamp"], frame["close"], color="#1f77b4", label="Close")
    ax2.set_ylabel("Price (USDT)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("Sentiment vs Close Price")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.bar(frame["timestamp"], frame["negative_ratio"], width=0.03, color="#ff7f0e", alpha=0.75)
    ax.plot(frame["timestamp"], frame["sentiment"], color="#d62728", marker="o", markersize=3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Negative Ratio vs Sentiment")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    colors = ["#2ca02c" if p < 0.05 else "#7f7f7f" for p in lags["p_value"]]
    ax.bar(lags["lag"], lags["correlation"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (hours): sentiment(t) → return(t+lag)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lagged Cross-Correlation (green = p < 0.05)")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    future = frame["log_return"].shift(-1)
    ax.scatter(frame["sentiment"], future, color="#1f77b4", alpha=0.65, s=18)
    valid = pd.concat([frame["sentiment"], future], axis=1).dropna()
    if len(valid) > 2:
        slope, intercept = np.polyfit(valid.iloc[:, 0], valid.iloc[:, 1], 1)
        xs = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#d62728", linewidth=1.5)
    title = "Sentiment(t) vs Return(t+1)"
    if granger.get("sufficient_data"):
        title += f" | Granger p={granger['min_p_value']:.4f}"
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Next-period log return")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_analysis(
    records_path: str | Path,
    *,
    entity: str | None = None,
    symbol: str = "BTCUSDT",
    freq: str = "1h",
    max_lag: int = 8,
    live: bool = False,
    output: Path | None = None,
    csv_output: Path | None = None,
) -> dict[str, Any]:
    """End-to-end analysis used by both the CLI and the notebook."""

    sentiment = load_sentiment_series(records_path, entity=entity, freq=freq)
    simulated = not live
    price: pd.DataFrame | None = None
    if live:
        try:
            price = fetch_klines(
                symbol,
                interval=freq,
                start=sentiment["timestamp"].min().to_pydatetime(),
                end=sentiment["timestamp"].max().to_pydatetime()
                + timedelta(hours=1),
            )
            if price.empty:
                raise ValueError("Binance returned no klines")
        except Exception as exc:  # noqa: BLE001 - offline/demo fallback
            logger.warning("Live klines unavailable (%s); simulating instead", exc)
            simulated = True

    if simulated:
        price = simulate_klines(
            pd.DatetimeIndex(sentiment["timestamp"]),
            sentiment=sentiment["sentiment"],
        )

    merged = align_series(sentiment, price)
    if len(merged) < 8:
        raise ValueError("Not enough aligned observations for correlation analysis")

    lags = lagged_correlations(merged, max_lag=max_lag)
    granger = granger_causality(merged, max_lag=min(max_lag, 8))
    output = output or default_settings.reports_dir / "sentiment_price_correlation.png"
    csv_output = csv_output or default_settings.data_dir / "sentiment_price.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_output, index=False)
    plot_report(
        merged,
        lags,
        granger,
        output,
        entity=entity or "ALL",
        symbol=symbol,
        simulated=simulated,
    )

    significant = lags[lags["p_value"] < 0.05]
    best = (
        significant.loc[significant["correlation"].abs().idxmax()]
        if not significant.empty
        else lags.loc[lags["correlation"].abs().idxmax()]
    )
    return {
        "entity": entity or "ALL",
        "symbol": symbol,
        "frequency": freq,
        "simulated_klines": simulated,
        "observations": int(len(merged)),
        "sentiment_mean": round(float(merged["sentiment"].mean()), 4),
        "sentiment_std": round(float(merged["sentiment"].std()), 4),
        "best_lag_hours": int(best["lag"]),
        "best_lag_correlation": round(float(best["correlation"]), 4),
        "best_lag_p_value": round(float(best["p_value"]), 4),
        "granger": granger,
        "figure": str(output),
        "csv": str(csv_output),
        "lag_table": lags.to_dict(orient="records"),
    }


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Sentiment / price lead-lag analysis")
    parser.add_argument("--records", default="data/processed.jsonl")
    parser.add_argument("--entity", default="币安")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--live", action="store_true", help="fetch Binance klines")
    parser.add_argument("--output", default="reports/sentiment_price_correlation.png")
    parser.add_argument("--csv", default="data/sentiment_price.csv")
    args = parser.parse_args()

    result = run_analysis(
        args.records,
        entity=args.entity,
        symbol=args.symbol,
        freq=args.freq,
        max_lag=args.max_lag,
        live=args.live,
        output=Path(args.output),
        csv_output=Path(args.csv),
    )
    printable = {k: v for k, v in result.items() if k != "lag_table"}
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()


