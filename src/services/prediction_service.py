from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Literal

import numpy as np
import pandas as pd

from src.models.lstm_inference import predict_lstm_signal
from src.models.timesfm_inference import predict_timesfm_signal
from src.models.weight_adjustor import WeightAdjustor


PredictionPeriod = Literal["1d", "1w", "1m", "3m", "6m", "1y"]

HORIZON_DAYS: dict[PredictionPeriod, int] = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

RETURN_THRESHOLDS: dict[PredictionPeriod, float] = {
    "1d": 0.005,
    "1w": 0.01,
    "1m": 0.01,
    "3m": 0.02,
    "6m": 0.03,
    "1y": 0.05,
}

SHORT_SENTIMENT_PERIODS = {"1d", "1w"}
MIN_PRICE_ROWS = 40


def build_prediction_response(
    *,
    ticker: str,
    period: PredictionPeriod,
    historical_prices: list[dict],
    sp500_prices: list[dict] | None = None,
    latest_realtime: dict | None = None,
    news_items: list[dict] | None = None,
    explain: bool = True,
    now: datetime | None = None,
    adjustor: WeightAdjustor | None = None,
) -> dict[str, object]:
    ticker = ticker.upper()
    now = now or datetime.now(timezone.utc)
    adjustor = adjustor or WeightAdjustor()

    price_df = prepare_price_frame(historical_prices, latest_realtime)
    if len(price_df) < MIN_PRICE_ROWS:
        raise ValueError(
            f"Not enough price history for {ticker}. Need at least {MIN_PRICE_ROWS} rows, "
            f"got {len(price_df)}."
        )

    market_signal = build_market_pattern_signal(
        ticker=ticker,
        period=period,
        price_df=price_df,
        stock_rows=historical_prices,
        sp500_rows=sp500_prices or [],
    )
    forecast_signal = build_forecast_signal(price_df, period)
    sentiment_signal = build_recent_sentiment_signal(
        ticker=ticker,
        period=period,
        news_items=news_items or [],
        now=now,
    )

    kwargs = {
        "period": period,
        "lstm_signal": market_signal["direction"],
        "timesfm_signal": forecast_signal["direction"],
        "sentiment_score": sentiment_signal["score"],
        "lstm_confidence": market_signal["confidence"],
        "timesfm_confidence": forecast_signal["confidence"],
        "sentiment_confidence": sentiment_signal["confidence"],
        "sentiment_relevance": sentiment_signal["relevance"],
        "news_count": sentiment_signal["news_count"],
    }

    result = adjustor.explain_prediction(**kwargs)
    if not explain:
        result.pop("explanation", None)

    return {
        "status": "success",
        "ticker": ticker,
        "period": period,
        "data_points": len(price_df),
        "latest_price": float(price_df["close"].iloc[-1]),
        "latest_price_time": _isoformat(price_df["date"].iloc[-1]),
        "signal_sources": {
            "recent_market_pattern": market_signal["source"],
            "time_series_forecast": forecast_signal["source"],
            "recent_news_sentiment": sentiment_signal["source"],
        },
        **result,
    }


def prepare_price_frame(
    historical_prices: list[dict],
    latest_realtime: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for doc in historical_prices:
        price = doc.get("close") if doc.get("close") is not None else doc.get("price")
        date = doc.get("date")
        sp500_value = doc.get("sp500_value")
        if price is None or date is None:
            continue
        if _is_nan(price):
            continue

        rows.append(
            {
                "date": pd.to_datetime(date, utc=True),
                "close": float(price),
                "sp500_value": _float_or_none(sp500_value),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "close", "sp500_value"])

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if latest_realtime and latest_realtime.get("price") is not None:
        realtime_price = latest_realtime["price"]
        realtime_date = latest_realtime.get("date")
        if realtime_date is not None and not _is_nan(realtime_price):
            last_sp500 = df["sp500_value"].dropna().iloc[-1] if df["sp500_value"].notna().any() else None
            realtime_row = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(realtime_date, utc=True),
                        "close": float(realtime_price),
                        "sp500_value": last_sp500,
                    }
                ]
            )
            df = pd.concat([df, realtime_row], ignore_index=True)
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    df["sp500_value"] = df["sp500_value"].ffill()
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def build_market_pattern_signal(
    *,
    ticker: str,
    period: PredictionPeriod,
    price_df: pd.DataFrame,
    stock_rows: list[dict],
    sp500_rows: list[dict],
) -> dict[str, object]:
    try:
        return predict_lstm_signal(
            ticker=ticker,
            period=period,
            stock_rows=stock_rows,
            sp500_rows=sp500_rows,
        )
    except Exception as exc:
        signal = build_recent_market_pattern_signal(price_df, period)
        signal["source"] = "price_history_fallback"
        signal["fallback_reason"] = str(exc)
        return signal


def build_recent_market_pattern_signal(
    price_df: pd.DataFrame,
    period: PredictionPeriod,
) -> dict[str, object]:
    horizon_days = HORIZON_DAYS[period]
    threshold = RETURN_THRESHOLDS[period]
    relative_returns = _relative_returns(price_df)
    window = min(max(horizon_days, 5), len(relative_returns))
    relative_return = float(relative_returns.tail(window).sum())

    return {
        "direction": classify_relative_return(relative_return, threshold),
        "confidence": confidence_from_return(relative_return, threshold),
        "relative_return": round(relative_return, 6),
        "source": "price_history_fallback",
    }


def build_forecast_signal(
    price_df: pd.DataFrame,
    period: PredictionPeriod,
) -> dict[str, object]:
    try:
        return predict_timesfm_signal(period=period, price_df=price_df)
    except Exception as exc:
        signal = build_time_series_forecast_signal(price_df, period)
        signal["fallback_reason"] = str(exc)
        return signal


def build_time_series_forecast_signal(
    price_df: pd.DataFrame,
    period: PredictionPeriod,
) -> dict[str, object]:
    horizon_days = HORIZON_DAYS[period]
    threshold = RETURN_THRESHOLDS[period]
    relative_returns = _relative_returns(price_df)
    lookback_window = min(max(horizon_days * 2, 20), 126, len(relative_returns))
    expected_relative_return = float(relative_returns.tail(lookback_window).mean() * horizon_days)

    return {
        "direction": classify_relative_return(expected_relative_return, threshold),
        "confidence": confidence_from_return(expected_relative_return, threshold),
        "relative_return": round(expected_relative_return, 6),
        "source": "price_history_forecast_proxy",
    }


def build_recent_sentiment_signal(
    *,
    ticker: str,
    period: PredictionPeriod,
    news_items: list[dict],
    now: datetime,
) -> dict[str, object]:
    if period not in SHORT_SENTIMENT_PERIODS:
        return {
            "score": 0.0,
            "direction": "neutral",
            "confidence": None,
            "relevance": None,
            "news_count": 0,
            "used_for_period": False,
            "source": "not_used_for_period",
        }

    lookback = timedelta(days=1 if period == "1d" else 7)
    matching_items = []
    for item in news_items:
        if str(item.get("ticker", "")).upper() != ticker:
            continue

        timestamp = _parse_datetime(item.get("timestamp"))
        if timestamp is None or now - timestamp > lookback:
            continue

        matching_items.append(item)

    if not matching_items:
        return {
            "score": 0.0,
            "direction": "neutral",
            "confidence": None,
            "relevance": None,
            "news_count": 0,
            "used_for_period": True,
            "source": "no_recent_news",
        }

    weighted_scores = []
    total_weight = 0.0
    confidences = []
    relevances = []
    for item in matching_items:
        score = float(item.get("sentiment_score", 0.0) or 0.0)
        confidence = float(item.get("sentiment_confidence", 0.0) or 0.0)
        relevance = float(item.get("relevance_score", 1.0) or 1.0)
        weight = max(0.0, confidence) * max(0.0, relevance)

        weighted_scores.append(score * weight)
        total_weight += weight
        confidences.append(confidence)
        relevances.append(relevance)

    aggregate_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    aggregate_score = max(-1.0, min(1.0, aggregate_score))

    return {
        "score": round(aggregate_score, 4),
        "direction": sentiment_direction(aggregate_score),
        "confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "relevance": round(float(np.mean(relevances)), 4) if relevances else None,
        "news_count": len(matching_items),
        "used_for_period": True,
        "source": "recent_news_sentiment",
    }


def classify_relative_return(relative_return: float, threshold: float) -> str:
    if relative_return > threshold:
        return "better"
    if relative_return < -threshold:
        return "worse"
    return "neutral"


def confidence_from_return(relative_return: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.5

    strength = abs(relative_return) / (threshold * 3)
    return round(max(0.35, min(0.95, 0.35 + strength * 0.60)), 4)


def sentiment_direction(sentiment_score: float) -> str:
    if sentiment_score > 0:
        return "positive"
    if sentiment_score < 0:
        return "negative"
    return "neutral"


def _relative_returns(price_df: pd.DataFrame) -> pd.Series:
    df = price_df.copy()
    df["stock_log_return"] = np.log(df["close"] / df["close"].shift(1))

    if df["sp500_value"].notna().sum() >= 2:
        df["sp500_log_return"] = np.log(df["sp500_value"] / df["sp500_value"].shift(1))
    else:
        df["sp500_log_return"] = 0.0

    returns = df["stock_log_return"] - df["sp500_log_return"].fillna(0.0)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _isoformat(value: object) -> str:
    parsed = pd.to_datetime(value, utc=True)
    return parsed.isoformat()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if _is_nan(value):
        return None
    return value


def _is_nan(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
