from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd


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

CONTEXT_LEN = 512


def predict_timesfm_signal(
    *,
    period: PredictionPeriod,
    price_df: pd.DataFrame,
) -> dict[str, object]:
    horizon_days = HORIZON_DAYS[period]
    threshold = RETURN_THRESHOLDS[period]
    df = price_df.dropna(subset=["close", "sp500_value"]).copy()
    df = df.sort_values("date")

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"TimesFM needs at least {CONTEXT_LEN} complete rows, got {len(df)}.")

    stock_context = df["close"].tail(CONTEXT_LEN).to_numpy(dtype=float)
    sp500_context = df["sp500_value"].tail(CONTEXT_LEN).to_numpy(dtype=float)

    model = _timesfm_model(horizon_days)
    forecast, _ = model.forecast([stock_context, sp500_context])

    stock_forecast = forecast[0]
    sp500_forecast = forecast[1]

    stock_current = stock_context[-1]
    sp500_current = sp500_context[-1]
    stock_pred_final = stock_forecast[horizon_days - 1]
    sp500_pred_final = sp500_forecast[horizon_days - 1]

    stock_pred_return = (stock_pred_final - stock_current) / stock_current
    sp500_pred_return = (sp500_pred_final - sp500_current) / sp500_current
    relative_return = float(stock_pred_return - sp500_pred_return)

    return {
        "direction": classify_relative_return(relative_return, threshold),
        "confidence": confidence_from_return(relative_return, threshold),
        "relative_return": round(relative_return, 6),
        "source": "foundation_time_series_model",
    }


@lru_cache(maxsize=6)
def _timesfm_model(horizon_days: int):
    import timesfm

    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=horizon_days,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )


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
