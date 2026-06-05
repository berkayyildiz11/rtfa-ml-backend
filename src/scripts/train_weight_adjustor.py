from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.weight_adjustor import SIGNAL_SCORES, SUPPORTED_PERIODS


STOCK_DATA_PATH = Path("data/training/stock_data.csv")
SP500_DATA_PATH = Path("data/training/sp500_data.csv")
OUTPUT_PATH = Path("saved_models/weight_adjustor/weights.json")

NEUTRAL_THRESHOLD = 0.03
TARGET_SIGNAL_COLUMN = "actual_signal"
LSTM_SIGNAL_COLUMN = "lstm_signal"
TIMESFM_SIGNAL_COLUMN = "timesfm_signal"

HORIZON_CONFIGS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

TIMESFM_RELIABILITY_NOTES = {
    "1y": "TimesFM 1y historical calibration is treated as limited because the "
    "current project does not store full historical TimesFM prediction series.",
}


def classify_relative_return(relative_return: float, threshold: float = NEUTRAL_THRESHOLD) -> str:
    if relative_return > threshold:
        return "better"
    if relative_return < -threshold:
        return "worse"
    return "neutral"


def prepare_calibration_frame(
    stock_df: pd.DataFrame,
    sp500_df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    stock_df = stock_df.copy()
    sp500_df = sp500_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    sp500_df["date"] = pd.to_datetime(sp500_df["date"])

    stock_df = stock_df.sort_values(["ticker", "date"])
    sp500_df = sp500_df.sort_values("date")

    sp500_df["sp500_log_return"] = np.log(sp500_df["close"] / sp500_df["close"].shift(1))
    sp500_features = sp500_df[["date", "close", "sp500_log_return"]].rename(
        columns={"close": "sp500_close"}
    )

    df = stock_df.merge(sp500_features, on="date", how="inner")
    if "sp500_value" not in df.columns:
        df["sp500_value"] = df["sp500_close"]

    df["relative_return"] = df["log_return"] - df["sp500_log_return"]
    df["future_relative_return"] = (
        df.groupby("ticker")["relative_return"]
        .transform(
            lambda x: x.shift(-1)
            .rolling(window=horizon_days)
            .sum()
            .shift(-(horizon_days - 1))
        )
    )

    trailing_lstm_window = min(max(horizon_days, 5), 63)
    trailing_timesfm_window = min(max(horizon_days * 2, 10), 126)

    df["lstm_proxy_return"] = (
        df.groupby("ticker")["relative_return"]
        .transform(lambda x: x.rolling(trailing_lstm_window, min_periods=3).sum())
    )
    df["timesfm_proxy_return"] = (
        df.groupby("ticker")["relative_return"]
        .transform(lambda x: x.rolling(trailing_timesfm_window, min_periods=5).mean())
        * horizon_days
    )

    df = df.dropna(
        subset=[
            "future_relative_return",
            "lstm_proxy_return",
            "timesfm_proxy_return",
        ]
    )

    df[TARGET_SIGNAL_COLUMN] = df["future_relative_return"].map(classify_relative_return)
    df[LSTM_SIGNAL_COLUMN] = df["lstm_proxy_return"].map(classify_relative_return)
    df[TIMESFM_SIGNAL_COLUMN] = df["timesfm_proxy_return"].map(classify_relative_return)

    return df[
        [
            "ticker",
            "date",
            TARGET_SIGNAL_COLUMN,
            LSTM_SIGNAL_COLUMN,
            TIMESFM_SIGNAL_COLUMN,
        ]
    ].reset_index(drop=True)


def combine_signal_scores(
    lstm_signals: pd.Series,
    timesfm_signals: pd.Series,
    lstm_weight: float,
) -> list[str]:
    timesfm_weight = 1.0 - lstm_weight
    scores = (
        lstm_signals.map(SIGNAL_SCORES).astype(float) * lstm_weight
        + timesfm_signals.map(SIGNAL_SCORES).astype(float) * timesfm_weight
    )

    return [classify_relative_return(score, threshold=0.20) for score in scores]


def calibrate_period(
    calibration_df: pd.DataFrame,
    *,
    weight_step: float = 0.05,
) -> dict[str, object]:
    if calibration_df.empty:
        raise ValueError("calibration_df must contain at least one row")

    y_true = calibration_df[TARGET_SIGNAL_COLUMN]
    best_result = None
    candidate_count = 0

    for lstm_weight in np.round(np.arange(0.0, 1.0 + weight_step, weight_step), 4):
        candidate_count += 1
        y_pred = combine_signal_scores(
            calibration_df[LSTM_SIGNAL_COLUMN],
            calibration_df[TIMESFM_SIGNAL_COLUMN],
            float(lstm_weight),
        )
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=["worse", "neutral", "better"],
            average="macro",
            zero_division=0,
        )

        if best_result is None or macro_f1 > best_result["macro_f1"]:
            best_result = {
                "lstm": round(float(lstm_weight), 4),
                "timesfm": round(float(1.0 - lstm_weight), 4),
                "sentiment": 0.0,
                "macro_f1": round(float(macro_f1), 6),
            }

    best_result["candidate_count"] = candidate_count
    return best_result


def build_weights_payload(
    stock_df: pd.DataFrame,
    sp500_df: pd.DataFrame,
    *,
    weight_step: float = 0.05,
) -> dict[str, object]:
    weights = {}
    periods = {}

    for period in SUPPORTED_PERIODS:
        horizon_days = HORIZON_CONFIGS[period]
        calibration_df = prepare_calibration_frame(stock_df, sp500_df, horizon_days)
        result = calibrate_period(calibration_df, weight_step=weight_step)

        if period == "1y":
            result["lstm"] = max(float(result["lstm"]), 0.65)
            result["timesfm"] = round(1.0 - result["lstm"], 4)

        weights[period] = {
            "lstm": result["lstm"],
            "timesfm": result["timesfm"],
            "sentiment": result["sentiment"],
        }
        periods[period] = {
            "horizon_days": horizon_days,
            "macro_f1": result["macro_f1"],
            "candidate_count": result["candidate_count"],
            "sample_count": int(len(calibration_df)),
            "timesfm_note": TIMESFM_RELIABILITY_NOTES.get(period),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "macro_f1",
        "prediction_source": "price_history_proxy_signals",
        "sentiment_calibrated": False,
        "sentiment_note": "Historical sentiment is not available; sentiment remains a runtime overlay for 1d and 1w.",
        "weights": weights,
        "periods": periods,
    }


def train_weight_adjustor(
    stock_csv_path: str | Path = STOCK_DATA_PATH,
    sp500_csv_path: str | Path = SP500_DATA_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> dict[str, object]:
    stock_df = pd.read_csv(stock_csv_path)
    sp500_df = pd.read_csv(sp500_csv_path)

    payload = build_weights_payload(stock_df, sp500_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=4)

    return payload


if __name__ == "__main__":
    result = train_weight_adjustor()
    print(json.dumps(result, indent=4))
