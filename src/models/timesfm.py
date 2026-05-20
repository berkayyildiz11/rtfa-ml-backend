import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import timesfm

from sklearn.metrics import mean_absolute_error, mean_squared_error


HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

THRESHOLDS = {
    "1d": 0.005,
    "1w": 0.01,
    "1m": 0.01,
    "3m": 0.02,
    "6m": 0.03,
}

CONTEXT_LEN = 512


def load_stock_from_mongodb(ticker: str) -> pd.DataFrame:
    load_dotenv()

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = "stock_tracking_db"
    collection_name = "historical_prices"

    if not mongo_uri:
        raise ValueError("MONGODB_URI is missing in .env")

    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]

    cursor = collection.find(
        {"ticker": ticker},
        {
            "_id": 0,
            "ticker": 1,
            "date": 1,
            "close": 1,
            "sp500_value": 1,
        },
    ).sort("date", 1)

    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["sp500_value"] = pd.to_numeric(df["sp500_value"], errors="coerce")

    df = df.dropna(subset=["close", "sp500_value"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def classify_relative_return(relative_return: float, threshold: float) -> str:
    if relative_return > threshold:
        return "better"
    elif relative_return < -threshold:
        return "worse"
    return "neutral"


def evaluate_forecast(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def build_timesfm_model(max_horizon: int):
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=max_horizon,
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


def run_timesfm_relative(symbol: str = "AAPL"):
    df = load_stock_from_mongodb(symbol)

    stock_prices = df["close"].values.astype(float)
    sp500_values = df["sp500_value"].values.astype(float)

    max_horizon = max(HORIZONS.values())

    if len(df) < CONTEXT_LEN + max_horizon:
        raise ValueError(
            f"Not enough data. Need at least {CONTEXT_LEN + max_horizon} rows, "
            f"but got {len(df)}."
        )

    stock_context = stock_prices[-CONTEXT_LEN - max_horizon:-max_horizon]
    stock_actual_future = stock_prices[-max_horizon:]

    sp500_context = sp500_values[-CONTEXT_LEN - max_horizon:-max_horizon]
    sp500_actual_future = sp500_values[-max_horizon:]

    stock_current = stock_context[-1]
    sp500_current = sp500_context[-1]

    model = build_timesfm_model(max_horizon)

    forecast, _ = model.forecast([stock_context, sp500_context])

    stock_forecast = forecast[0]
    sp500_forecast = forecast[1]

    results = {}

    print(f"\nTimesFM relative performance results for {symbol} vs S&P500")

    for horizon_name, horizon_days in HORIZONS.items():
        threshold = THRESHOLDS[horizon_name]

        stock_pred_final = stock_forecast[horizon_days - 1]
        sp500_pred_final = sp500_forecast[horizon_days - 1]

        stock_actual_final = stock_actual_future[horizon_days - 1]
        sp500_actual_final = sp500_actual_future[horizon_days - 1]

        stock_pred_return = (stock_pred_final - stock_current) / stock_current
        sp500_pred_return = (sp500_pred_final - sp500_current) / sp500_current
        relative_pred_return = stock_pred_return - sp500_pred_return

        stock_actual_return = (stock_actual_final - stock_current) / stock_current
        sp500_actual_return = (sp500_actual_final - sp500_current) / sp500_current
        actual_relative_return = stock_actual_return - sp500_actual_return

        predicted_signal = classify_relative_return(relative_pred_return, threshold)
        actual_signal = classify_relative_return(actual_relative_return, threshold)

        stock_metrics = evaluate_forecast(
            stock_actual_future[:horizon_days],
            stock_forecast[:horizon_days],
        )

        sp500_metrics = evaluate_forecast(
            sp500_actual_future[:horizon_days],
            sp500_forecast[:horizon_days],
        )

        results[horizon_name] = {
            "horizon_days": horizon_days,
            "threshold": threshold,
            "stock_pred_return": float(stock_pred_return),
            "sp500_pred_return": float(sp500_pred_return),
            "relative_pred_return": float(relative_pred_return),
            "predicted_signal": predicted_signal,
            "stock_actual_return": float(stock_actual_return),
            "sp500_actual_return": float(sp500_actual_return),
            "actual_relative_return": float(actual_relative_return),
            "actual_signal": actual_signal,
            "stock_mae": stock_metrics["mae"],
            "stock_rmse": stock_metrics["rmse"],
            "sp500_mae": sp500_metrics["mae"],
            "sp500_rmse": sp500_metrics["rmse"],
        }

        print(f"\n{horizon_name}")
        print(f"Stock predicted return: {stock_pred_return:.4%}")
        print(f"S&P500 predicted return: {sp500_pred_return:.4%}")
        print(f"Relative predicted return: {relative_pred_return:.4%}")
        print(f"Predicted signal: {predicted_signal}")
        print(f"Actual relative return: {actual_relative_return:.4%}")
        print(f"Actual signal: {actual_signal}")

    save_dir = Path("results/timesfm")
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"timesfm_relative_{symbol}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_timesfm_relative("AAPL")