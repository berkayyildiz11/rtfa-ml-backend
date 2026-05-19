import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import timesfm
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZONS = {
    "1_day": 1,
    "1_week": 5,
    "1_month": 21,
    "3_months": 63,
    "6_months": 126,
}

CONTEXT_LEN = 512

def load_stock_from_mongodb(ticker: str) -> pd.DataFrame:
    load_dotenv()

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = "stock_tracking_db"
    collection_name = "historical_prices"

    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]

    cursor = collection.find(
        {"ticker": ticker},
        {
            "_id": 0,
            "ticker": 1,
            "date": 1,
            "close": 1,
            "open": 1,
            "high": 1,
            "low": 1,
            "volume": 1,
            "log_return": 1,
            "macd": 1,
            "rsi_14": 1,
            "sp500_value": 1,
        },
    ).sort("date", 1)

    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df

def evaluate_forecast(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "mae": mae,
        "rmse": rmse,
    }

def run_timesfm(symbol: str = "AAPL"):
    df = load_stock_from_mongodb(symbol)

    close_prices = df["close"].values.astype(float)

    max_horizon = max(HORIZONS.values())

    if len(close_prices) < CONTEXT_LEN + max_horizon:
        raise ValueError(
            f"Not enough data. Need at least {CONTEXT_LEN + max_horizon} rows, "
            f"but got {len(close_prices)}."
        )
    
    context = close_prices[-CONTEXT_LEN - max_horizon:-max_horizon]
    actual_future = close_prices[-max_horizon:]

    model = timesfm.TimesFm(
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

    forecast, _ = model.forecast([context])
    forecast = forecast[0]

    print(f"\nTimesFM results for {symbol}")

    for name, horizon in HORIZONS.items():
        actual = actual_future[:horizon]
        predicted = forecast[:horizon]

        metrics = evaluate_forecast(actual, predicted)

        print(f"\n{name}")
        print(f"Actual final value: {actual[-1]:.4f}")
        print(f"Predicted final value: {predicted[-1]:.4f}")
        print(metrics)


if __name__ == "__main__":
    run_timesfm("AAPL")