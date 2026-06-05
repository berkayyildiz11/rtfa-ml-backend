from __future__ import annotations

from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import pandas as pd
import torch

from src.models.lstm import (
    FEATURE_COLUMNS,
    INFERENCE_WINDOW_BY_PERIOD,
    LSTMRelativeReturnClassifier,
)


PredictionPeriod = Literal["1d", "1w", "1m", "3m", "6m", "1y"]

MODEL_DIR = Path("saved_models/lstm")
CLASS_LABELS = ["worse", "neutral", "better"]


def predict_lstm_signal(
    *,
    ticker: str,
    period: PredictionPeriod,
    stock_rows: list[dict],
    sp500_rows: list[dict],
    model_dir: str | Path = MODEL_DIR,
) -> dict[str, object]:
    model_dir = Path(model_dir)
    model_path = model_dir / f"best_lstm_{period}.pt"
    scaler_path = model_dir / f"scaler_{period}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing LSTM model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing LSTM scaler artifact: {scaler_path}")

    model_input = build_lstm_feature_window(
        ticker=ticker,
        period=period,
        stock_rows=stock_rows,
        sp500_rows=sp500_rows,
        scaler_path=scaler_path,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = LSTMRelativeReturnClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=64,
        num_layers=1,
        dropout=0.4,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        X = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    return {
        "direction": CLASS_LABELS[class_index],
        "confidence": round(confidence, 4),
        "probabilities": {
            label: round(float(probabilities[idx]), 4)
            for idx, label in enumerate(CLASS_LABELS)
        },
        "source": "trained_market_pattern_model",
    }


def build_lstm_feature_window(
    *,
    ticker: str,
    period: PredictionPeriod,
    stock_rows: list[dict],
    sp500_rows: list[dict],
    scaler_path: str | Path,
) -> np.ndarray:
    window_size = INFERENCE_WINDOW_BY_PERIOD[period]
    df = prepare_lstm_inference_frame(stock_rows, sp500_rows)
    ticker_df = df[df["ticker"].str.upper() == ticker.upper()].copy()
    ticker_df = ticker_df.sort_values("date")

    if len(ticker_df) < window_size:
        raise ValueError(
            f"Not enough complete LSTM rows for {ticker}. Need {window_size}, got {len(ticker_df)}."
        )

    with Path(scaler_path).open("rb") as f:
        scaler = pickle.load(f)

    features = ticker_df[FEATURE_COLUMNS].tail(window_size).astype(float)
    scaled_features = scaler.transform(features)
    return scaled_features.astype(np.float32)


def prepare_lstm_inference_frame(
    stock_rows: list[dict],
    sp500_rows: list[dict],
) -> pd.DataFrame:
    stock_df = pd.DataFrame(stock_rows)
    sp500_df = pd.DataFrame(sp500_rows)

    if stock_df.empty:
        raise ValueError("No stock rows supplied for LSTM inference.")
    if sp500_df.empty:
        raise ValueError("No S&P500 rows supplied for LSTM inference.")

    stock_df["date"] = pd.to_datetime(stock_df["date"], utc=True)
    sp500_df["date"] = pd.to_datetime(sp500_df["date"], utc=True)

    stock_df = stock_df.sort_values(["ticker", "date"]).copy()
    sp500_df = sp500_df.sort_values("date").copy()

    if "log_return" not in stock_df.columns:
        stock_df["log_return"] = (
            stock_df.groupby("ticker")["close"]
            .transform(lambda x: np.log(x / x.shift(1)))
        )

    sp500_df["sp500_log_return"] = np.log(sp500_df["close"] / sp500_df["close"].shift(1))
    sp500_features = sp500_df[
        ["date", "close", "vix", "rsi_14", "beta", "correlation_sp_vix", "sp500_log_return"]
    ].rename(columns={"close": "sp500_close", "rsi_14": "sp500_rsi_14"})

    df = stock_df.merge(sp500_features, on="date", how="inner")
    if "sp500_value" not in df.columns:
        df["sp500_value"] = df["sp500_close"]

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing LSTM feature column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[FEATURE_COLUMNS] = df.groupby("ticker", group_keys=False)[FEATURE_COLUMNS].apply(
        lambda group: group.ffill().bfill()
    )
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
