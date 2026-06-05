from __future__ import annotations

from pathlib import Path
import pickle
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.lstm import FEATURE_COLUMNS, HORIZON_CONFIGS, prepare_dataframe


STOCK_DATA_PATH = Path("data/training/stock_data.csv")
SP500_DATA_PATH = Path("data/training/sp500_data.csv")
OUTPUT_DIR = Path("saved_models/lstm")


def fit_lstm_scalers(
    stock_csv_path: str | Path = STOCK_DATA_PATH,
    sp500_csv_path: str | Path = SP500_DATA_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, str]:
    stock_df = pd.read_csv(stock_csv_path)
    sp500_df = pd.read_csv(sp500_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for period, horizon_days in HORIZON_CONFIGS.items():
        df = prepare_dataframe(stock_df, sp500_df, horizon_days)
        dates = sorted(df["date"].unique())
        train_end = int(len(dates) * 0.7)
        train_dates = dates[:train_end]
        train_df = df[df["date"].isin(train_dates)].copy()

        scaler = StandardScaler()
        scaler.fit(train_df[FEATURE_COLUMNS])

        scaler_path = output_dir / f"scaler_{period}.pkl"
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)
        saved_paths[period] = str(scaler_path)

    return saved_paths


if __name__ == "__main__":
    paths = fit_lstm_scalers()
    for period, path in paths.items():
        print(f"{period}: {path}")
