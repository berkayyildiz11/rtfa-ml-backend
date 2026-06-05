from pathlib import Path
import pickle
import tempfile
import unittest

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.lstm import FEATURE_COLUMNS
from src.models.lstm_inference import build_lstm_feature_window, prepare_lstm_inference_frame


def make_stock_rows(rows: int = 40) -> list[dict]:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    output = []

    for idx, date in enumerate(dates):
        close = 100 + idx
        output.append(
            {
                "ticker": "AAPL",
                "date": date,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + idx,
                "log_return": 0.001,
                "macd": 0.1,
                "rsi_14": 55.0,
                "sp500_value": 4000 + idx,
            }
        )

    return output


def make_sp500_rows(rows: int = 40) -> list[dict]:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    output = []

    for idx, date in enumerate(dates):
        output.append(
            {
                "date": date,
                "close": 4000 + idx,
                "vix": 15.0,
                "rsi_14": 52.0,
                "beta": 1.0,
                "correlation_sp_vix": -0.8,
            }
        )

    return output


class LSTMInferenceTests(unittest.TestCase):
    def test_prepare_lstm_inference_frame_builds_required_features(self):
        frame = prepare_lstm_inference_frame(make_stock_rows(), make_sp500_rows())

        for column in FEATURE_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, frame.columns)

        self.assertGreaterEqual(len(frame), 30)

    def test_build_lstm_feature_window_uses_saved_scaler(self):
        frame = prepare_lstm_inference_frame(make_stock_rows(), make_sp500_rows())
        scaler = StandardScaler()
        scaler.fit(frame[FEATURE_COLUMNS])

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler_1d.pkl"
            with scaler_path.open("wb") as f:
                pickle.dump(scaler, f)

            window = build_lstm_feature_window(
                ticker="AAPL",
                period="1d",
                stock_rows=make_stock_rows(),
                sp500_rows=make_sp500_rows(),
                scaler_path=scaler_path,
            )

        self.assertEqual(window.shape, (30, len(FEATURE_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
