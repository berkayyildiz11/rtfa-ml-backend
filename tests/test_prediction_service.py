from datetime import datetime, timedelta, timezone
import json
import unittest

from src.services.prediction_service import build_prediction_response, prepare_price_frame


def make_history(now: datetime, rows: int = 80) -> list[dict]:
    start = now - timedelta(days=rows)
    history = []

    for idx in range(rows):
        date = start + timedelta(days=idx)
        history.append(
            {
                "ticker": "AAPL",
                "date": date,
                "close": 100 + idx * 0.5,
                "sp500_value": 4000 + idx * 1.5,
            }
        )

    return history


class PredictionServiceTests(unittest.TestCase):
    def test_prediction_response_with_explanation_uses_public_shape(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        response = build_prediction_response(
            ticker="aapl",
            period="1w",
            historical_prices=make_history(now),
            latest_realtime={
                "symbol": "AAPL",
                "price": 142.0,
                "date": now,
            },
            news_items=[
                {
                    "ticker": "AAPL",
                    "timestamp": (now - timedelta(hours=3)).isoformat(),
                    "sentiment_score": 0.8,
                    "sentiment_confidence": 0.9,
                    "relevance_score": 1.0,
                }
            ],
            explain=True,
            now=now,
        )

        serialized = json.dumps(response).lower()

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["ticker"], "AAPL")
        self.assertIn("explanation", response)
        self.assertIn("confidence", response["prediction"])
        self.assertIn("recent_market_pattern", response["weights"])
        self.assertIn("time_series_forecast", response["weights"])
        self.assertIn("recent_news_sentiment", response["weights"])
        self.assertGreater(
            response["explanation"]["contributions"]["recent_news_sentiment"],
            0.0,
        )

        for forbidden_term in ("lstm", "timesfm", "logit", "tensor", "hidden"):
            with self.subTest(forbidden_term=forbidden_term):
                self.assertNotIn(forbidden_term, serialized)

    def test_prediction_response_without_explanation_omits_explanation_block(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        response = build_prediction_response(
            ticker="AAPL",
            period="1m",
            historical_prices=make_history(now),
            explain=False,
            now=now,
        )

        self.assertEqual(response["status"], "success")
        self.assertNotIn("explanation", response)
        self.assertIn("prediction", response)
        self.assertIn("weights", response)
        self.assertIn("signals", response)

    def test_prediction_response_includes_explanation_by_default(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        response = build_prediction_response(
            ticker="AAPL",
            period="1m",
            historical_prices=make_history(now),
            now=now,
        )

        self.assertIn("explanation", response)

    def test_long_horizon_sentiment_is_not_used(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        response = build_prediction_response(
            ticker="AAPL",
            period="1y",
            historical_prices=make_history(now, rows=300),
            news_items=[
                {
                    "ticker": "AAPL",
                    "timestamp": now.isoformat(),
                    "sentiment_score": 1.0,
                    "sentiment_confidence": 1.0,
                    "relevance_score": 1.0,
                }
            ],
            explain=True,
            now=now,
        )

        self.assertFalse(response["signals"]["recent_news_sentiment"]["used_for_period"])
        self.assertEqual(
            response["explanation"]["contributions"]["recent_news_sentiment"],
            0.0,
        )

    def test_prepare_price_frame_appends_latest_realtime_price(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        frame = prepare_price_frame(
            make_history(now),
            {
                "symbol": "AAPL",
                "price": 150.0,
                "date": now,
            },
        )

        self.assertEqual(float(frame["close"].iloc[-1]), 150.0)


if __name__ == "__main__":
    unittest.main()
