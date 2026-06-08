import asyncio
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

from src.services.prediction_service import (
    build_prediction_response_async,
    build_prediction_response,
    build_recent_sentiment_signal,
    prepare_price_frame,
)


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
            use_finbert_for_prediction=False,
        )

        serialized = json.dumps(response).lower()

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["ticker"], "AAPL")
        self.assertIn("explanation", response)
        self.assertIn("explanatory_text", response)
        self.assertIsInstance(response["explanatory_text"], str)
        self.assertGreater(len(response["explanatory_text"]), 0)
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
            use_finbert_for_prediction=False,
        )

        self.assertEqual(response["status"], "success")
        self.assertNotIn("explanation", response)
        self.assertNotIn("explanatory_text", response)
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
            use_finbert_for_prediction=False,
        )

        self.assertIn("explanation", response)
        self.assertIn("explanatory_text", response)

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
            use_finbert_for_prediction=False,
        )

        self.assertFalse(response["signals"]["recent_news_sentiment"]["used_for_period"])
        self.assertEqual(
            response["explanation"]["contributions"]["recent_news_sentiment"],
            0.0,
        )

    def test_short_horizon_prediction_sentiment_uses_finbert_when_enabled(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)

        class FakeAnalyzer:
            def analyze_article(self, headline, summary):
                return {
                    "sentiment_score": 0.7,
                    "sentiment_label": "positive",
                    "sentiment_confidence": 0.8,
                }

        with patch(
            "src.services.prediction_service.get_finbert_analyzer",
            return_value=FakeAnalyzer(),
        ):
            signal = build_recent_sentiment_signal(
                ticker="AAPL",
                period="1w",
                news_items=[
                    {
                        "ticker": "AAPL",
                        "timestamp": now.isoformat(),
                        "headline": "Apple beats earnings expectations",
                        "summary": "Analysts raised targets.",
                        "relevance_score": 0.9,
                    }
                ],
                now=now,
                use_finbert=True,
            )

        self.assertEqual(signal["source"], "finbert_prediction_sentiment")
        self.assertEqual(signal["direction"], "positive")
        self.assertAlmostEqual(signal["score"], 0.7)
        self.assertAlmostEqual(signal["confidence"], 0.8)

    def test_prediction_sentiment_can_use_cached_lightweight_scores(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        signal = build_recent_sentiment_signal(
            ticker="AAPL",
            period="1w",
            news_items=[
                {
                    "ticker": "AAPL",
                    "timestamp": now.isoformat(),
                    "sentiment_score": -0.6,
                    "sentiment_confidence": 0.75,
                    "relevance_score": 1.0,
                }
            ],
            now=now,
            use_finbert=False,
        )

        self.assertEqual(signal["source"], "cached_news_sentiment")
        self.assertEqual(signal["direction"], "negative")

    def test_async_prediction_response_uses_precomputed_sentiment(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)

        response = asyncio.run(
            build_prediction_response_async(
                ticker="AAPL",
                period="1w",
                historical_prices=make_history(now),
                precomputed_sentiment_signal={
                    "score": 0.6,
                    "direction": "positive",
                    "confidence": 0.8,
                    "relevance": 0.9,
                    "news_count": 3,
                    "used_for_period": True,
                    "source": "cached_prediction_sentiment",
                },
                explain=True,
                now=now,
                use_finbert_for_prediction=False,
            )
        )

        self.assertEqual(
            response["signal_sources"]["recent_news_sentiment"],
            "cached_prediction_sentiment",
        )
        self.assertGreater(
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
