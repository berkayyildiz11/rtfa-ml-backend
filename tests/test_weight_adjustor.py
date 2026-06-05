import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.models.weight_adjustor import SUPPORTED_PERIODS, WeightAdjustor
from src.scripts.train_weight_adjustor import build_weights_payload


class WeightAdjustorTests(unittest.TestCase):
    def test_sentiment_is_used_for_one_day_predictions(self):
        adjustor = WeightAdjustor(weights_path=None)

        weights = adjustor.adjust_weights(
            "1d",
            sentiment_score=0.8,
            sentiment_confidence=0.9,
            sentiment_relevance=1.0,
            news_count=4,
        )

        self.assertGreater(weights.sentiment, 0)
        self.assertAlmostEqual(
            weights.lstm + weights.timesfm + weights.sentiment,
            1.0,
            places=3,
        )

    def test_sentiment_is_used_for_one_week_predictions(self):
        adjustor = WeightAdjustor(weights_path=None)

        weights = adjustor.adjust_weights(
            "1w",
            sentiment_score=-0.7,
            sentiment_confidence=0.8,
            sentiment_relevance=0.9,
            news_count=3,
        )

        self.assertGreater(weights.sentiment, 0)

    def test_sentiment_is_ignored_for_longer_predictions(self):
        adjustor = WeightAdjustor(weights_path=None)

        for period in ("1m", "3m", "6m", "1y"):
            with self.subTest(period=period):
                weights = adjustor.adjust_weights(
                    period,
                    sentiment_score=1.0,
                    sentiment_confidence=1.0,
                    sentiment_relevance=1.0,
                    news_count=10,
                )

                self.assertEqual(weights.sentiment, 0.0)
                self.assertAlmostEqual(weights.lstm + weights.timesfm, 1.0)

    def test_lower_model_confidence_reduces_that_model_weight(self):
        adjustor = WeightAdjustor(weights_path=None)

        weights = adjustor.adjust_weights(
            "1m",
            lstm_confidence=0.2,
            timesfm_confidence=0.9,
        )

        self.assertLess(weights.lstm, weights.timesfm)

    def test_combine_predictions_returns_weighted_signal(self):
        adjustor = WeightAdjustor(neutral_threshold=0.2, weights_path=None)

        prediction = adjustor.combine_predictions(
            "1d",
            lstm_signal="better",
            timesfm_signal="neutral",
            sentiment_score=0.9,
            sentiment_confidence=0.9,
            sentiment_relevance=1.0,
            news_count=5,
        )

        self.assertEqual(prediction.signal, "better")
        self.assertGreater(prediction.score, 0.2)

    def test_invalid_period_raises_clear_error(self):
        adjustor = WeightAdjustor(weights_path=None)

        with self.assertRaisesRegex(ValueError, "Unsupported prediction period"):
            adjustor.adjust_weights("2w")

    def test_one_year_period_is_supported_by_fallback_weights(self):
        adjustor = WeightAdjustor(weights_path=None)

        weights = adjustor.adjust_weights("1y")

        self.assertEqual(weights.sentiment, 0.0)
        self.assertAlmostEqual(weights.lstm, 0.65, places=2)
        self.assertAlmostEqual(weights.timesfm, 0.35, places=2)

    def test_calibrated_weights_are_loaded_from_json(self):
        payload = {
            "weights": {
                "1y": {
                    "lstm": 0.8,
                    "timesfm": 0.2,
                    "sentiment": 0.0,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "weights.json"
            weights_path.write_text(json.dumps(payload))

            adjustor = WeightAdjustor(weights_path=weights_path)
            weights = adjustor.adjust_weights("1y")

        self.assertAlmostEqual(weights.lstm, 0.8)
        self.assertAlmostEqual(weights.timesfm, 0.2)
        self.assertEqual(weights.sentiment, 0.0)

    def test_uncalibrated_sentiment_keeps_runtime_overlay_budget(self):
        payload = {
            "sentiment_calibrated": False,
            "weights": {
                "1w": {
                    "lstm": 0.35,
                    "timesfm": 0.65,
                    "sentiment": 0.0,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "weights.json"
            weights_path.write_text(json.dumps(payload))

            adjustor = WeightAdjustor(weights_path=weights_path)
            weights = adjustor.adjust_weights(
                "1w",
                sentiment_score=0.8,
                sentiment_confidence=0.9,
                sentiment_relevance=1.0,
                news_count=5,
            )

        self.assertGreater(weights.sentiment, 0.0)
        self.assertLess(weights.lstm, 0.35)
        self.assertLess(weights.timesfm, 0.65)

    def test_calibrated_sentiment_value_is_honored_when_available(self):
        payload = {
            "sentiment_calibrated": True,
            "weights": {
                "1d": {
                    "lstm": 0.5,
                    "timesfm": 0.5,
                    "sentiment": 0.0,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "weights.json"
            weights_path.write_text(json.dumps(payload))

            adjustor = WeightAdjustor(weights_path=weights_path)
            weights = adjustor.adjust_weights(
                "1d",
                sentiment_score=1.0,
                sentiment_confidence=1.0,
                sentiment_relevance=1.0,
                news_count=5,
            )

        self.assertEqual(weights.sentiment, 0.0)

    def test_missing_weights_file_uses_fallback_weights(self):
        adjustor = WeightAdjustor(weights_path="missing/weights.json")

        weights = adjustor.adjust_weights("1y")

        self.assertAlmostEqual(weights.lstm, 0.65, places=2)
        self.assertAlmostEqual(weights.timesfm, 0.35, places=2)

    def test_explanation_contributions_sum_to_score_difference(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1w",
            lstm_signal="neutral",
            timesfm_signal="better",
            sentiment_score=0.8,
            lstm_confidence=0.6,
            timesfm_confidence=0.9,
            sentiment_confidence=0.85,
            sentiment_relevance=0.95,
            news_count=5,
        )

        explanation = result["explanation"]
        contribution_sum = sum(explanation["contributions"].values())

        self.assertAlmostEqual(
            contribution_sum,
            explanation["final_score"] - explanation["baseline_score"],
            places=3,
        )

    def test_explanation_uses_public_signal_names(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1d",
            lstm_signal="better",
            timesfm_signal="neutral",
            sentiment_score=0.3,
            sentiment_confidence=0.7,
            sentiment_relevance=0.9,
            news_count=2,
        )

        serialized = json.dumps(result).lower()

        self.assertIn("recent_market_pattern", result["weights"])
        self.assertIn("time_series_forecast", result["weights"])
        self.assertIn("recent_news_sentiment", result["weights"])

        forbidden_terms = [
            "lstm",
            "timesfm",
            "logit",
            "tensor",
            "hidden",
            "scaler",
            "feature_columns",
            "saved_models",
            "/users/",
        ]
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, serialized)

    def test_explanation_includes_user_facing_confidence_objects(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1w",
            lstm_signal="neutral",
            timesfm_signal="better",
            sentiment_score=0.8,
            lstm_confidence=0.61,
            timesfm_confidence=0.78,
            sentiment_confidence=0.84,
            sentiment_relevance=0.92,
            news_count=5,
        )

        prediction_confidence = result["prediction"]["confidence"]
        forecast_confidence = result["signals"]["time_series_forecast"]["confidence"]
        sentiment_confidence = result["signals"]["recent_news_sentiment"]["confidence"]

        self.assertEqual(
            set(prediction_confidence),
            {"score", "percentage", "label", "display"},
        )
        self.assertIsInstance(prediction_confidence["percentage"], int)
        self.assertIn(prediction_confidence["label"], {"low", "moderate", "high"})
        self.assertEqual(forecast_confidence["percentage"], 78)
        self.assertEqual(forecast_confidence["label"], "high")
        self.assertEqual(forecast_confidence["display"], "78% (high)")
        self.assertEqual(sentiment_confidence["percentage"], 84)
        self.assertEqual(sentiment_confidence["label"], "high")

    def test_missing_signal_confidence_is_user_facing_unspecified(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1m",
            lstm_signal="neutral",
            timesfm_signal="better",
        )

        market_confidence = result["signals"]["recent_market_pattern"]["confidence"]

        self.assertIsNone(market_confidence["score"])
        self.assertIsNone(market_confidence["percentage"])
        self.assertEqual(market_confidence["label"], "unspecified")
        self.assertEqual(market_confidence["display"], "unspecified")

    def test_strong_forecast_confidence_has_positive_public_contribution(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1m",
            lstm_signal="neutral",
            timesfm_signal="better",
            timesfm_confidence=0.95,
        )

        contributions = result["explanation"]["contributions"]

        self.assertGreater(contributions["time_series_forecast"], 0.0)
        self.assertEqual(contributions["recent_news_sentiment"], 0.0)

    def test_short_horizon_sentiment_has_public_contribution(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1w",
            lstm_signal="neutral",
            timesfm_signal="neutral",
            sentiment_score=0.9,
            sentiment_confidence=0.9,
            sentiment_relevance=1.0,
            news_count=5,
        )

        self.assertGreater(
            result["explanation"]["contributions"]["recent_news_sentiment"],
            0.0,
        )

    def test_long_horizon_sentiment_is_explained_as_unused(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1y",
            lstm_signal="better",
            timesfm_signal="neutral",
            sentiment_score=1.0,
            sentiment_confidence=1.0,
            sentiment_relevance=1.0,
            news_count=5,
        )

        self.assertEqual(
            result["explanation"]["contributions"]["recent_news_sentiment"],
            0.0,
        )
        self.assertIn(
            "recent news sentiment was not used",
            " ".join(result["explanation"]["decision_details"]),
        )

    def test_disagreement_appears_in_decision_details(self):
        adjustor = WeightAdjustor(weights_path=None)

        result = adjustor.explain_prediction(
            "1m",
            lstm_signal="worse",
            timesfm_signal="better",
            lstm_confidence=0.8,
            timesfm_confidence=0.8,
        )

        details = " ".join(result["explanation"]["decision_details"])

        self.assertIn("did not fully agree", details)


class WeightAdjustorCalibrationTests(unittest.TestCase):
    def test_calibration_payload_contains_all_periods(self):
        dates = pd.bdate_range("2022-01-03", periods=340)
        stock_rows = []
        sp500_rows = []

        for idx, date in enumerate(dates):
            sp500_close = 4000 + idx * 2
            sp500_rows.append(
                {
                    "date": date,
                    "close": sp500_close,
                }
            )

            stock_close = 100 + idx * 0.8 + (idx % 7) * 0.1
            stock_rows.append(
                {
                    "ticker": "AAPL",
                    "date": date,
                    "open": stock_close,
                    "high": stock_close + 1,
                    "low": stock_close - 1,
                    "close": stock_close,
                    "volume": 1000,
                    "log_return": 0.001 + (idx % 5) * 0.0001,
                    "macd": 0.0,
                    "rsi_14": 50.0,
                    "sp500_value": sp500_close,
                }
            )

        payload = build_weights_payload(
            pd.DataFrame(stock_rows),
            pd.DataFrame(sp500_rows),
            weight_step=0.5,
        )

        self.assertEqual(set(payload["weights"]), set(SUPPORTED_PERIODS))
        self.assertEqual(set(payload["periods"]), set(SUPPORTED_PERIODS))
        self.assertFalse(payload["sentiment_calibrated"])
        self.assertGreaterEqual(payload["weights"]["1y"]["lstm"], 0.65)


if __name__ == "__main__":
    unittest.main()
