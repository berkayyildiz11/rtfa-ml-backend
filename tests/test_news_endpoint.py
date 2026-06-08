import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import main


def make_news_items(count: int) -> list[dict]:
    return [
        {
            "ticker": "AAPL",
            "timestamp": datetime(2026, 6, 5, tzinfo=timezone.utc).isoformat(),
            "headline": f"Headline {idx}",
            "sentiment_score": 0.1,
            "sentiment_confidence": 0.5,
            "relevance_score": 0.8,
        }
        for idx in range(count)
    ]


class NewsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main.NEWS_CACHE["data"] = []
        main.NEWS_CACHE["last_updated"] = None
        main.NEWS_REFRESH_TASK = None
        main.PREDICTION_SENTIMENT_CACHE["data"] = {}
        main.PREDICTION_SENTIMENT_CACHE["last_updated"] = None

    async def test_news_endpoint_uses_disk_cache_without_waiting_for_refresh(self):
        items = make_news_items(25)

        with (
            patch.object(main, "load_news_from_disk", return_value=items),
            patch.object(main, "ensure_news_refresh_started") as refresh,
        ):
            response = await main.get_latest_news(page=1, limit=10)

        refresh.assert_called_once()
        self.assertEqual(response["total"], 25)
        self.assertEqual(response["totalPages"], 3)
        self.assertEqual(len(response["data"]), 10)

    async def test_news_endpoint_serves_later_pages_from_cache(self):
        main.NEWS_CACHE["data"] = make_news_items(25)
        main.NEWS_CACHE["last_updated"] = datetime.now(timezone.utc)

        with patch.object(main, "run_news_pipeline", AsyncMock()) as fetcher:
            response = await main.get_latest_news(page=2, limit=10)

        fetcher.assert_not_awaited()
        self.assertEqual(response["page"], 2)
        self.assertEqual(len(response["data"]), 10)
        self.assertEqual(response["data"][0]["headline"], "Headline 10")

    async def test_news_endpoint_returns_empty_quickly_while_refreshing(self):
        with (
            patch.object(main, "load_news_from_disk", return_value=[]),
            patch.object(main, "ensure_news_refresh_started") as refresh,
        ):
            response = await main.get_latest_news(page=1, limit=10)

        refresh.assert_called()
        self.assertEqual(response["total"], 0)
        self.assertEqual(response["data"], [])

    async def test_refresh_news_cache_uses_hourly_batch_settings(self):
        items = make_news_items(30)

        with (
            patch.object(main, "run_news_pipeline", AsyncMock(return_value=items)) as fetcher,
            patch.object(main, "refresh_prediction_sentiment_cache_from_news") as sentiment_refresh,
        ):
            await main.refresh_news_cache()

        fetcher.assert_awaited_once_with(
            max_items=main.NEWS_CACHE_TARGET_ITEMS,
            lookback_days=main.NEWS_LOOKBACK_DAYS,
        )
        sentiment_refresh.assert_called_once()
        self.assertEqual(main.NEWS_CACHE["data"], items)
        self.assertIsNotNone(main.NEWS_CACHE["last_updated"])

    async def test_news_endpoint_ranks_relevant_effective_news(self):
        timestamp = datetime(2026, 6, 5, tzinfo=timezone.utc).isoformat()
        main.NEWS_CACHE["data"] = [
            {
                "ticker": "MSFT",
                "timestamp": timestamp,
                "headline": "Low relevance",
                "sentiment_score": 1.0,
                "sentiment_confidence": 1.0,
                "relevance_score": 0.2,
            },
            {
                "ticker": "NVDA",
                "timestamp": timestamp,
                "headline": "High relevance strong sentiment",
                "sentiment_score": 0.8,
                "sentiment_confidence": 0.9,
                "relevance_score": 0.95,
            },
            {
                "ticker": "AAPL",
                "timestamp": timestamp,
                "headline": "Moderate relevance neutral sentiment",
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.5,
                "relevance_score": 0.8,
            },
        ]
        main.NEWS_CACHE["last_updated"] = datetime.now(timezone.utc)

        response = await main.get_latest_news(page=1, limit=10)

        headlines = [item["headline"] for item in response["data"]]
        self.assertEqual(headlines[0], "High relevance strong sentiment")
        self.assertNotIn("Low relevance", headlines)

    async def test_prediction_sentiment_cache_is_built_for_short_periods(self):
        now = datetime(2026, 6, 5, tzinfo=timezone.utc)
        items = make_news_items(2)

        with patch.object(
            main,
            "build_recent_sentiment_signal",
            return_value={
                "score": 0.2,
                "direction": "positive",
                "confidence": 0.7,
                "relevance": 0.9,
                "news_count": 1,
                "used_for_period": True,
                "source": "test_cache",
            },
        ) as builder:
            main.refresh_prediction_sentiment_cache_from_news(
                items,
                now=now,
                use_finbert=False,
            )

        self.assertEqual(builder.call_count, len(main.STOCKS) * 2)
        self.assertEqual(
            main.get_cached_prediction_sentiment("AAPL", "1d")["source"],
            "test_cache",
        )


if __name__ == "__main__":
    unittest.main()
