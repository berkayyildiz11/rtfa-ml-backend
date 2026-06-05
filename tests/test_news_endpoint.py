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
        }
        for idx in range(count)
    ]


class NewsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main.NEWS_CACHE["data"] = []
        main.NEWS_CACHE["last_updated"] = None

    async def test_news_endpoint_fetches_large_cache_once(self):
        items = make_news_items(25)

        with patch.object(main, "run_news_pipeline", AsyncMock(return_value=items)) as fetcher:
            response = await main.get_latest_news(page=1, limit=10)

        fetcher.assert_awaited_once_with(max_items=main.NEWS_CACHE_TARGET_ITEMS)
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


if __name__ == "__main__":
    unittest.main()
