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
        main.NEWS_REFRESH_TASK = None

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


if __name__ == "__main__":
    unittest.main()
