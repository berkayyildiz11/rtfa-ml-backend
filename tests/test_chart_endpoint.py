import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import main


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, *args):
        return self

    async def to_list(self, length):
        return self.documents


class FakeCollection:
    def __init__(self, find_results=None, latest=None):
        self.find_results = find_results or []
        self.latest = latest
        self.find_calls = []
        self.find_one_calls = []

    def find(self, query):
        self.find_calls.append(query)
        if len(self.find_results) >= len(self.find_calls):
            return FakeCursor(self.find_results[len(self.find_calls) - 1])
        return FakeCursor([])

    async def find_one(self, query, *args, **kwargs):
        self.find_one_calls.append((query, args, kwargs))
        return self.latest


class FakeDB:
    def __init__(self, historical_prices):
        self.historical_prices = historical_prices


class ChartEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_week_falls_back_to_realtime_series_when_daily_data_is_empty(self):
        historical = FakeCollection(find_results=[[]])
        trades = FakeCollection(
            find_results=[
                [],
                [
                    {
                        "symbol": "AAPL",
                        "date": datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc),
                        "price": 190.0,
                    },
                    {
                        "symbol": "AAPL",
                        "date": datetime(2026, 6, 10, 12, 1, tzinfo=timezone.utc),
                        "price": 191.0,
                    },
                ],
            ],
            latest={
                "symbol": "AAPL",
                "date": datetime(2026, 6, 10, 12, 1, tzinfo=timezone.utc),
                "price": 191.0,
            },
        )

        with (
            patch.object(main, "db", FakeDB(historical)),
            patch.object(main, "trades_col", trades),
        ):
            response = await main.get_stock_chart_data("aapl", period="1w")

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["ticker"], "AAPL")
        self.assertEqual([point["price"] for point in response["data"]], [190.0, 191.0])
        self.assertEqual(trades.find_calls[0]["ticker"], "AAPL")
        self.assertEqual(trades.find_calls[1]["symbol"], "AAPL")
        self.assertEqual(trades.find_one_calls, [])

    async def test_one_month_keeps_existing_daily_series_and_appends_latest_price(self):
        historical = FakeCollection(find_results=[[]])
        trades = FakeCollection(
            find_results=[
                [
                    {
                        "ticker": "AAPL",
                        "date": datetime(2026, 5, 28, tzinfo=timezone.utc),
                        "close": 188.0,
                    }
                ]
            ],
            latest={
                "symbol": "AAPL",
                "date": datetime(2026, 6, 10, 12, 1, tzinfo=timezone.utc),
                "price": 191.0,
            },
        )

        with (
            patch.object(main, "db", FakeDB(historical)),
            patch.object(main, "trades_col", trades),
        ):
            response = await main.get_stock_chart_data("aapl", period="1m")

        self.assertEqual(response["status"], "success")
        self.assertEqual([point["price"] for point in response["data"]], [188.0, 191.0])
        self.assertEqual(len(trades.find_calls), 1)
        self.assertEqual(trades.find_calls[0]["ticker"], "AAPL")
        self.assertEqual(len(trades.find_one_calls), 1)


if __name__ == "__main__":
    unittest.main()
