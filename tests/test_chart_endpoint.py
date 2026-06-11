import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import main


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents
        self.sort_args = None

    def sort(self, *args):
        self.sort_args = args
        return self

    async def to_list(self, length):
        return self.documents


class FakeCollection:
    def __init__(self, documents=None, latest=None):
        self.documents = documents or []
        self.latest = latest
        self.find_calls = []
        self.find_one_calls = []

    def find(self, query):
        self.find_calls.append(query)
        return FakeCursor(self.documents)

    async def find_one(self, query, *args, **kwargs):
        self.find_one_calls.append((query, args, kwargs))
        return self.latest


class FakeDB:
    def __init__(self, historical_prices):
        self.historical_prices = historical_prices


class ChartEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_week_chart_uses_historical_ohlc_and_realtime_price(self):
        historical = FakeCollection(
            [
                {
                    "ticker": "AAPL",
                    "date": datetime(2026, 6, 8, tzinfo=timezone.utc),
                    "open": 198.0,
                    "high": 202.0,
                    "low": 197.5,
                    "close": 201.25,
                    "volume": 123456,
                }
            ]
        )
        realtime = FakeCollection(
            latest={
                "symbol": "AAPL",
                "date": datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc),
                "price": 203.5,
            }
        )

        with (
            patch.object(main, "db", FakeDB(historical)),
            patch.object(main, "trades_col", realtime),
        ):
            response = await main.get_stock_chart_data("aapl", period="1w")

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["ticker"], "AAPL")
        self.assertEqual(len(response["data"]), 2)
        self.assertEqual(historical.find_calls[0]["ticker"], "AAPL")
        self.assertEqual(realtime.find_calls, [])

        historical_point = response["data"][0]
        self.assertEqual(historical_point["price"], 201.25)
        self.assertEqual(historical_point["open"], 198.0)
        self.assertEqual(historical_point["high"], 202.0)
        self.assertEqual(historical_point["low"], 197.5)
        self.assertEqual(historical_point["close"], 201.25)
        self.assertEqual(historical_point["volume"], 123456.0)

        realtime_point = response["data"][1]
        self.assertEqual(realtime_point["price"], 203.5)
        self.assertEqual(realtime_point["open"], 203.5)
        self.assertEqual(realtime_point["high"], 203.5)
        self.assertEqual(realtime_point["low"], 203.5)
        self.assertEqual(realtime_point["close"], 203.5)

    def test_format_chart_point_skips_nan_prices(self):
        point = main.format_chart_point(
            {
                "date": datetime(2026, 6, 8, tzinfo=timezone.utc),
                "close": float("nan"),
            }
        )

        self.assertIsNone(point)


if __name__ == "__main__":
    unittest.main()
