import unittest

from data.fetcher import _dedupe_and_sort_news, _limit_news_by_ticker, _lightweight_sentiment


class NewsFetcherTests(unittest.TestCase):
    def test_lightweight_sentiment_detects_positive_headline(self):
        result = _lightweight_sentiment(
            "Company beats earnings as profits rise",
            "Analysts raised their outlook after strong growth.",
        )

        self.assertEqual(result["sentiment_label"], "positive")
        self.assertGreater(result["sentiment_score"], 0)

    def test_lightweight_sentiment_detects_negative_headline(self):
        result = _lightweight_sentiment(
            "Company misses estimates as shares drop",
            "Management warned of lower demand and margin concerns.",
        )

        self.assertEqual(result["sentiment_label"], "negative")
        self.assertLess(result["sentiment_score"], 0)

    def test_lightweight_sentiment_keeps_neutral_news_neutral(self):
        result = _lightweight_sentiment(
            "Company announces annual meeting date",
            "The event will take place next month.",
        )

        self.assertEqual(result["sentiment_label"], "neutral")
        self.assertEqual(result["sentiment_score"], 0.0)

    def test_news_limit_keeps_ticker_diversity(self):
        items = []
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            for index in range(4):
                items.append(
                    {
                        "ticker": ticker,
                        "timestamp": f"2026-06-08T12:0{index}:00+00:00",
                        "url": f"https://example.com/{ticker}/{index}",
                    }
                )

        limited = _limit_news_by_ticker(_dedupe_and_sort_news(items), max_items=6)
        tickers = {item["ticker"] for item in limited}

        self.assertEqual(len(limited), 6)
        self.assertEqual(tickers, {"AAPL", "MSFT", "NVDA"})


if __name__ == "__main__":
    unittest.main()
