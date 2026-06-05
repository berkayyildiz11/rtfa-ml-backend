import unittest

from data.fetcher import _lightweight_sentiment


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


if __name__ == "__main__":
    unittest.main()
