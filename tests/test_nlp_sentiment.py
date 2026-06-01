import unittest

from src.models.nlp_sentiment import FinBERTSentiment


class FakePipeline:
    def __init__(self, results):
        self.results = list(results)

    def __call__(self, text, truncation=True):
        return [self.results.pop(0)]


def make_analyzer(results):
    analyzer = FinBERTSentiment.__new__(FinBERTSentiment)
    analyzer.nlp = FakePipeline(results)
    return analyzer


class FinBERTSentimentTests(unittest.TestCase):
    def test_positive_label_maps_to_positive_score(self):
        analyzer = make_analyzer([
            {"label": "positive", "score": 0.9},
        ])

        score = analyzer.analyze_headline("Company raises full-year guidance")

        self.assertEqual(score, 0.9)

    def test_negative_label_maps_to_negative_score(self):
        analyzer = make_analyzer([
            {"label": "negative", "score": 0.82},
        ])

        score = analyzer.analyze_headline("Company misses revenue estimates")

        self.assertEqual(score, -0.82)

    def test_low_confidence_directional_output_becomes_neutral(self):
        analyzer = make_analyzer([
            {"label": "positive", "score": 0.55},
        ])

        result = analyzer.analyze_article("Company shares edge higher", "")

        self.assertEqual(result["sentiment_label"], "neutral")
        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["sentiment_confidence"], 0.55)

    def test_headline_has_more_weight_than_summary(self):
        analyzer = make_analyzer([
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.6},
        ])

        result = analyzer.analyze_article(
            "Company beats earnings expectations",
            "Management warned about temporary margin pressure.",
        )

        self.assertEqual(result["sentiment_label"], "positive")
        self.assertEqual(result["sentiment_score"], 0.45)
        self.assertEqual(result["sentiment_confidence"], 0.81)

    def test_neutral_label_returns_neutral(self):
        analyzer = make_analyzer([
            {"label": "neutral", "score": 0.95},
        ])

        score = analyzer.analyze_headline("Company announces annual meeting date")

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
