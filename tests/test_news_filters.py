import unittest

from src.utils.news_filters import analyze_article_relevance, is_article_relevant


class NewsFilterTests(unittest.TestCase):
    def test_direct_company_match_passes(self):
        result = analyze_article_relevance(
            "AAPL",
            "Apple shares rise after stronger iPhone demand",
            "Analysts raised their price targets after the report.",
        )

        self.assertTrue(result["is_relevant"])
        self.assertGreaterEqual(result["relevance_score"], 0.55)
        self.assertIn("apple", result["matched_aliases"])
        self.assertEqual(result["relevance_reason"], "headline_company_match")

    def test_product_only_summary_match_does_not_pass(self):
        result = analyze_article_relevance(
            "AAPL",
            "Retailers prepare for holiday electronics discounts",
            "Several stores expect stronger demand for phones including the iPhone.",
        )

        self.assertFalse(result["is_relevant"])
        self.assertLess(result["relevance_score"], 0.55)
        self.assertEqual(result["matched_aliases"], ["iphone"])

    def test_ambiguous_cost_alias_does_not_match_ordinary_prose(self):
        result = analyze_article_relevance(
            "COST",
            "Rising labor cost pressures retailers",
            "Transportation cost remains a concern across the sector.",
        )

        self.assertFalse(result["is_relevant"])
        self.assertEqual(result["matched_aliases"], ["cost"])

    def test_uppercase_ambiguous_ticker_match_passes(self):
        result = analyze_article_relevance(
            "COST",
            "COST stock rises after earnings",
            "The retailer reported stronger member traffic.",
        )

        self.assertTrue(result["is_relevant"])
        self.assertEqual(result["relevance_reason"], "headline_ticker_match")

    def test_compatibility_wrapper_returns_bool(self):
        result = is_article_relevant(
            "NVDA",
            "NVIDIA unveils new AI chips",
            "The company said data-center demand remains strong.",
        )

        self.assertIsInstance(result, bool)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
