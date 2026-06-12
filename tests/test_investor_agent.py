import unittest
from datetime import datetime, timezone

from src.services.investor_agent import (
    MAX_OPEN_POSITIONS,
    allocate_cash,
    as_utc_datetime,
    build_investment_opportunity,
    select_buy_candidates,
)
from src.services.investor_agent_v2 import (
    RUN_DAYS as V2_RUN_DAYS,
    select_agent_v2_buy_candidates,
)


def make_prediction(
    *,
    ticker: str = "AAPL",
    period: str = "1d",
    score: float = 0.5,
    confidence: float = 0.8,
    price: float = 100.0,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "period": period,
        "latest_price": price,
        "latest_price_time": "2026-06-09T12:00:00+00:00",
        "prediction": {
            "signal": "better" if score > 0.2 else "neutral",
            "score": score,
            "confidence": {
                "score": confidence,
                "percentage": int(confidence * 100),
                "label": "high",
                "display": f"{int(confidence * 100)}% (high)",
            },
        },
        "weights": {},
        "signals": {},
    }


class InvestorAgentTests(unittest.TestCase):
    def test_opportunity_uses_1w_more_than_1d(self):
        opportunity = build_investment_opportunity(
            "AAPL",
            make_prediction(period="1d", score=0.0, confidence=0.8),
            make_prediction(period="1w", score=0.6, confidence=0.8),
        )

        self.assertTrue(opportunity["eligible"])
        self.assertAlmostEqual(opportunity["combined_score"], 0.36)

    def test_weak_or_low_confidence_signal_is_not_eligible(self):
        weak = build_investment_opportunity(
            "AAPL",
            make_prediction(period="1d", score=0.1, confidence=0.9),
            make_prediction(period="1w", score=0.2, confidence=0.9),
        )
        low_confidence = build_investment_opportunity(
            "MSFT",
            make_prediction(period="1d", score=0.6, confidence=0.4),
            make_prediction(period="1w", score=0.6, confidence=0.4),
        )

        self.assertFalse(weak["eligible"])
        self.assertFalse(low_confidence["eligible"])

    def test_agent_can_choose_fewer_than_all_available_stocks(self):
        opportunities = [
            build_investment_opportunity(
                "AAPL",
                make_prediction(ticker="AAPL", period="1d", score=0.5, confidence=0.8),
                make_prediction(ticker="AAPL", period="1w", score=0.5, confidence=0.8),
            ),
            build_investment_opportunity(
                "MSFT",
                make_prediction(ticker="MSFT", period="1d", score=0.0, confidence=0.8),
                make_prediction(ticker="MSFT", period="1w", score=0.0, confidence=0.8),
            ),
            build_investment_opportunity(
                "NVDA",
                make_prediction(ticker="NVDA", period="1d", score=0.4, confidence=0.9),
                make_prediction(ticker="NVDA", period="1w", score=0.6, confidence=0.9),
            ),
        ]

        candidates = select_buy_candidates(opportunities, positions={})

        self.assertEqual([item["ticker"] for item in candidates], ["NVDA", "AAPL"])

    def test_candidate_selection_caps_open_positions(self):
        opportunities = [
            build_investment_opportunity(
                f"T{idx}",
                make_prediction(ticker=f"T{idx}", period="1d", score=0.6, confidence=0.8),
                make_prediction(ticker=f"T{idx}", period="1w", score=0.6, confidence=0.8),
            )
            for idx in range(10)
        ]

        candidates = select_buy_candidates(opportunities, positions={})

        self.assertEqual(len(candidates), MAX_OPEN_POSITIONS)

    def test_v1_does_not_top_up_held_positions_when_slots_are_full(self):
        positions = {
            f"T{idx}": {
                "ticker": f"T{idx}",
                "quantity": 1.0,
                "avg_cost": 100.0,
                "cost_basis": 100.0,
                "market_value": 100.0,
            }
            for idx in range(MAX_OPEN_POSITIONS)
        }
        opportunities = [
            build_investment_opportunity(
                ticker,
                make_prediction(ticker=ticker, period="1d", score=0.6, confidence=0.8),
                make_prediction(ticker=ticker, period="1w", score=0.6, confidence=0.8),
            )
            for ticker in [*positions.keys(), "NEW"]
        ]

        candidates = select_buy_candidates(opportunities, positions=positions)

        self.assertEqual(candidates, [])

    def test_v2_held_strong_positions_can_be_topped_up_when_slots_are_full(self):
        positions = {
            f"T{idx}": {
                "ticker": f"T{idx}",
                "quantity": 1.0,
                "avg_cost": 100.0,
                "cost_basis": 100.0,
                "market_value": 100.0,
            }
            for idx in range(MAX_OPEN_POSITIONS)
        }
        opportunities = [
            build_investment_opportunity(
                ticker,
                make_prediction(ticker=ticker, period="1d", score=0.6, confidence=0.8),
                make_prediction(ticker=ticker, period="1w", score=0.6, confidence=0.8),
            )
            for ticker in [*positions.keys(), "NEW"]
        ]

        candidates = select_agent_v2_buy_candidates(opportunities, positions=positions)

        self.assertEqual(len(candidates), MAX_OPEN_POSITIONS)
        self.assertNotIn("NEW", [candidate["ticker"] for candidate in candidates])
        self.assertIn("T0", [candidate["ticker"] for candidate in candidates])

    def test_v2_default_run_is_five_days(self):
        self.assertEqual(V2_RUN_DAYS, 5)

    def test_allocation_respects_position_cap_when_topping_up(self):
        candidate = build_investment_opportunity(
            "AAPL",
            make_prediction(ticker="AAPL", period="1d", score=0.8, confidence=0.9),
            make_prediction(ticker="AAPL", period="1w", score=0.8, confidence=0.9),
        )

        allocation = allocate_cash(
            [candidate],
            available_cash=1000.0,
            portfolio_start_value=10000.0,
            positions={
                "AAPL": {
                    "ticker": "AAPL",
                    "quantity": 1.0,
                    "avg_cost": 100.0,
                    "cost_basis": 2400.0,
                    "market_value": 2400.0,
                }
            },
        )

        self.assertEqual(allocation["AAPL"], 100.0)

    def test_allocation_weights_higher_confidence_signal_more(self):
        lower = build_investment_opportunity(
            "AAPL",
            make_prediction(ticker="AAPL", period="1d", score=0.5, confidence=0.6),
            make_prediction(ticker="AAPL", period="1w", score=0.5, confidence=0.6),
        )
        higher = build_investment_opportunity(
            "NVDA",
            make_prediction(ticker="NVDA", period="1d", score=0.5, confidence=0.9),
            make_prediction(ticker="NVDA", period="1w", score=0.5, confidence=0.9),
        )

        allocation = allocate_cash(
            [lower, higher],
            available_cash=1000.0,
            portfolio_start_value=10000.0,
            positions={},
        )

        self.assertGreater(allocation["NVDA"], allocation["AAPL"])

    def test_mongo_naive_datetime_is_treated_as_utc(self):
        parsed = as_utc_datetime(datetime(2026, 6, 9, 12, 0, 0))

        self.assertEqual(parsed.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
