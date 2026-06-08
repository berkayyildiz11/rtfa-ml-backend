import unittest
from unittest.mock import AsyncMock, patch

import main


class InvestorAgentEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_endpoint_runs_first_cycle_after_creating_run(self):
        with (
            patch.object(main, "ENABLE_INVESTOR_AGENT", True),
            patch.object(
                main,
                "start_agent_run",
                AsyncMock(return_value={"status": "started", "run": {"run_id": "run-1"}}),
            ) as start_run,
            patch.object(
                main,
                "run_daily_agent_cycle",
                AsyncMock(return_value={"status": "success", "run_id": "run-1"}),
            ) as run_cycle,
        ):
            response = await main.start_investor_agent()

        start_run.assert_awaited_once_with(main.db)
        run_cycle.assert_awaited_once()
        self.assertEqual(response["status"], "started")
        self.assertEqual(response["first_cycle"]["status"], "success")

    async def test_start_endpoint_respects_disabled_flag(self):
        with (
            patch.object(main, "ENABLE_INVESTOR_AGENT", False),
            patch.object(main, "start_agent_run", AsyncMock()) as start_run,
        ):
            response = await main.start_investor_agent()

        start_run.assert_not_awaited()
        self.assertEqual(response["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
