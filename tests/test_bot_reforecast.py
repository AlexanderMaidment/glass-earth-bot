"""
Scheduled re-forecasts inside the bot: Market Pulse questions are forecast
again when due, FutureEval questions never twice. SYNTHETIC fixtures only.
"""

import asyncio
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from support import OfflineTestCase, load_fixture


def _with_history(page, question_id, forecast_at):
    """Give one sub-question a SYNTHETIC earlier forecast by the bot."""
    page = copy.deepcopy(page)
    for post in page["results"]:
        for question in post["group_of_questions"]["questions"]:
            if question["id"] == question_id:
                n = question["scaling"]["inbound_outcome_count"] + 1
                low, high = question["scaling"]["range_min"], question["scaling"]["range_max"]
                question["scaling"]["continuous_range"] = [
                    low + (high - low) * i / (n - 1) for i in range(n)
                ]
                values = [0.001 + 0.998 * i / (n - 1) for i in range(n)]
                entry = {"start_time": forecast_at.timestamp(), "end_time": None, "forecast_values": values}
                question["my_forecasts"] = {"history": [entry], "latest": entry}
    return page


class ReforecastIntegrationTests(OfflineTestCase):
    def test_market_pulse_question_forecast_a_week_ago_is_forecast_again(self):
        now = datetime.now(timezone.utc)
        page = load_fixture("synthetic_group_posts.json")
        page = _with_history(page, 90011, now - timedelta(days=8))
        page = _with_history(page, 90012, now - timedelta(days=2))
        self.metaculus.posts_page = page
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_on_tournament("market-pulse-26q4", return_exceptions=True))
        ids = sorted(f["question"] for f in self.metaculus.forecasts())
        # 90011 is due (8 days), 90012 is not (2 days), the rest are new.
        self.assertEqual(ids, [90011, 90021, 90022])
        reasons = {record["question_id"]: record["forecast_reason"] for record in self.sealed_records()}
        self.assertIn("scheduled re-forecast", reasons[90011])
        self.assertEqual(reasons[90021], "first forecast")
        self.assertTrue(any("90012" not in s and "not due" in s for s in bot.skipped_not_due))

    def test_ledger_time_counts_when_metaculus_reports_an_older_forecast(self):
        now = datetime.now(timezone.utc)
        page = _with_history(load_fixture("synthetic_group_posts.json"), 90011, now - timedelta(days=8))
        self.metaculus.posts_page = page
        Path("forecasts").mkdir()
        recent = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        Path("forecasts/ledger.jsonl").write_text(
            json.dumps({"question_id": 90011, "recorded_at_utc": recent}) + "\n", encoding="utf-8"
        )
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_on_tournament("market-pulse-26q4", return_exceptions=True))
        self.assertNotIn(90011, [f["question"] for f in self.metaculus.forecasts()])

    def test_futureeval_question_is_never_forecast_twice(self):
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "fall-futureeval-2026"
        page = _with_history(page, 90011, datetime.now(timezone.utc) - timedelta(days=30))
        self.metaculus.posts_page = page
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_on_tournament("fall-futureeval-2026", return_exceptions=True))
        ids = sorted(f["question"] for f in self.metaculus.forecasts())
        self.assertNotIn(90011, ids)
        # A second run skips everything now in the ledger.
        before = len(self.metaculus.forecasts())
        bot2 = self.make_bot(publish=True, skip=True)
        reports = asyncio.run(bot2.forecast_on_tournament("fall-futureeval-2026", return_exceptions=True))
        self.assertEqual(reports, [])
        self.assertEqual(len(self.metaculus.forecasts()), before)


if __name__ == "__main__":
    unittest.main()
