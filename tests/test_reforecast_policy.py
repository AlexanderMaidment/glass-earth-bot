"""
The re-forecast policy on its own: when a Market Pulse question is due again,
and that FutureEval and MiniBench questions never are. Times are SYNTHETIC.
"""

import unittest
from datetime import datetime, timedelta, timezone

from reforecast import MARKET_PULSE_POLICY, forecast_due, latest, policy_for

NOW = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)


class PolicyTests(unittest.TestCase):
    def due(self, last_hours_ago=None, closes_in_hours=None):
        return forecast_due(
            now=NOW,
            close_time=None if closes_in_hours is None else NOW + timedelta(hours=closes_in_hours),
            last_forecast_at=None if last_hours_ago is None else NOW - timedelta(hours=last_hours_ago),
            policy=MARKET_PULSE_POLICY,
        )

    def test_never_forecast_is_due(self):
        self.assertEqual(self.due(None, 500), (True, "first forecast"))

    def test_recent_forecast_is_not_due(self):
        due, reason = self.due(72, 500)
        self.assertFalse(due)
        self.assertIn("not due", reason)

    def test_week_old_forecast_is_due(self):
        due, reason = self.due(7 * 24, 500)
        self.assertTrue(due)
        self.assertIn("scheduled re-forecast", reason)

    def test_final_refresh_before_close(self):
        due, reason = self.due(21, 30)
        self.assertTrue(due)
        self.assertIn("final refresh", reason)

    def test_no_second_final_refresh_within_twenty_hours(self):
        self.assertFalse(self.due(5, 30)[0])

    def test_closed_question_is_never_due(self):
        self.assertEqual(self.due(None, -1), (False, "closed"))
        self.assertEqual(self.due(500, 0), (False, "closed"))

    def test_naive_times_are_read_as_utc(self):
        naive = (NOW - timedelta(days=8)).replace(tzinfo=None)
        self.assertTrue(
            forecast_due(now=NOW, close_time=None, last_forecast_at=naive, policy=MARKET_PULSE_POLICY)[0]
        )

    def test_one_forecast_tournaments_never_get_a_policy(self):
        self.assertIsNotNone(policy_for(["market-pulse-26q4"]))
        self.assertIsNone(policy_for(["fall-futureeval-2026"]))
        self.assertIsNone(policy_for(["minibench"]))
        self.assertIsNone(policy_for(["market-pulse-26q4", "fall-futureeval-2026"]))
        self.assertIsNone(policy_for(["bot-testing-area"]))

    def test_latest_takes_the_later_time(self):
        a = NOW - timedelta(days=3)
        b = NOW - timedelta(days=1)
        self.assertEqual(latest(a, None, b), b)
        self.assertIsNone(latest(None, None))


if __name__ == "__main__":
    unittest.main()
