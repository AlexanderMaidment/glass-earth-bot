"""
Model routing: which model forecasts a question within the daily cap.
Amounts are SYNTHETIC.
"""

import unittest
from decimal import Decimal

import support  # noqa: F401  (blocks the network)
import routing
from routing import DEFAULT_FORECASTER, HIGH_VALUE_FORECASTER, choose_route, prize_value_usd


class RouteTests(unittest.TestCase):
    def route(self, value, spent, cap="5.00", reserve="2.00"):
        return choose_route(
            value_usd=Decimal(value),
            worst_case_default_usd=Decimal("0.90"),
            worst_case_high_value_usd=Decimal("2.00"),
            spent_today_usd=Decimal(spent),
            cap_usd=Decimal(cap),
            reserve_usd=Decimal(reserve),
        )

    def test_high_value_gets_fable_while_it_fits(self):
        r = self.route("142", "0")
        self.assertEqual((r.action, r.model), ("forecast", HIGH_VALUE_FORECASTER))

    def test_high_value_falls_back_to_opus(self):
        # 5.00 - 2.20 = 2.80 left: Fable (2.00) plus a default (0.90) does not fit.
        r = self.route("142", "2.20")
        self.assertEqual((r.action, r.model), ("forecast", DEFAULT_FORECASTER))

    def test_high_value_skipped_when_even_opus_does_not_fit(self):
        r = self.route("142", "4.20")
        self.assertEqual(r.action, "skip")
        self.assertIn("daily cap", r.reason)

    def test_low_value_keeps_out_of_the_reserve(self):
        self.assertEqual(self.route("16", "2.00").model, DEFAULT_FORECASTER)
        self.assertEqual(self.route("16", "2.20").action, "skip")

    def test_prize_values(self):
        self.assertTrue(routing.is_high_value(prize_value_usd(["fall-futureeval-2026"], 1.0)))
        self.assertTrue(routing.is_high_value(prize_value_usd(["market-pulse-26q4"], None)))
        self.assertFalse(routing.is_high_value(prize_value_usd(["minibench-2026-11-02"], 1.0)))
        self.assertFalse(routing.is_high_value(prize_value_usd(["bot-testing-area"], 1.0)))
        # A down-weighted question is not high value.
        self.assertFalse(routing.is_high_value(prize_value_usd(["fall-futureeval-2026"], 0.5)))

    def test_route_refuses_float_money(self):
        with self.assertRaises(TypeError):
            choose_route(
                value_usd=1.0,
                worst_case_default_usd=Decimal("1"),
                worst_case_high_value_usd=Decimal("1"),
                spent_today_usd=Decimal("0"),
                cap_usd=Decimal("1"),
                reserve_usd=Decimal("0"),
            )


if __name__ == "__main__":
    unittest.main()
