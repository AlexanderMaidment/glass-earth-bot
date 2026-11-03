"""
Model routing: which model forecasts a question within the daily cap.
Amounts are SYNTHETIC.
"""

import unittest
from decimal import Decimal

import support  # noqa: F401  (blocks the network)
import routing
from routing import (
    DEFAULT_FORECASTER,
    HIGH_VALUE_FORECASTER,
    SettingError,
    choose_route,
    ensemble_size_from_env,
    high_value_model_from_env,
    prize_value_usd,
)


class RouteTests(unittest.TestCase):
    def route(self, value, spent, cap="5.00", reserve="2.00", fable=True):
        # What each question needs to start: Opus 0.90, Fable 2.00 (SYNTHETIC).
        return choose_route(
            value_usd=Decimal(value),
            default_question_usd=Decimal("0.90"),
            high_value_question_usd=Decimal("2.00") if fable else None,
            spent_today_usd=Decimal(spent),
            cap_usd=Decimal(cap),
            reserve_usd=Decimal(reserve),
            use_high_value_model=fable,
        )

    def test_high_value_gets_opus_when_fable_is_off(self):
        r = self.route("142", "0", fable=False)
        self.assertEqual((r.action, r.model, r.tier), ("forecast", DEFAULT_FORECASTER, "default"))
        self.assertIn("Fable is off", r.reason)
        # High-value questions may still use the reserve that low-value ones keep out of.
        self.assertEqual(self.route("142", "3.50", fable=False).action, "forecast")
        self.assertEqual(self.route("16", "3.50", fable=False).action, "skip")

    def test_fable_needs_its_worst_case(self):
        with self.assertRaises(ValueError):
            choose_route(
                value_usd=Decimal("142"),
                default_question_usd=Decimal("0.90"),
                high_value_question_usd=None,
                spent_today_usd=Decimal("0"),
                cap_usd=Decimal("5"),
                reserve_usd=Decimal("0"),
                use_high_value_model=True,
            )

    def test_high_value_gets_fable_while_it_fits(self):
        r = self.route("142", "0")
        self.assertEqual((r.action, r.model), ("forecast", HIGH_VALUE_FORECASTER))
        self.assertEqual(r.reserved_usd, Decimal("2.00"))

    def test_high_value_falls_back_to_opus(self):
        # 5.00 - 2.20 = 2.80 left: Fable (2.00) plus a default (0.90) does not fit.
        r = self.route("142", "2.20")
        self.assertEqual((r.action, r.model), ("forecast", DEFAULT_FORECASTER))
        self.assertEqual(r.reserved_usd, Decimal("0.90"))

    def test_high_value_skipped_when_even_opus_does_not_fit(self):
        r = self.route("142", "4.20")
        self.assertEqual(r.action, "skip")
        self.assertEqual(r.reserved_usd, Decimal("0.90"))
        self.assertIn("daily cap: US$0.8000 left; a question needs US$0.9000 to start", r.reason)

    def test_low_value_keeps_out_of_the_reserve(self):
        self.assertEqual(self.route("16", "2.00").model, DEFAULT_FORECASTER)
        skipped = self.route("16", "2.20")
        self.assertEqual(skipped.action, "skip")
        self.assertIn("and US$2.0000 is kept for high-value questions", skipped.reason)

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
                default_question_usd=Decimal("1"),
                high_value_question_usd=Decimal("1"),
                spent_today_usd=Decimal("0"),
                cap_usd=Decimal("1"),
                reserve_usd=Decimal("0"),
                use_high_value_model=True,
            )


class SettingTests(unittest.TestCase):
    """The two repository variables. GitHub passes an unset one as ""."""

    def test_defaults_are_one_forecast_and_fable_off(self):
        self.assertEqual(ensemble_size_from_env(), 1)
        self.assertEqual(ensemble_size_from_env(""), 1)
        self.assertFalse(high_value_model_from_env())
        self.assertFalse(high_value_model_from_env(""))

    def test_good_values(self):
        self.assertEqual([ensemble_size_from_env(v) for v in ("1", "3", " 5 ")], [1, 3, 5])
        self.assertTrue(high_value_model_from_env("true"))
        self.assertTrue(high_value_model_from_env(" TRUE "))
        self.assertFalse(high_value_model_from_env("false"))

    def test_bad_values_stop_with_the_variable_named(self):
        for value in ("0", "6", "-1", "3.0", "three", "1e1", "٣"):
            with self.assertRaises(SettingError, msg=repr(value)) as caught:
                ensemble_size_from_env(value)
            self.assertIn("GLASS_EARTH_ENSEMBLE_SIZE", str(caught.exception))
        for value in ("yes", "1", "on", "tru"):
            with self.assertRaises(SettingError, msg=repr(value)) as caught:
                high_value_model_from_env(value)
            self.assertIn("GLASS_EARTH_HIGH_VALUE_MODEL", str(caught.exception))

    def test_read_from_the_environment(self):
        import os

        os.environ["GLASS_EARTH_ENSEMBLE_SIZE"] = "3"
        os.environ["GLASS_EARTH_HIGH_VALUE_MODEL"] = "true"
        try:
            self.assertEqual((ensemble_size_from_env(), high_value_model_from_env()), (3, True))
        finally:
            os.environ.pop("GLASS_EARTH_ENSEMBLE_SIZE")
            os.environ.pop("GLASS_EARTH_HIGH_VALUE_MODEL")


if __name__ == "__main__":
    unittest.main()
