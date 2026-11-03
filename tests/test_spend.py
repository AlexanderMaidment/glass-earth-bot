"""
Prices, money handling and the spend meter on its own. The prices are the
observed OpenRouter list prices recorded in spend.py; every other amount is
SYNTHETIC.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import support  # noqa: F401  (blocks the network)
import spend
from spend import CallUsage, SpendCapReached, SpendRecordUnreadable, cost_of_call


class PriceTests(unittest.TestCase):
    def test_prices_are_decimal_and_match_the_observed_list(self):
        opus = spend.price_for("openrouter/anthropic/claude-opus-5.5")
        fable = spend.price_for("openrouter/anthropic/claude-fable-5.1")
        self.assertEqual(opus.prompt * 1_000_000, Decimal("4"))
        self.assertEqual(opus.completion * 1_000_000, Decimal("20"))
        self.assertEqual(fable.prompt * 1_000_000, Decimal("10"))
        self.assertEqual(fable.completion * 1_000_000, Decimal("50"))
        for price in spend.MODEL_PRICES.values():
            self.assertIsInstance(price.prompt, Decimal)
        self.assertEqual(spend.PRICES_RETRIEVED_AT_UTC, "2026-11-03T04:29:51Z")

    def test_floats_are_refused_as_money(self):
        with self.assertRaises(TypeError):
            spend.to_decimal(0.1)

    def test_unpriced_model_is_refused(self):
        with self.assertRaises(KeyError):
            spend.price_for("openrouter/example/not-a-model")

    def test_cost_from_tokens(self):
        cost, source = cost_of_call(
            "openrouter/anthropic/claude-opus-5.5", CallUsage(1000, 500, None, None)
        )
        self.assertEqual(source, "calculated_from_tokens")
        self.assertEqual(cost, Decimal("0.004") + Decimal("0.01"))

    def test_reported_cost_wins(self):
        cost, source = cost_of_call(
            "openrouter/anthropic/claude-opus-5.5", CallUsage(1000, 500, None, Decimal("0.0123"))
        )
        self.assertEqual((cost, source), (Decimal("0.0123"), "openrouter_reported"))

    def test_web_search_call_without_reported_cost_is_unknown(self):
        cost, source = cost_of_call("openrouter/openai/gpt-6.1-terra:online", CallUsage(1000, 500, None, None))
        self.assertEqual((cost, source), (None, "unknown"))

    def test_long_prompt_price(self):
        price = spend.price_for("openrouter/openai/gpt-6.1-terra")
        self.assertEqual(price.per_token(271999)[0], Decimal("0.000001"))
        self.assertEqual(price.per_token(272000)[0], Decimal("0.000002"))


class CapEnvTests(unittest.TestCase):
    def test_default_cap(self):
        import os

        os.environ.pop(spend.CAP_ENV, None)
        # US$2.00 a day by default, with 40% of it (US$0.80) kept for
        # high-value questions.
        self.assertEqual(spend.daily_cap_from_env(), (Decimal("2.00"), Decimal("0.800")))
        # GitHub passes an unset repository variable as an empty string.
        self.assertEqual(spend.daily_cap_from_env("", ""), (Decimal("2.00"), Decimal("0.800")))
        self.assertEqual(spend.daily_cap_from_env("5.00"), (Decimal("5.00"), Decimal("2.00")))
        self.assertEqual(spend.daily_cap_from_env("1.50")[0], Decimal("1.50"))
        # Unless set, a smaller cap keeps 40% of itself as the reserve.
        self.assertEqual(spend.daily_cap_from_env("1.50")[1], Decimal("0.600"))
        self.assertEqual(spend.daily_cap_from_env("2.00", "0"), (Decimal("2.00"), Decimal("0")))
        # A set reserve can never exceed the cap.
        self.assertEqual(spend.daily_cap_from_env("1.50", "9")[1], Decimal("1.50"))
        with self.assertRaises(ValueError):
            spend.daily_cap_from_env("-1")


class MeterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "spend.jsonl"
        self.now = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self._tmp.cleanup()

    def meter(self, cap="1.00"):
        return spend.SpendMeter(
            cap=Decimal(cap),
            reserve=Decimal("0"),
            log_path=self.path,
            read_paths=[self.path],
            run_mode="offline-test",
            dry_run=True,
            clock=lambda: self.now,
        )

    def test_reads_only_todays_spending(self):
        lines = [
            {"utc_day": "2026-11-03", "cap_basis_usd": "0.40"},
            {"utc_day": "2026-11-02", "cap_basis_usd": "9.00"},
            {"utc_day": "2026-11-03", "cap_basis_usd": "0.10"},
        ]
        self.path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
        self.assertEqual(self.meter().spent_today, Decimal("0.50"))

    def test_unreadable_line_stops_spending(self):
        self.path.write_text('{"utc_day": "2026-11-03", "cap_basis_usd": \n', encoding="utf-8")
        with self.assertRaises(SpendRecordUnreadable):
            self.meter()

    def test_reserve_settle_and_refuse(self):
        meter = self.meter("0.30")
        reservation = meter.reserve_call(Decimal("0.20"), "openrouter/anthropic/claude-opus-5.5", "forecaster")
        self.assertEqual(meter.in_flight, Decimal("0.20"))
        with self.assertRaises(SpendCapReached):
            meter.reserve_call(Decimal("0.20"), "openrouter/anthropic/claude-opus-5.5", "forecaster")
        meter.settle(reservation, CallUsage(1000, 500, None, Decimal("0.05")))
        self.assertEqual((meter.in_flight, meter.spent_this_run), (Decimal("0"), Decimal("0.05")))
        records = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["event"] for r in records], ["call_refused", "model_call"])
        self.assertEqual(records[1]["cost_usd"], "0.05")
        self.assertEqual(records[1]["utc_day"], "2026-11-03")

    def test_new_utc_day_starts_from_that_days_records(self):
        meter = self.meter("1.00")
        meter.settle(meter.reserve_call(Decimal("0.9"), "openrouter/openai/gpt-6-luna", "parser"),
                     CallUsage(10, 10, None, Decimal("0.9")))
        self.assertFalse(meter.can_afford(Decimal("0.2")))
        self.now = datetime(2026, 11, 4, 0, 1, tzinfo=timezone.utc)
        self.assertTrue(meter.can_afford(Decimal("0.2")))

    def test_spend_log_location(self):
        self.assertEqual(spend.spend_log_paths(True), (spend.SPEND_LOG_PATH, [spend.SPEND_LOG_PATH]))
        log, reads = spend.spend_log_paths(False)
        self.assertEqual(log, spend.LOCAL_SPEND_LOG_PATH)
        self.assertIn(spend.SPEND_LOG_PATH, reads)


if __name__ == "__main__":
    unittest.main()
