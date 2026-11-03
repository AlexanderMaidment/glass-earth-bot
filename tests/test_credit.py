"""
The OpenRouter credit check at the start of each run, with fake key checks
only: nothing is fetched. Every amount and key here is SYNTHETIC.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

import support
from support import OfflineTestCase, load_fixture

import credit

FAKE_KEY = "sk-or-synthetic-credit-key"


def key_report(remaining, limit="20"):
    """A SYNTHETIC report shaped like OpenRouter's, its numbers as fetch_key_info parses them."""
    return {
        "data": {
            "label": "SYNTHETIC",
            "limit": None if limit is None else Decimal(limit),
            "limit_remaining": None if remaining is None else Decimal(remaining),
            "usage": Decimal("0"),
        }
    }


class FakeResponse(io.BytesIO):
    """Enough of what urlopen returns for fetch_key_info."""


class KeyCheckTests(unittest.TestCase):
    def test_the_real_check_asks_the_key_endpoint_and_reads_decimals(self):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["method"] = request.get_method()
            seen["authorization"] = request.get_header("Authorization")
            seen["timeout"] = timeout
            body = '{"data": {"limit": 20, "limit_remaining": 4.86713448, "usage": 15.13286552}}'
            return FakeResponse(body.encode("utf-8"))

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            info = support.REAL_FETCH_KEY_INFO(FAKE_KEY)
        self.assertEqual(seen["url"], "https://openrouter.ai/api/v1/key")
        self.assertEqual(seen["method"], "GET")
        self.assertEqual(seen["authorization"], f"Bearer {FAKE_KEY}")
        self.assertEqual(seen["timeout"], credit.FETCH_TIMEOUT_SECONDS)
        # Amounts arrive as exact Decimal, never as float.
        self.assertIsInstance(info["data"]["limit_remaining"], Decimal)
        found = credit.read_credit(FAKE_KEY, lambda key: info)
        self.assertEqual((found.remaining_usd, found.limit_usd), (Decimal("4.86713448"), Decimal("20")))

    def test_under_five_dollars_is_low_and_five_or_more_is_not(self):
        for remaining, low in (
            ("4.99", True),
            ("0.10", True),
            ("-0.50", True),
            ("5.00", False),
            ("5", False),
            ("5.01", False),
            ("19.87", False),
        ):
            found = credit.read_credit(FAKE_KEY, lambda key: key_report(remaining))
            self.assertEqual(found.remaining_usd, Decimal(remaining))
            self.assertEqual(found.low, low, remaining)

    def test_a_check_that_fails_is_unknown_never_zero(self):
        def unreachable(key):
            raise TimeoutError("SYNTHETIC timeout")

        cases = [
            (unreachable, "the key check failed: TimeoutError"),
            (lambda key: {}, "no data"),
            (lambda key: {"data": "SYNTHETIC"}, "no data"),
            (lambda key: key_report(None, limit=None), "no spending limit"),
            (lambda key: key_report(None), "no amount left"),
            # A float is refused as money, as everywhere else in the bot.
            (lambda key: {"data": {"limit": 20, "limit_remaining": 4.12}}, "could not be read"),
            (lambda key: {"data": {"limit": 20, "limit_remaining": "four"}}, "could not be read"),
            (lambda key: {"data": {"limit": 20, "limit_remaining": "NaN"}}, "could not be read"),
        ]
        for fetch, why in cases:
            found = credit.read_credit(FAKE_KEY, fetch)
            self.assertIsNone(found.remaining_usd, why)
            self.assertFalse(found.low, why)
            self.assertIn(why, found.unknown_because)
            self.assertIn("Unknown", found.line())
            with self.assertRaises(ValueError):
                found.warning()
        missing = credit.read_credit(None, lambda key: self.fail("no key, so no check"))
        self.assertIn("OPENROUTER_API_KEY is not set", missing.unknown_because)

    def test_the_warning_goes_to_the_log_an_annotation_and_the_summary_page(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        summary = Path(tmp.name) / "summary.md"
        low = credit.read_credit(FAKE_KEY, lambda key: key_report("4.12"))
        printed = []
        with self.assertLogs("credit", level="WARNING") as logs:
            credit.warn(low, github_actions=True, step_summary=str(summary), out=printed.append)
        expected = "Low OpenRouter credit: US$4.1200 left on the bot's key of its US$20 limit, under the US$5.00 warning level."
        self.assertIn(expected, logs.output[0])
        self.assertEqual(printed, [f"::warning title=Low OpenRouter credit::{low.warning()}"])
        self.assertIn(expected, summary.read_text(encoding="utf-8"))
        # Outside GitHub Actions: in the log only.
        printed.clear()
        with self.assertLogs("credit", level="WARNING"):
            credit.warn(low, github_actions=False, step_summary=None, out=printed.append)
        self.assertEqual(printed, [])

    def test_the_failing_exit_is_once_per_utc_day_and_only_when_it_applies(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "credit_warnings.jsonl"
        low = credit.read_credit(FAKE_KEY, lambda key: key_report("4.12"))
        fine = credit.read_credit(FAKE_KEY, lambda key: key_report("5.00"))
        unknown = credit.read_credit(FAKE_KEY, lambda key: {})
        day = datetime(2026, 11, 4, 3, 0, tzinfo=timezone.utc)

        def end(found, *, github_actions=True, dry_run=False, now=day):
            return credit.end_of_run_failure(
                found, github_actions=github_actions, dry_run=dry_run, run_mode="tournament", path=path, now=now
            )

        for found, options in ((fine, {}), (unknown, {}), (low, {"github_actions": False}), (low, {"dry_run": True})):
            self.assertIsNone(end(found, **options))
        self.assertFalse(path.exists())
        self.assertIn("ends with a failure on purpose", end(low))
        self.assertIsNone(end(low, now=day + timedelta(hours=20)))
        # An unreadable line cannot hold back the next day's warning.
        with open(path, "a", encoding="utf-8") as file:
            file.write("{not json\n")
        self.assertIsNotNone(end(low, now=day + timedelta(days=1)))
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("{\"")]
        self.assertEqual([l["utc_day"] for l in lines], ["2026-11-04", "2026-11-05"])
        self.assertEqual(lines[0]["limit_remaining_usd"], "4.12")
        self.assertEqual(lines[0]["warn_below_usd"], "5.00")


class RunTests(OfflineTestCase):
    """main.run with a fake key check, as the workflows run it."""

    def setUp(self):
        super().setUp()
        self.summary = Path(self._tmp.name) / "step_summary.md"
        self.set_env(
            OPENROUTER_API_KEY=FAKE_KEY,
            GITHUB_STEP_SUMMARY=str(self.summary),
            GLASS_EARTH_LEDGER_KEY=support.FAKE_LEDGER_KEY_TEXT,
        )
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "bot-testing-area"
        self.metaculus.posts_page = page

    def run_bot(self, *extra):
        import main

        with contextlib.redirect_stdout(io.StringIO()) as console:
            reports = main.run(["--mode", "test_questions", "--max-questions", "1", *extra])
        self.assertNotIn(FAKE_KEY, console.getvalue())
        return reports, console.getvalue()

    def warning_lines(self):
        return self.read_jsonl("forecasts/credit_warnings.jsonl")

    def test_a_publishing_run_on_github_fails_once_a_utc_day_after_forecasting(self):
        self.set_env(GITHUB_ACTIONS="true")
        self.openrouter_key.report = key_report("4.12")
        with self.assertRaises(SystemExit) as caught:
            self.run_bot()
        message = str(caught.exception)
        self.assertIn("Low OpenRouter credit: US$4.1200 left", message)
        self.assertIn("ends with a failure on purpose", message)
        self.assertNotIn(FAKE_KEY, message)
        # The failure came after the forecasting: the forecast was submitted
        # and what it cost was recorded.
        self.assertEqual(len(self.metaculus.forecasts()), 1)
        self.assertTrue([r for r in self.read_jsonl("forecasts/spend.jsonl") if r["event"] == "model_call"])
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        [line] = self.warning_lines()
        self.assertEqual((line["utc_day"], line["event"]), (today, "low_credit_run_failed"))
        self.assertEqual(line["limit_remaining_usd"], "4.12")
        # A second run on the same UTC day warns again but ends normally.
        reports, console = self.run_bot()
        self.assertEqual(len(reports), 1)
        self.assertIn("::warning title=Low OpenRouter credit::", console)
        self.assertEqual(len(self.warning_lines()), 1)
        # A line for another day does not count, so the next day fails again.
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        Path("forecasts/credit_warnings.jsonl").write_text(
            json.dumps(dict(line, utc_day=yesterday)) + "\n", encoding="utf-8"
        )
        with self.assertRaises(SystemExit):
            self.run_bot()
        self.assertEqual([l["utc_day"] for l in self.warning_lines()], [yesterday, today])
        summary = self.summary.read_text(encoding="utf-8")
        self.assertIn("**Low OpenRouter credit.**", summary)
        self.assertIn("ends with a failure on purpose", summary)
        self.assertEqual(self.openrouter_key.keys, [FAKE_KEY] * 3)

    def test_no_failing_exit_for_a_dry_run_or_off_github(self):
        self.openrouter_key.report = key_report("4.12")
        # A dry run on GitHub: the warning, and the run ends normally.
        self.set_env(GITHUB_ACTIONS="true")
        reports, console = self.run_bot("--dry-run")
        self.assertEqual(len(reports), 1)
        self.assertIn("::warning title=Low OpenRouter credit::", console)
        # A publishing run on your own computer: in the log only.
        os.environ.pop("GITHUB_ACTIONS")
        with self.assertLogs("credit", level="WARNING"):
            reports, console = self.run_bot()
        self.assertEqual(len(reports), 1)
        self.assertNotIn("::warning", console)
        self.assertFalse(Path("forecasts/credit_warnings.jsonl").exists())

    def test_no_warning_at_or_above_five_dollars(self):
        self.set_env(GITHUB_ACTIONS="true")
        for remaining, shown in (("5.00", "5.0000"), ("19.87", "19.8700")):
            self.openrouter_key.report = key_report(remaining)
            reports, console = self.run_bot()
            self.assertEqual(len(reports), 1)
            self.assertIn(
                f"OpenRouter credit left on the bot's key: US${shown} of its US$20 limit", console
            )
            self.assertNotIn("::warning", console)
        self.assertFalse(Path("forecasts/credit_warnings.jsonl").exists())
        self.assertFalse(self.summary.exists())

    def test_a_failed_check_is_unknown_and_the_run_carries_on(self):
        self.set_env(GITHUB_ACTIONS="true")
        # With no report set, the fake key check fails like an unreachable endpoint.
        reports, console = self.run_bot()
        self.assertEqual(len(reports), 1)
        self.assertFalse(isinstance(reports[0], BaseException))
        self.assertIn(
            "OpenRouter credit left on the bot's key: Unknown (the key check failed: "
            "ConnectionRefusedError). The run carries on.",
            console,
        )
        self.assertNotIn("::warning", console)
        self.assertFalse(Path("forecasts/credit_warnings.jsonl").exists())
        self.assertEqual(self.openrouter_key.keys, [FAKE_KEY])

    def test_the_key_check_can_be_passed_to_run(self):
        import main

        seen = []

        def fetch(key):
            seen.append(key)
            return key_report("12.50")

        with contextlib.redirect_stdout(io.StringIO()) as console:
            main.run(["--mode", "test_questions", "--dry-run", "--max-questions", "1"], fetch_key_info=fetch)
        self.assertEqual(seen, [FAKE_KEY])
        self.assertEqual(self.openrouter_key.keys, [])
        self.assertIn("US$12.5000 of its US$20 limit", console.getvalue())


if __name__ == "__main__":
    unittest.main()
