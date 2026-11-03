"""
The command line the go-live script runs, end to end with fakes: main.run
in test-questions dry-run mode on one SYNTHETIC question, then the go-live
script reading what that run wrote.
"""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from support import OfflineTestCase, load_fixture

SPEC = importlib.util.spec_from_file_location(
    "go_live_for_cli_test", Path(__file__).resolve().parent.parent / "scripts" / "go_live.py"
)
go_live = importlib.util.module_from_spec(SPEC)
sys.modules["go_live_for_cli_test"] = go_live
SPEC.loader.exec_module(go_live)

FAKE_KEY = "sk-or-synthetic-cli-key"


class CliDryRunTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "bot-testing-area"
        self.metaculus.posts_page = page

    def tearDown(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        super().tearDown()

    def test_one_question_dry_run_writes_what_go_live_prints(self):
        import main

        started = time.time()
        # The bot prints emoji; capture them as text whatever the console's code page.
        with contextlib.redirect_stdout(io.StringIO()) as console:
            reports = main.run(
                [
                    "--mode", "test_questions", "--dry-run", "--max-questions", "1",
                    "--daily-cap-usd", "2.00", "--high-value-reserve-usd", "0",
                ]
            )
        self.assertIn("publish=no (dry run)", console.getvalue())
        self.assertIn("Spend (UTC day", console.getvalue())
        self.assertEqual(len(reports), 1)
        self.assertFalse(isinstance(reports[0], BaseException))
        self.assertEqual(self.metaculus.posts, [])
        spend_lines = self.read_jsonl("forecasts/dry_run/spend.jsonl")
        self.assertTrue(spend_lines)
        self.assertTrue(all(line["dry_run"] for line in spend_lines))
        self.assertTrue(all(line["daily_cap_usd"] == "2.00" for line in spend_lines))
        self.assertFalse(Path("forecasts/spend.jsonl").exists())

        printed = []
        ctx = go_live.Context(
            repo=Path.cwd(),
            run=None,
            out=printed.append,
            ask=lambda prompt: "",
            fetch_key_info=lambda key: {},
            python=sys.executable,
        )
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="SYNTHETIC log\n", stderr="")
        self.assertTrue(go_live.show_result(ctx, result, {"OPENROUTER_API_KEY": FAKE_KEY}, started))
        text = "\n".join(printed)
        self.assertIn("ONE DRY-RUN FORECAST", text)
        self.assertIn("*Final Prediction*", text)
        self.assertIn("Model: anthropic/claude-opus-5.5", text)
        self.assertRegex(text, r"Cost: US\$[0-9.]+ over \d+ model calls")
        self.assertNotIn(FAKE_KEY, text)

    def test_the_setup_is_logged(self):
        import main

        with contextlib.redirect_stdout(io.StringIO()) as console:
            main.run(["--mode", "test_questions", "--dry-run", "--max-questions", "1"])
        self.assertIn(
            "Setup: 1 forecast per question from anthropic/claude-opus-5.5 (GLASS_EARTH_ENSEMBLE_SIZE); "
            "anthropic/claude-fable-5.1 for high-value questions is off (GLASS_EARTH_HIGH_VALUE_MODEL).",
            console.getvalue(),
        )
        self.assertIn(
            "An Opus question needs US$0.27 left of the cap to start: US$0.27 per forecast "
            "(GLASS_EARTH_QUESTION_RESERVATION_USD), 4 times that for a conditional question.",
            console.getvalue(),
        )
        self.assertIn("Daily cap: US$2.0000 a UTC day; US$0.8000 of it is kept", console.getvalue())

    def test_unusable_settings_stop_the_run_before_anything_is_fetched_or_spent(self):
        import main

        for name, value in (
            ("GLASS_EARTH_ENSEMBLE_SIZE", "7"),
            ("GLASS_EARTH_ENSEMBLE_SIZE", "three"),
            ("GLASS_EARTH_HIGH_VALUE_MODEL", "yes"),
            ("GLASS_EARTH_DAILY_CAP_USD", "four"),
            ("GLASS_EARTH_DAILY_CAP_UNTIL_UTC", "8 November"),
            ("GLASS_EARTH_DAILY_CAP_UNTIL_UTC", "2026-11-31"),
            ("GLASS_EARTH_QUESTION_RESERVATION_USD", "0"),
            ("GLASS_EARTH_QUESTION_RESERVATION_USD", "a quarter"),
        ):
            self.set_env(**{name: value})
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main.run(["--mode", "test_questions", "--dry-run"])
            self.assertIn(name, str(caught.exception))
            self.assertTrue(str(caught.exception).startswith("Stopped before forecasting"))
            os.environ.pop(name)
        self.assertEqual(self.metaculus.gets, [])
        self.assertEqual(self.models.calls, [])
        self.assertFalse(Path("forecasts").exists())

    def test_the_cap_with_an_end_date_reaches_the_spend_record(self):
        import main

        today = datetime.now(timezone.utc).date()
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="4.00")
        # SYNTHETIC end dates: the last day was yesterday, then it is today.
        for until, cap, said in (
            ((today - timedelta(days=1)).isoformat(), "2.00", "the default again"),
            (today.isoformat(), "4.00", f"until the end of {today.isoformat()}"),
        ):
            self.set_env(GLASS_EARTH_DAILY_CAP_UNTIL_UTC=until)
            Path("forecasts/dry_run/spend.jsonl").unlink(missing_ok=True)
            with contextlib.redirect_stdout(io.StringIO()) as console:
                reports = main.run(["--mode", "test_questions", "--dry-run", "--max-questions", "1"])
            self.assertFalse(isinstance(reports[0], BaseException))
            self.assertIn(said, console.getvalue())
            lines = self.read_jsonl("forecasts/dry_run/spend.jsonl")
            self.assertTrue(lines)
            self.assertTrue(all(line["daily_cap_usd"] == cap for line in lines), until)
