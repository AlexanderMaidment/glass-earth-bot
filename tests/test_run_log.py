"""
What a live run on GitHub Actions shows in its log (run_log.py). Each run is
the real main.main() in a process of its own (tests/live_run_driver.py),
with the offline fakes: SYNTHETIC questions, model answers, keys and costs,
and no network. That process's stdout and stderr are exactly what the run's
log on GitHub would show. The helpers are also tested on their own.
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from support import FAKE_TOKEN  # support also blocks the network

import run_log
from spend import SpendCapReached

DRIVER = Path(__file__).resolve().parent / "live_run_driver.py"
MARKER = "SYNTHETIC-CONTENT-MARKER"
OPTIONS = ("SYNTHETIC Amberflow", "SYNTHETIC Brightwater", "SYNTHETIC Coldbrook")
# Text from the questions, the model answers, the forecasts and the stray
# writes. Every title, option label, research and reasoning text in the
# fakes says SYNTHETIC; the group labels and forecast forms are listed too.
CONTENT = (
    "SYNTHETIC",
    "Examplecorp",
    "Samplesoft",
    "Revenue (synthetic)",
    "Gross margin (synthetic)",
    "made-up",
    "nothing decisive",
    "Reference class",
    "Probability",
    "probability_yes",
    "Percentile",
    "continuous_cdf",
    "30.0%",
    "50.0%",
    "stray",
    MARKER,
    FAKE_TOKEN,
    "sk-or-synthetic-driver-key",
)
# How every line a quiet run shows starts.
ALLOWED_STARTS = (
    "Live run on GitHub Actions: ",
    "\U0001f916  Running mode=",
    "Setup: ",
    "OpenRouter credit left on the bot's key: ",
    "::warning title=Low OpenRouter credit::",
    "Daily cap: ",
    "An Opus question needs ",
    "Spend (UTC day ",
    "Ledger reveals: ",
    "::warning title=Sealed ledger::",
    "Forecasting on ",
    "Stopped: ",
    "Low OpenRouter credit: ",
)
NOT_DUE_LINE = re.compile(r"^\d+ Market Pulse question\(s\) were not due for a new forecast\.$")
QUESTION_LINE = re.compile(
    r"^Question (\d+|\?) \((https://www\.metaculus\.com/questions/\d+/?|no address)\): "
    r"(posted|failed \([A-Za-z]+( \([A-Za-z, ]+\))?\)|skipped \((not open|already forecast|not due|daily cap)\))"
    r"(, cost (US\$[0-9.]+ over \d+ model calls|Unknown \(\d+ of \d+ model calls had no reported cost; "
    r"at least US\$[0-9.]+\)))?$"
)
COUNTS_LINE = re.compile(
    r"^Questions: \d+ found, \d+ selected, \d+ skipped \(\d+ not open, \d+ already forecast, "
    r"\d+ not due, \d+ daily cap\), \d+ posted, \d+ failed\.$"
)
STOPPED_LINE = re.compile(r"^Stopped: [A-Za-z]+( \([A-Za-z, ]+\))?\. The details are only in the run's log file on the runner\.$")
# Written while the test harness imports forecasting-tools, before the run
# starts and without its token (a live run on GitHub has the token by then).
HARNESS_IMPORT_LINE = "METACULUS_TOKEN environment variable and/or token field not set"


class LiveRunLogTests(unittest.TestCase):
    """The driver's runs, started together once for every test here."""

    RUNS = {
        "batch": ("batch", []),
        "failure": ("failure", []),
        "crash": ("crash", []),
        "dry": ("batch", ["--mode", "test_questions", "--dry-run", "--max-questions", "1"]),
    }

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folders, cls.results, started = {}, {}, {}
        environment = dict(os.environ, PYTHONUTF8="1")
        for name, (scenario, argv) in cls.RUNS.items():
            folder = Path(cls._tmp.name) / name
            folder.mkdir()
            cls.folders[name] = folder
            out, err = open(folder / "stdout.txt", "wb"), open(folder / "stderr.txt", "wb")
            process = subprocess.Popen(
                [sys.executable, str(DRIVER), scenario, str(folder), *argv],
                stdout=out,
                stderr=err,
                env=environment,
            )
            started[name] = (process, out, err)
        for name, (process, out, err) in started.items():
            process.wait(timeout=600)
            out.close()
            err.close()
            folder = cls.folders[name]
            cls.results[name] = (
                process.returncode,
                (folder / "stdout.txt").read_text(encoding="utf-8"),
                (folder / "stderr.txt").read_text(encoding="utf-8"),
            )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def log_file(self, name):
        return self.folders[name] / "runner_temp" / "glass-earth-run-4242.log"

    def assert_only_allowed_lines(self, text):
        for line in text.splitlines():
            if not line.strip() or line == HARNESS_IMPORT_LINE:
                continue
            if line.startswith("Question "):
                self.assertRegex(line, QUESTION_LINE)
            elif line.startswith("Questions: "):
                self.assertRegex(line, COUNTS_LINE)
            elif line.startswith("Stopped: "):
                self.assertRegex(line, STOPPED_LINE)
            elif line[:1].isdigit():
                self.assertRegex(line, NOT_DUE_LINE)
            else:
                self.assertTrue(line.startswith(ALLOWED_STARTS), f"Not an allowed line: {line!r}")

    def assert_no_content(self, text):
        for content in CONTENT:
            self.assertNotIn(content, text)

    def assert_quiet(self, name):
        code, out, err = self.results[name]
        self.assert_only_allowed_lines(out)
        self.assert_only_allowed_lines(err)
        self.assert_no_content(out + err)
        # Only what the bot shows goes to stdout; stderr keeps nothing of the run.
        self.assertEqual([line for line in err.splitlines() if line.strip() and line != HARNESS_IMPORT_LINE], [])
        summary = self.folders[name] / "step_summary.md"
        if summary.exists():
            self.assert_no_content(summary.read_text(encoding="utf-8"))
        # The full log is on the runner's disk, outside the checkout.
        self.assertTrue(self.log_file(name).exists())
        self.assertEqual(list((self.folders[name] / "checkout").rglob("*.log")), [])
        return code, out, self.log_file(name).read_text(encoding="utf-8")

    def test_a_live_batch_shows_only_content_free_lines(self):
        code, out, log = self.assert_quiet("batch")
        self.assertEqual(code, 0, out)
        self.assertIn("Live run on GitHub Actions: this log shows only lines with no question content.", out)
        self.assertIn("\U0001f916  Running mode=tournament, publish=yes", out)
        self.assertIn("OpenRouter credit left on the bot's key: US$12.5000", out)
        self.assertIn("Daily cap: US$2.0000 a UTC day", out)
        self.assertIn("Ledger reveals: 0 forecast(s) revealed", out)
        for question_id in (70001, 70002, 70003, 70009, 90011, 90012, 90021, 90022):
            self.assertRegex(out, rf"(?m)^Question {question_id} \(https://www\.metaculus\.com/questions/\d+\): posted, cost ")
        self.assertIn(
            "Questions: 8 found, 8 selected, 0 skipped (0 not open, 0 already forecast, 0 not due, "
            "0 daily cap), 8 posted, 0 failed.",
            out,
        )
        self.assertIn("8 skipped (0 not open, 4 already forecast, 4 not due, 0 daily cap)", out)
        self.assertRegex(out, r"(?m)^Spend \(UTC day [0-9-]+\): US\$[0-9.]+ of the US\$2\.0000 daily cap, US\$[0-9.]+ in this run")
        # Everything else went to the log file: research, reasoning, the
        # forecasts, the template's report summary, and every stray write.
        for text in ("SYNTHETIC research", "Reference class and base rate", "Probability: 30%", "Bot: GlassEarthBot",
                     "stray print", "stray stderr", "stray fd 1", "stray fd 2", "stray __stdout__",
                     "stray warning", "stray LiteLLM warning", *OPTIONS):
            self.assertIn(text, log)

    def test_a_failed_question_shows_only_its_exception_type(self):
        code, out, log = self.assert_quiet("failure")
        self.assertEqual(code, 1)
        self.assertRegex(
            out,
            r"(?m)^Question 70002 \(https://www\.metaculus\.com/questions/7002\): failed \([A-Za-z]+( \([A-Za-z, ]+\))?\), cost ",
        )
        self.assertRegex(out, r"(?m)^Question 70001 \(https://www\.metaculus\.com/questions/7001\): posted, cost ")
        self.assertIn("Stopped: RuntimeError. The details are only in the run's log file on the runner.", out)
        self.assertIn(MARKER, log)
        self.assertIn("Traceback", log)

    def test_an_exception_that_ends_the_run_shows_only_its_type(self):
        code, out, log = self.assert_quiet("crash")
        self.assertEqual(code, 1)
        self.assertIn("Stopped: RuntimeError. The details are only in the run's log file on the runner.", out)
        self.assertNotRegex(out, r"(?m)^Question ")
        self.assertIn(MARKER, log)
        self.assertIn("Traceback", log)

    def test_a_dry_run_on_github_actions_keeps_its_full_log(self):
        code, out, err = self.results["dry"]
        self.assertEqual(code, 0, out + err)
        self.assertIn("publish=no (dry run)", out)
        # As before: the research and reasoning are in the console.
        self.assertIn("SYNTHETIC research", out + err)
        self.assertIn("Reference class and base rate", out + err)
        self.assertFalse(self.log_file("dry").exists())


class HelperTests(unittest.TestCase):
    def test_only_a_publishing_run_on_github_actions_is_quiet(self):
        self.assertTrue(run_log.quiet_wanted(True, {"GITHUB_ACTIONS": "true"}))
        self.assertFalse(run_log.quiet_wanted(False, {"GITHUB_ACTIONS": "true"}))
        self.assertFalse(run_log.quiet_wanted(True, {}))
        self.assertFalse(run_log.quiet_wanted(True, {"GITHUB_ACTIONS": "false"}))

    def test_an_error_shows_as_types_never_its_message(self):
        try:
            try:
                raise SpendCapReached(f"{MARKER} SYNTHETIC: model output")
            except SpendCapReached as cause:
                raise RuntimeError(f"{MARKER} SYNTHETIC: wrapped") from cause
        except RuntimeError as error:
            chained = error
        group = ExceptionGroup(f"{MARKER} group", [chained, ValueError(f"{MARKER} SYNTHETIC")])
        self.assertEqual(run_log.error_kind(chained), "RuntimeError (SpendCapReached)")
        self.assertEqual(run_log.error_kind(group), "ExceptionGroup (SpendCapReached, ValueError)")
        self.assertEqual(run_log.error_kind(KeyError(f"{MARKER}")), "KeyError")
        self.assertEqual(run_log.failed(group), "failed (ExceptionGroup (SpendCapReached, ValueError))")

    def test_a_question_line_has_its_id_address_outcome_and_cost(self):
        self.assertEqual(
            run_log.question_line(7, "https://www.metaculus.com/questions/70/a-title-slug/", 70, run_log.skipped("daily cap")),
            "Question 7 (https://www.metaculus.com/questions/70/): skipped (daily cap)",
        )
        self.assertEqual(
            run_log.question_line(None, None, None, run_log.done(True), "US$0.10 over 5 model calls"),
            "Question ? (no address): posted, cost US$0.10 over 5 model calls",
        )
        with self.assertRaises(ValueError):
            run_log.skipped("SYNTHETIC: a reason with question text")

    def test_a_stop_shows_its_message_only_when_the_bot_made_it_from_its_settings(self):
        stops = (
            (SystemExit("Stopped before forecasting: GLASS_EARTH_LEDGER_KEY is not set."), 1,
             "Stopped before forecasting: GLASS_EARTH_LEDGER_KEY is not set."),
            (SystemExit("Low OpenRouter credit: US$4.1200 left on the bot's key."), 1,
             "Low OpenRouter credit: US$4.1200 left on the bot's key."),
            (SystemExit(f"{MARKER} SYNTHETIC: text from somewhere"), 1, f"Stopped. {run_log.DETAILS}"),
            (SystemExit(3), 3, f"Stopped with exit code 3. {run_log.DETAILS}"),
            (SystemExit(0), 0, None),
            (SystemExit(), 0, None),
        )
        for stop, code, shown in stops:
            with contextlib.redirect_stdout(io.StringIO()) as console:
                with self.assertLogs("run_log", level="ERROR") if isinstance(stop.code, str) else contextlib.nullcontext():
                    self.assertEqual(run_log.public_exit(stop), code)
            self.assertEqual(console.getvalue().strip(), shown or "")

    def test_the_log_file_is_never_in_the_checkout(self):
        here = Path.cwd()
        self.assertEqual(
            run_log.log_file_path({"RUNNER_TEMP": str(here / "SYNTHETIC-inside"), "GITHUB_RUN_ID": "7"}),
            Path(tempfile.gettempdir()) / "glass-earth-run-7.log",
        )
        outside = Path(tempfile.gettempdir()) / "SYNTHETIC-runner-temp"
        self.assertEqual(run_log.log_file_path({"RUNNER_TEMP": str(outside), "GITHUB_RUN_ID": "7"}), outside / "glass-earth-run-7.log")


if __name__ == "__main__":
    unittest.main()
