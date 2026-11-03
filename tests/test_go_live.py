"""
scripts/go_live.py, with fakes only: a fake .env holding SYNTHETIC values, a
fake runner standing in for git, gh and the dry run, and a fake OpenRouter
key check. Nothing is set, pushed, run or fetched for real.
"""

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import support  # noqa: F401  (blocks the network for this module too)

SPEC = importlib.util.spec_from_file_location(
    "go_live", Path(__file__).resolve().parent.parent / "scripts" / "go_live.py"
)
go_live = importlib.util.module_from_spec(SPEC)
sys.modules["go_live"] = go_live
SPEC.loader.exec_module(go_live)

TOKEN = "synthetic-metaculus-token-0001"
KEY = "sk-or-synthetic-key-0002"
LEDGER_KEY = base64.urlsafe_b64encode(b"SYNTHETIC-go-live-ledger-key-32b").decode("ascii")
ENV_FILE = f"METACULUS_TOKEN={TOKEN}\nOPENROUTER_API_KEY={KEY}\nGLASS_EARTH_LEDGER_KEY={LEDGER_KEY}\n"


def done(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRunner:
    def __init__(self, repo: Path, **options):
        self.repo = repo
        self.calls = []
        self.options = {
            "tracked": False,
            "ignored": True,
            "remote": "https://github.com/AlexanderMaidment/glass-earth-bot.git",
            "dry_run_ok": True,
            "head": "abc123synthetic",
            "branch": go_live.BRANCH,
            "dirty": "",
            **options,
        }

    def __call__(self, args, **kwargs):
        args = list(args)
        self.calls.append((args, kwargs))
        o = self.options
        if args[:2] == ["git", "ls-files"]:
            return done(0 if o["tracked"] else 1)
        if args[:2] == ["git", "check-ignore"]:
            return done(0 if o["ignored"] else 1)
        if args[:3] == ["git", "remote", "get-url"]:
            return done(0, o["remote"] + "\n")
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return done(0, o["branch"] + "\n")
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return done(0, o["head"] + "\n")
        if args[:2] == ["git", "status"]:
            return done(0, o["dirty"])
        if args[:2] in (["git", "fetch"], ["git", "merge-base"], ["git", "push"]):
            return done(0)
        if args[:2] == ["gh", "--version"]:
            return done(0, "gh version synthetic\n")
        if args[:3] in (["gh", "secret", "set"], ["gh", "variable", "set"]):
            return done(0)
        if len(args) > 1 and args[1] == "main.py":
            return self.dry_run(kwargs)
        raise AssertionError(f"Unexpected command {args}")

    def dry_run(self, kwargs):
        env = kwargs["env"]
        assert env["METACULUS_TOKEN"] == TOKEN and env["OPENROUTER_API_KEY"] == KEY
        # A dry run seals nothing, so it never gets the ledger key.
        assert "GLASS_EARTH_LEDGER_KEY" not in env
        assert LEDGER_KEY not in env.values()
        # The child must write UTF-8, or the bot's emoji banner stops it on Windows.
        assert env["PYTHONIOENCODING"] == "utf-8" and kwargs["encoding"] == "utf-8"
        if not self.options["dry_run_ok"]:
            return done(1, "", f"SYNTHETIC failure; a careless library printed {KEY}")
        folder = self.repo / "forecasts" / "dry_run"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Forecasts-for-synthetic--1-questions.json").write_text(
            json.dumps([{"explanation": f"# SUMMARY\n*Final Prediction*: 35.0%\nSYNTHETIC comment {TOKEN}"}]),
            encoding="utf-8",
        )
        line = {
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "title": "SYNTHETIC question?",
            "group_option": None,
            "url": "https://www.metaculus.com/questions/1/",
            "final_forecast_readable": "35.0%",
            "forecaster_model": "anthropic/claude-opus-5.5",
            "route": "default (prize value US$0.0000 per question)",
            "cost_usd": "0.2834",
            "cost_usd_cap_basis": "0.2834",
            "model_calls": 11,
        }
        with open(folder / "ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        return done(0, f"Bot produced (dry run) 1 forecast(s). token was {TOKEN}\n", "")

    def commands(self):
        return [args for args, _ in self.calls]


class GoLiveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        (self.repo / ".env").write_text(ENV_FILE, encoding="utf-8")
        self.printed = []
        self.answers = []
        self.key_checks = []

    def tearDown(self):
        self._tmp.cleanup()

    def go(self, argv, runner, answer=""):
        def fetch(key):
            self.key_checks.append(key)
            return {"data": {"limit": None, "limit_remaining": None, "usage": 0.5}}

        return go_live.main(
            argv,
            repo=self.repo,
            run=runner,
            out=self.printed.append,
            ask=lambda prompt: answer,
            fetch_key_info=fetch,
        )

    def everything_printed(self):
        return "\n".join(self.printed)

    def assert_no_secret_shown(self, runner):
        text = self.everything_printed()
        log = self.repo / "forecasts" / "dry_run" / "go_live_dry_run.log"
        for secret in (TOKEN, KEY, LEDGER_KEY, LEDGER_KEY.rstrip("=")):
            self.assertNotIn(secret, text)
            for args, _ in runner.calls:
                self.assertNotIn(secret, " ".join(args))
            if log.exists():
                self.assertNotIn(secret, log.read_text(encoding="utf-8"))

    def test_step_one_sets_secrets_on_stdin_runs_one_dry_run_and_stops(self):
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go([], runner), 0)
        secret_calls = [(a, k) for a, k in runner.calls if a[:3] == ["gh", "secret", "set"]]
        self.assertEqual(
            [a[3] for a, _ in secret_calls],
            ["METACULUS_TOKEN", "OPENROUTER_API_KEY", "GLASS_EARTH_LEDGER_KEY"],
        )
        self.assertEqual([k["input"] for _, k in secret_calls], [TOKEN, KEY, LEDGER_KEY])
        self.assertIn(
            "Read METACULUS_TOKEN, OPENROUTER_API_KEY and GLASS_EARTH_LEDGER_KEY from .env (values not shown).",
            self.printed,
        )
        for args, _ in secret_calls:
            self.assertEqual(args[4:], ["--repo", "AlexanderMaidment/glass-earth-bot"])
        dry = [a for a in runner.commands() if len(a) > 1 and a[1] == "main.py"]
        self.assertEqual(len(dry), 1)
        self.assertIn("--dry-run", dry[0])
        self.assertEqual(dry[0][dry[0].index("--max-questions") + 1], "1")
        self.assertEqual(dry[0][dry[0].index("--daily-cap-usd") + 1], "2.00")
        self.assertEqual(dry[0][dry[0].index("--high-value-reserve-usd") + 1], "0")
        text = self.everything_printed()
        self.assertIn("Forecast: 35.0%", text)
        self.assertIn("Cost: US$0.2834 over 11 model calls", text)
        self.assertIn("SYNTHETIC comment", text)
        self.assertIn("STOPPED. Nothing has been published", text)
        self.assertEqual(self.key_checks, [KEY])
        # Step 1 never pushes or switches anything on.
        for args in runner.commands():
            self.assertNotEqual(args[:2], ["git", "push"])
            self.assertNotEqual(args[:3], ["gh", "variable", "set"])
        self.assert_no_secret_shown(runner)
        self.assertTrue((self.repo / go_live.MARKER).exists())

    def test_placeholder_value_stops_before_anything_is_set(self):
        (self.repo / ".env").write_text("METACULUS_TOKEN=\nOPENROUTER_API_KEY=REPLACE_ME\n", encoding="utf-8")
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go([], runner), 1)
        self.assertFalse(any(a[:2] == ["gh", "secret"] for a in runner.commands()))
        self.assertIn("placeholder", self.everything_printed())

    def test_a_missing_or_unusable_ledger_key_stops_before_anything_is_set(self):
        short = base64.urlsafe_b64encode(bytes(16)).decode("ascii")
        for value, said in (
            (None, "empty or still a placeholder"),
            ("", "empty or still a placeholder"),
            (short, "decodes to 16 bytes"),
            (base64.b64encode(bytes([0xFB, 0xFF] * 16)).decode("ascii"), "not URL-safe base64"),
            ("SYNTHETIC-not-a-key", "decodes to"),
        ):
            env = ENV_FILE.replace(f"GLASS_EARTH_LEDGER_KEY={LEDGER_KEY}\n", "")
            if value is not None:
                env += f"GLASS_EARTH_LEDGER_KEY={value}\n"
            (self.repo / ".env").write_text(env, encoding="utf-8")
            self.printed.clear()
            runner = FakeRunner(self.repo)
            self.assertEqual(self.go([], runner), 1)
            self.assertFalse(any(a[:2] == ["gh", "secret"] for a in runner.commands()))
            self.assertFalse(any(len(a) > 1 and a[1] == "main.py" for a in runner.commands()))
            text = self.everything_printed()
            self.assertIn("GLASS_EARTH_LEDGER_KEY", text)
            self.assertIn(said, text)
            if value:
                self.assertNotIn(value, text)
            self.assertFalse((self.repo / go_live.MARKER).exists())

    def test_missing_env_file_stops(self):
        (self.repo / ".env").unlink()
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go([], runner), 1)
        self.assertFalse(any(a[:2] == ["gh", "secret"] for a in runner.commands()))

    def test_tracked_or_unignored_env_stops(self):
        for options in ({"tracked": True}, {"ignored": False}, {"remote": "https://example.invalid/other.git"}):
            runner = FakeRunner(self.repo, **options)
            self.assertEqual(self.go([], runner), 1)
            self.assertFalse(any(a[:2] == ["gh", "secret"] for a in runner.commands()))

    def test_failed_dry_run_shows_no_old_forecast_and_masks_the_log(self):
        folder = self.repo / "forecasts" / "dry_run"
        folder.mkdir(parents=True)
        old = folder / "Forecasts-for-old--1-questions.json"
        old.write_text(json.dumps([{"explanation": "OLD SYNTHETIC comment"}]), encoding="utf-8")
        os.utime(old, (time.time() - 3600, time.time() - 3600))
        runner = FakeRunner(self.repo, dry_run_ok=False)
        self.assertEqual(self.go([], runner), 1)
        text = self.everything_printed()
        self.assertIn("did not produce a forecast", text)
        self.assertNotIn("OLD SYNTHETIC comment", text)
        self.assertIn("[hidden]", text)
        self.assert_no_secret_shown(runner)
        self.assertFalse((self.repo / go_live.MARKER).exists())

    def test_step_two_needs_a_good_step_one(self):
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go(["--publish"], runner, answer="PUBLISH"), 1)
        self.assertFalse(any(a[:2] == ["git", "push"] for a in runner.commands()))

    def test_step_two_needs_the_typed_word(self):
        runner = FakeRunner(self.repo)
        self.go([], runner)
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go(["--publish"], runner, answer="yes"), 1)
        self.assertFalse(any(a[:2] == ["git", "push"] for a in runner.commands()))
        self.assertFalse(any(a[:3] == ["gh", "variable", "set"] for a in runner.commands()))

    def test_step_two_refuses_after_the_code_changed(self):
        self.go([], FakeRunner(self.repo))
        runner = FakeRunner(self.repo, head="def456synthetic")
        self.assertEqual(self.go(["--publish"], runner, answer="PUBLISH"), 1)
        self.assertFalse(any(a[:2] == ["git", "push"] for a in runner.commands()))

    def test_step_two_pushes_then_switches_on(self):
        self.go([], FakeRunner(self.repo))
        runner = FakeRunner(self.repo)
        self.assertEqual(self.go(["--publish"], runner, answer="PUBLISH"), 0)
        commands = runner.commands()
        push = commands.index(["git", "push", "origin", f"{go_live.BRANCH}:main"])
        variable = commands.index(
            ["gh", "variable", "set", "GLASS_EARTH_LIVE", "--body", "true", "--repo", go_live.REPOSITORY]
        )
        self.assertLess(push, variable)
        self.assertFalse(any(a[:3] == ["gh", "secret", "set"] for a in commands))


if __name__ == "__main__":
    unittest.main()
