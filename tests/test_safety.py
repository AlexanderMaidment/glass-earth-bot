"""
Repository-wide safety checks: the pinned litellm is not the compromised
release, publishing stays switched off by default, every workflow that can
spend records its spending (and any low-credit warning day), every workflow
that runs the bot gets the ledger key from its secrets and commits each
sealed line with its box, and Glass Earth text has no em dashes.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
EM_DASH = chr(0x2014)


def text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


class DependencyTests(unittest.TestCase):
    def test_locked_litellm_is_not_the_compromised_release(self):
        # The FutureEval resources page (read 2026-11-03) reports that
        # litellm 1.107.0 on PyPI (21 Oct 2026) carried credential-stealing
        # malware. The lock must never resolve to it.
        lock = text(REPO / "uv.lock")
        match = re.search(r'name = "litellm"\nversion = "([^"]+)"', lock)
        self.assertIsNotNone(match)
        self.assertNotEqual(match.group(1), "1.107.0")

    def test_workflows_install_from_the_lock(self):
        for path in WORKFLOWS.glob("*.yaml"):
            for line in text(path).splitlines():
                if "uv sync" in line:
                    self.assertIn("--locked", line, f"{path.name}: {line.strip()}")


class WorkflowGateTests(unittest.TestCase):
    def test_publishing_workflows_stay_off_without_their_switches(self):
        self.assertIn("if: vars.GLASS_EARTH_LIVE == 'true'", text(WORKFLOWS / "run_bot_on_tournament.yaml"))
        self.assertIn("if: vars.GLASS_EARTH_CUP_LIVE == 'true'", text(WORKFLOWS / "run_bot_on_metaculus_cup.yaml"))
        test_bot = text(WORKFLOWS / "test_bot.yaml")
        self.assertIn("default: false", test_bot)
        self.assertIn("--dry-run", test_bot)

    def test_spending_workflows_share_one_queue_test_first_and_commit_spend(self):
        for name in ("run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml", "test_bot.yaml"):
            body = text(WORKFLOWS / name)
            self.assertIn("group: glass-earth-model-spend", body, name)
            self.assertIn("forecasts/spend.jsonl", body, name)
            self.assertLess(body.index("Run the offline tests"), body.index("- name: Run bot"), name)
            self.assertIn("GLASS_EARTH_DAILY_CAP_USD: ${{ vars.GLASS_EARTH_DAILY_CAP_USD }}", body, name)
            # The cap's end date, after which it is the default US$2.00 again.
            self.assertIn(
                "GLASS_EARTH_DAILY_CAP_UNTIL_UTC: ${{ vars.GLASS_EARTH_DAILY_CAP_UNTIL_UTC }}", body, name
            )
            # What a question needs left of the cap to start, per Opus forecast.
            self.assertIn(
                "GLASS_EARTH_QUESTION_RESERVATION_USD: ${{ vars.GLASS_EARTH_QUESTION_RESERVATION_USD }}",
                body,
                name,
            )
            # The last step runs even after a failed bot step, and commits
            # the day a run failed on purpose to warn of low credit.
            last_step = body[body.rindex("- name:"):]
            self.assertIn("if: always()", last_step, name)
            self.assertIn("bash scripts/commit_and_push.sh", last_step, name)
            self.assertIn("forecasts/credit_warnings.jsonl", last_step, name)
            # The forecasting setup comes from repository variables too.
            self.assertIn("GLASS_EARTH_ENSEMBLE_SIZE: ${{ vars.GLASS_EARTH_ENSEMBLE_SIZE }}", body, name)
            self.assertIn("GLASS_EARTH_HIGH_VALUE_MODEL: ${{ vars.GLASS_EARTH_HIGH_VALUE_MODEL }}", body, name)

    def test_workflows_that_run_the_bot_pass_the_ledger_key_and_commit_the_boxes(self):
        runs_bot = sorted(p.name for p in WORKFLOWS.glob("*.yaml") if "main.py" in text(p))
        self.assertEqual(runs_bot, ["run_bot_on_metaculus_cup.yaml", "run_bot_on_tournament.yaml", "test_bot.yaml"])
        for path in WORKFLOWS.glob("*.yaml"):
            body = text(path)
            # The key comes only from the repository's secrets, and only the
            # step that runs the bot gets it.
            self.assertNotIn("vars.GLASS_EARTH_LEDGER_KEY", body, path.name)
            self.assertEqual(body.count("secrets.GLASS_EARTH_LEDGER_KEY"), 1 if path.name in runs_bot else 0, path.name)
            if path.name not in runs_bot:
                continue
            run_step = body[body.index("- name: Run bot"):]
            run_step = run_step[: run_step.index("- name:", 1)]
            self.assertIn("GLASS_EARTH_LEDGER_KEY: ${{ secrets.GLASS_EARTH_LEDGER_KEY }}", run_step, path.name)
            # A sealed line is committed with its box, so neither is ever left behind.
            last_step = body[body.rindex("- name:"):]
            self.assertIn("forecasts/ledger.jsonl forecasts/sealed_boxes.jsonl", last_step, path.name)

    def test_the_bot_workflows_show_only_content_free_output(self):
        for name in ("run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml", "test_bot.yaml"):
            body = text(WORKFLOWS / name)
            # The offline tests' own output goes to a file on the runner; the
            # log shows only the result and any failing test's name.
            step = body[body.index("- name: Run the offline tests"):]
            step = step[: step.index("- name:", 1)]
            self.assertIn('log="$RUNNER_TEMP/offline-tests.log"', step, name)
            self.assertIn('python -m unittest discover -s tests > "$log" 2>&1', step, name)
            shown = [
                line.strip()
                for line in step.splitlines()
                if line.strip().startswith(("grep", "cat", "tail", "head", "echo", "tee", "printf"))
            ]
            self.assertEqual(
                shown,
                ["grep -E '^(Ran |OK)' \"$log\"", "grep -E '^(Ran |FAILED|FAIL: |ERROR: )' \"$log\" || true"],
                name,
            )
            # Nothing from the runner is uploaded, and the bot is run as main.py.
            self.assertNotIn("upload-artifact", body, name)
            self.assertIn("uv run python main.py", body, name)
        # main.py run as a script goes through main(), which keeps a live run
        # on GitHub Actions quiet (run_log.py).
        self.assertTrue(text(REPO / "main.py").rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())'))

    def test_spend_log_merges_like_the_ledger(self):
        attributes = text(REPO / ".gitattributes")
        self.assertIn("forecasts/ledger.jsonl merge=union", attributes)
        self.assertIn("forecasts/sealed_boxes.jsonl merge=union", attributes)
        self.assertIn("forecasts/spend.jsonl merge=union", attributes)
        self.assertIn("forecasts/credit_warnings.jsonl merge=union", attributes)
        self.assertIn("forecasts/** -text", attributes)

    def test_env_and_dry_run_output_are_ignored(self):
        ignore = text(REPO / ".gitignore")
        self.assertRegex(ignore, r"(?m)^\.env$")
        self.assertIn("forecasts/dry_run/", ignore)


class TextTests(unittest.TestCase):
    GLASS_EARTH_FILES = [
        "main.py",
        "ledger.py",
        "sealed_ledger.py",
        "run_log.py",
        "bot_helpers.py",
        "spend.py",
        "credit.py",
        "routing.py",
        "reforecast.py",
        "metered_llm.py",
        "GLASS_EARTH.md",
        "forecasts/README.md",
        "scripts/go_live.py",
        "scripts/backup_salts.py",
        "scripts/commit_and_push.sh",
        "scripts/timestamp_ledger.py",
    ]

    def files(self):
        paths = [REPO / name for name in self.GLASS_EARTH_FILES]
        paths += sorted((REPO / "tests").rglob("*.py")) + sorted((REPO / "tests").rglob("*.json"))
        paths += sorted(WORKFLOWS.glob("*.yaml"))
        return [p for p in paths if p.exists()]

    def test_no_em_dashes(self):
        for path in self.files():
            self.assertNotIn(EM_DASH, text(path), str(path.relative_to(REPO)))

    def test_no_key_shaped_strings(self):
        pattern = re.compile(r"sk-or-v1-[0-9a-f]{20,}|sk-ant-[A-Za-z0-9_-]{20,}")
        for path in self.files():
            self.assertIsNone(pattern.search(text(path)), str(path.relative_to(REPO)))


if __name__ == "__main__":
    unittest.main()
