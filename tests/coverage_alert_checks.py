"""
The coverage alert (scripts/coverage_alert.py), offline: SYNTHETIC Metaculus
listings from a fake API and a SYNTHETIC ledger. Nothing here touches the
network, reads .env or spends money. Run with
    python -m unittest discover -s tests -p coverage_alert_checks.py
The name is not test*.py on purpose: the forecasting workflows run only
test*.py before they forecast, so a failure here never stops a forecast.
"""

import contextlib
import fnmatch
import importlib.util
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from support import FAKE_TOKEN, REPO, OfflineTestCase  # noqa: I001  (blocks the network first)

import requests
from forecasting_tools import MetaculusClient

SPEC = importlib.util.spec_from_file_location("coverage_alert", REPO / "scripts" / "coverage_alert.py")
coverage_alert = importlib.util.module_from_spec(SPEC)
sys.modules["coverage_alert"] = coverage_alert
SPEC.loader.exec_module(coverage_alert)

NOW = datetime(2026, 11, 25, 21, 30, tzinfo=timezone.utc)  # 07:30 on 26 Nov in Brisbane
FE = MetaculusClient.CURRENT_AI_COMPETITION_ID
MB = MetaculusClient.CURRENT_MINIBENCH_ID
MP = MetaculusClient.CURRENT_MARKET_PULSE_ID
TITLE = "SYNTHETIC title that must never be printed"


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def question(qid, *, status="open", opened_hours_ago=30.0, closed_hours_ago=None, mine=False, resolution=None):
    opened = NOW - timedelta(hours=opened_hours_ago)
    q = {
        "id": qid,
        "title": f"{TITLE} {qid}",
        "type": "binary",
        "status": status,
        "open_time": stamp(opened),
        "scheduled_close_time": stamp(NOW + timedelta(days=5)),
        "actual_close_time": None,
        "resolution": resolution,
        "my_forecasts": {"history": [{"start_time": 1.0}] if mine else [], "latest": None},
    }
    if closed_hours_ago is not None:
        q["actual_close_time"] = stamp(NOW - timedelta(hours=closed_hours_ago))
        q["scheduled_close_time"] = q["actual_close_time"]
    return q


def post(pid, *questions, group=False):
    p = {"id": pid, "title": f"{TITLE} post {pid}"}
    if group:
        p["group_of_questions"] = {"questions": list(questions)}
    else:
        (p["question"],) = questions
    return p


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


class FakeApi:
    """Serves SYNTHETIC post listings by tournament and status, 100 a page."""

    def __init__(self):
        self.open = {}
        self.closed = {}
        self.calls = []
        self.answer = None  # (status, body) to send instead of a listing

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        if self.answer is not None:
            return FakeResponse(*self.answer)
        assert url == "https://www.metaculus.com/api/posts/", url
        (tournament,) = params["tournaments"]
        statuses = tuple(params["statuses"])
        source = self.open if statuses == ("open",) else self.closed
        assert statuses in (("open",), ("closed", "resolved")), statuses
        posts = source.get(tournament, [])
        offset, limit = int(params["offset"]), int(params["limit"])
        return FakeResponse(200, {"results": posts[offset : offset + limit]})


class CoverageAlertTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.api = FakeApi()
        self.pauses = []
        self.summary = Path("summary.md")
        self.environ = {"METACULUS_TOKEN": FAKE_TOKEN, "GLASS_EARTH_MARKET_PULSE": "true"}

    def write_ledger(self, *question_ids):
        path = Path("forecasts/ledger.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for qid in question_ids:
                f.write(json.dumps({"ledger_version": 3, "sealed": True, "question_id": qid,
                                    "url": f"https://www.metaculus.com/questions/{qid}",
                                    "recorded_at_utc": "2026-11-25T00:00:00Z", "seal_sha256": "0" * 64}) + "\n")

    def run_alert(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = coverage_alert.run(
                now=NOW,
                environ=self.environ,
                get=self.api,
                pause=self.pauses.append,
                step_summary=str(self.summary),
            )
        text = out.getvalue()
        summary = self.summary.read_text(encoding="utf-8") if self.summary.exists() else ""
        return code, text, summary

    # Alex's rule: open, no forecast from us, 24 hours after it opened.

    def test_an_open_question_without_our_forecast_after_24_hours_is_named(self):
        self.api.open[FE] = [post(101, question(1001, opened_hours_ago=30))]
        code, text, summary = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("::error title=Coverage gap::", text)
        self.assertIn("question 1001", text)
        self.assertIn("https://www.metaculus.com/questions/101/", text)
        self.assertIn("question 1001", summary)

    def test_an_open_question_under_24_hours_old_is_not_named(self):
        self.api.open[FE] = [post(101, question(1001, opened_hours_ago=23.5))]
        code, text, _ = self.run_alert()
        self.assertEqual(code, 0)
        self.assertNotIn("1001", text)
        self.assertIn("Coverage OK", text)

    def test_a_forecast_on_metaculus_or_in_the_ledger_counts_as_ours(self):
        self.write_ledger(1002)
        self.api.open[FE] = [
            post(101, question(1001, opened_hours_ago=40, mine=True)),
            post(102, question(1002, opened_hours_ago=40)),
        ]
        code, text, _ = self.run_alert()
        self.assertEqual(code, 0, text)

    def test_upcoming_questions_are_not_named(self):
        self.api.open[MB] = [post(103, question(1003, status="upcoming", opened_hours_ago=-5))]
        code, _, _ = self.run_alert()
        self.assertEqual(code, 0)

    def test_each_question_of_a_group_is_checked(self):
        self.api.open[MP] = [
            post(104, question(1004, opened_hours_ago=48, mine=True), question(1005, opened_hours_ago=48), group=True)
        ]
        code, text, _ = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("question 1005", text)
        self.assertNotIn("question 1004", text)

    # Short questions (FutureEval and MiniBench are open about 3 hours): a
    # question that closed recently with no forecast from us is named too.

    def test_a_question_that_closed_without_our_forecast_is_named_as_missed(self):
        self.api.closed[MB] = [
            post(105, question(1006, status="closed", opened_hours_ago=8, closed_hours_ago=5)),
            post(106, question(1007, status="resolved", opened_hours_ago=60, closed_hours_ago=50)),
            post(107, question(1008, status="resolved", opened_hours_ago=8, closed_hours_ago=5, resolution="annulled")),
            post(108, question(1009, status="closed", opened_hours_ago=8, closed_hours_ago=5, mine=True)),
        ]
        code, text, summary = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("question 1006", text)
        self.assertIn("closed", text)
        for qid in ("1007", "1008", "1009"):
            self.assertNotIn(f"question {qid}", text)
        self.assertIn("question 1006", summary)

    def test_the_missed_lookback_is_36_hours(self):
        self.api.closed[FE] = [
            post(109, question(1010, status="closed", opened_hours_ago=38, closed_hours_ago=35)),
            post(110, question(1011, status="closed", opened_hours_ago=40, closed_hours_ago=37)),
        ]
        code, text, _ = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("question 1010", text)
        self.assertNotIn("question 1011", text)

    # What it prints: IDs and links, never Metaculus's text or the token.

    def test_output_names_ids_and_links_never_titles_or_the_token(self):
        self.api.open[FE] = [post(101, question(1001, opened_hours_ago=30))]
        self.api.closed[MB] = [post(105, question(1006, status="closed", opened_hours_ago=8, closed_hours_ago=5))]
        code, text, summary = self.run_alert()
        self.assertEqual(code, 1)
        for shown in (text, summary):
            self.assertNotIn("SYNTHETIC", shown)
            self.assertNotIn(FAKE_TOKEN, shown)
            self.assertNotIn(chr(0x2014), shown)  # no em dashes

    def test_the_token_goes_only_in_the_authorization_header(self):
        self.api.open[FE] = [post(101, question(1001, opened_hours_ago=1))]
        self.run_alert()
        self.assertTrue(self.api.calls)
        for call in self.api.calls:
            self.assertEqual(call["headers"].get("Authorization"), f"Token {FAKE_TOKEN}")
            self.assertNotIn(FAKE_TOKEN, json.dumps(call["params"]))

    # Reading Metaculus: every page, GETs only, stop on the first failure.

    def test_every_page_is_read(self):
        page_one = [post(200 + i, question(2000 + i, opened_hours_ago=30, mine=True)) for i in range(100)]
        self.api.open[FE] = page_one + [post(400, question(4000, opened_hours_ago=30))]
        code, text, _ = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("question 4000", text)
        offsets = [c["params"]["offset"] for c in self.api.calls if c["params"]["tournaments"] == [FE]]
        self.assertIn(100, [int(o) for o in offsets])

    def test_requests_are_paused_between_and_never_post(self):
        with mock.patch.object(requests, "post", side_effect=AssertionError("no POST allowed")):
            code, _, _ = self.run_alert()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.pauses), len(self.api.calls) - 1)
        self.assertTrue(all(p == coverage_alert.PAUSE_SECONDS for p in self.pauses))

    def test_a_refusal_alerts_and_stops_every_later_request(self):
        self.api.answer = (403, {"detail": "SYNTHETIC refusal"})
        code, text, summary = self.run_alert()
        self.assertEqual(code, 1)
        self.assertEqual(len(self.api.calls), 1)
        self.assertIn("::error title=Coverage check failed::", text)
        self.assertIn("HTTP 403", text)
        self.assertIn("Nothing was checked", text)
        self.assertNotIn("SYNTHETIC", text + summary)

    def test_too_many_requests_or_a_page_that_is_not_json_alerts(self):
        for answer, words in (((429, {}), "429"), ((200, "<html>check</html>"), "not JSON"), ((502, {}), "HTTP 502")):
            with self.subTest(answer=answer[0]):
                self.api = FakeApi()
                self.api.answer = answer
                code, text, _ = self.run_alert()
                self.assertEqual(code, 1)
                self.assertIn(words, text)
                self.assertEqual(len(self.api.calls), 1)

    def test_no_token_alerts_without_asking_metaculus(self):
        self.environ.pop("METACULUS_TOKEN")
        code, text, _ = self.run_alert()
        self.assertEqual(code, 1)
        self.assertIn("METACULUS_TOKEN", text)
        self.assertEqual(self.api.calls, [])

    # Which tournaments: the same as the bot's tournament mode in main.py.

    def test_the_tournaments_are_the_bots_and_market_pulse_follows_its_switch(self):
        self.assertEqual(coverage_alert.tournaments({"GLASS_EARTH_MARKET_PULSE": "true"}), [
            ("FutureEval", FE), ("MiniBench", MB), ("Market Pulse", MP)])
        self.assertEqual(coverage_alert.tournaments({"GLASS_EARTH_MARKET_PULSE": " TRUE "})[-1], ("Market Pulse", MP))
        for off in ({}, {"GLASS_EARTH_MARKET_PULSE": "false"}, {"GLASS_EARTH_MARKET_PULSE": "yes"}):
            self.assertEqual([t for _, t in coverage_alert.tournaments(off)], [FE, MB])
        self.environ["GLASS_EARTH_MARKET_PULSE"] = "false"
        self.run_alert()
        asked = {c["params"]["tournaments"][0] for c in self.api.calls}
        self.assertEqual(asked, {FE, MB})

    def test_the_tournaments_match_main_pys_tournament_mode(self):
        source = (REPO / "main.py").read_text(encoding="utf-8")
        branch = source.split('elif run_mode == "tournament":', 1)[1].split("elif run_mode ==", 1)[0]
        in_main = re.findall(r"client\.(CURRENT_\w+_ID)", branch)
        self.assertEqual(in_main, list(coverage_alert.TOURNAMENT_ATTRIBUTES))
        self.assertIn(coverage_alert.MARKET_PULSE_ENV, source)

    # The workflow: daily, on GitHub, off until a repository variable says so.

    def test_the_workflow_is_gated_read_only_and_outside_the_spend_group(self):
        text = (REPO / ".github" / "workflows" / "coverage_alert.yaml").read_text(encoding="utf-8")
        self.assertIn("if: vars.GLASS_EARTH_COVERAGE_ALERT == 'true'", text)
        self.assertRegex(text, r"schedule:\s*\n\s*- cron: \"\d+ \d+ \* \* \*\"")
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("contents: read", text)
        self.assertNotIn("contents: write", text)
        self.assertEqual(re.findall(r"^\s*group:\s*(\S+)\s*$", text, re.M), ["glass-earth-coverage-alert"])
        secrets = set(re.findall(r"secrets\.([A-Z_]+)", text))
        self.assertEqual(secrets, {"METACULUS_TOKEN"})
        self.assertIn("scripts/coverage_alert.py", text)
        self.assertIn(f"-p {Path(__file__).name}", text)
        self.assertNotIn(chr(0x2014), text)

    def test_the_forecasting_runs_never_run_these_checks(self):
        # Their test step is "unittest discover -s tests", which takes test*.py
        # only, so nothing here can stop a forecasting run.
        self.assertFalse(fnmatch.fnmatch(Path(__file__).name, "test*.py"))
        for name in ("run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml", "learning_loop.yaml"):
            text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("unittest discover -s tests", text)
            self.assertNotIn(Path(__file__).name, text)


if __name__ == "__main__":
    import unittest

    unittest.main()
