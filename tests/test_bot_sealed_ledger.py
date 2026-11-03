"""
The sealed ledger inside the bot, on SYNTHETIC questions and SYNTHETIC model
answers: a publishing run writes only sealed lines and their boxes, a live run
without a usable ledger key stops before anything is fetched or spent, dry
runs are unchanged, and the ledger's readers count each forecast once.
"""

import asyncio
import base64
import contextlib
import io
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion

import sealed_ledger as sl
from ledger import ForecastLedger, ledger_entry
from support import FAKE_LEDGER_KEY, FAKE_LEDGER_KEY_TEXT, FAKE_TOKEN, OfflineTestCase, load_fixture

COMMON = dict(
    background_info="SYNTHETIC background.",
    resolution_criteria="SYNTHETIC criteria.",
    fine_print="SYNTHETIC fine print.",
    tournament_slugs=["bot-testing-area"],
)
BINARY_TITLE = "SYNTHETIC: Will the made-up comet be seen from the made-up town?"
CHOICE_TITLE = "SYNTHETIC: Which made-up river floods first?"
OPTIONS = ["SYNTHETIC Amberflow", "SYNTHETIC Brightwater", "SYNTHETIC Coldbrook"]


def binary(close_time=None):
    return BinaryQuestion(
        question_text=BINARY_TITLE,
        id_of_post=101,
        id_of_question=1001,
        page_url="https://www.metaculus.com/questions/101/",
        close_time=close_time,
        **COMMON,
    )


def multiple_choice():
    return MultipleChoiceQuestion(
        question_text=CHOICE_TITLE,
        id_of_post=102,
        id_of_question=1002,
        page_url="https://www.metaculus.com/questions/102/",
        options=OPTIONS,
        **COMMON,
    )


def all_keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from all_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from all_keys(inner)


class PublishingTests(OfflineTestCase):
    def test_a_publishing_run_writes_only_sealed_lines_and_their_boxes(self):
        import main

        bot = self.make_bot(publish=True, skip=True)
        order = []
        real_append = main.append_json_line

        def watched_box_append(path, value):
            order.append(("box", value["seal_sha256"]))
            real_append(path, value)

        real_ledger_append = bot.ledger.append

        def watched_ledger_append(entry):
            order.append(("line", entry["seal_sha256"]))
            real_ledger_append(entry)

        main.append_json_line = watched_box_append
        bot.ledger.append = watched_ledger_append
        try:
            reports = asyncio.run(bot.forecast_questions([binary(), multiple_choice()], return_exceptions=True))
        finally:
            main.append_json_line = real_append
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])
        self.assertEqual(len(self.metaculus.forecasts()), 2)

        # Each box was written before its sealed line.
        seals = [seal for kind, seal in order if kind == "line"]
        self.assertEqual(order, [step for seal in seals for step in (("box", seal), ("line", seal))])

        lines = self.read_jsonl("forecasts/ledger.jsonl")
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(tuple(line), sl.SEALED_LINE_FIELDS)
            self.assertIs(line["sealed"], True)
        self.assertEqual(
            [(l["question_id"], l["url"]) for l in lines],
            [(1001, "https://www.metaculus.com/questions/101/"), (1002, "https://www.metaculus.com/questions/102/")],
        )
        boxes = self.read_jsonl("forecasts/sealed_boxes.jsonl")
        self.assertEqual([tuple(b) for b in boxes], [sl.BOX_LINE_FIELDS] * 2)
        self.assertEqual([b["seal_sha256"] for b in boxes], seals)

        # No question text, option label, forecast, model or cost in the
        # ledger file, and no readable text in the boxes file.
        ledger_text = Path("forecasts/ledger.jsonl").read_text(encoding="utf-8")
        boxes_text = Path("forecasts/sealed_boxes.jsonl").read_text(encoding="utf-8")
        forecasts = [{k: v for k, v in f.items() if k not in ("question", "source")} for f in self.metaculus.forecasts()]
        for text in (BINARY_TITLE, CHOICE_TITLE, *OPTIONS, "30.0%", "probability", "opus", "cost", "SYNTHETIC"):
            self.assertNotIn(text, ledger_text)
        for text in (BINARY_TITLE, CHOICE_TITLE, *OPTIONS, "SYNTHETIC", FAKE_TOKEN):
            self.assertNotIn(text, boxes_text)
        self.assertFalse(set(all_keys(lines)) - set(sl.SEALED_LINE_FIELDS))
        self.assertFalse({"probability_yes", "probability_yes_per_category"} & set(all_keys(lines)))

        # The records hold what was sent, without the text copied from Metaculus.
        records = self.sealed_records()
        self.assertEqual([r["final_forecast"] for r in records], forecasts)
        for record in records:
            self.assertEqual(set(record), {*sl.RECORD_FIELDS, "salt"})
            self.assertNotIn(BINARY_TITLE, json.dumps(record))
            self.assertNotIn(CHOICE_TITLE, json.dumps(record))
            self.assertNotIn("forecast_readable", set(all_keys(record)))
            self.assertRegex(record["salt"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(records[0]["salt"], records[1]["salt"])

    def test_the_record_keeps_the_scheduled_close_time(self):
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(
            bot.forecast_questions([binary(datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc))], return_exceptions=True)
        )
        [record] = self.sealed_records()
        self.assertEqual(record["scheduled_close_time_utc"], "2026-11-20T12:00:00Z")
        self.assertNotIn("scheduled_close_time_utc", self.read_jsonl("forecasts/ledger.jsonl")[0])

    def test_every_field_of_a_full_line_is_either_recorded_or_withheld(self):
        entry = ledger_entry(question_id=1, final_forecast={}, context={}, run_mode="x")
        self.assertEqual(set(entry), set(sl.RECORD_FIELDS) | sl.WITHHELD_FIELDS)
        self.assertFalse(set(sl.RECORD_FIELDS) & sl.WITHHELD_FIELDS)
        self.assertLessEqual(set(sl.PUBLIC_FIELDS), set(sl.RECORD_FIELDS))

    def test_a_publishing_bot_without_a_usable_key_is_refused(self):
        import main

        short = base64.urlsafe_b64encode(bytes(16)).decode("ascii")
        for environment, given in ((None, None), (short, None), ("SYNTHETIC not a key!", None), (None, b"short")):
            if environment is None:
                os.environ.pop(sl.KEY_ENV, None)
            else:
                self.set_env(**{sl.KEY_ENV: environment})
            with self.assertRaises(sl.LedgerKeyError) as caught:
                main.GlassEarthBot(
                    run_mode="x",
                    spend_meter=self.make_meter(),
                    ledger_key=given,
                    publish_reports_to_metaculus=True,
                )
            self.assertNotIn(short, str(caught.exception))
        # A key in the environment is used when none is given.
        self.set_env(**{sl.KEY_ENV: FAKE_LEDGER_KEY_TEXT})
        main.GlassEarthBot(run_mode="x", spend_meter=self.make_meter(), publish_reports_to_metaculus=True)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.metaculus.gets, [])

    def test_a_bot_switched_to_publishing_later_stops_before_spending(self):
        bot = self.make_bot(publish=False, skip=False)
        bot.publish_reports_to_metaculus = True
        with self.assertRaises(sl.LedgerKeyError):
            asyncio.run(bot.forecast_questions([binary()]))
        self.assertEqual(self.models.calls, [])
        self.assertFalse(Path("forecasts/ledger.jsonl").exists())


class LiveRunKeyTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.set_env(OPENROUTER_API_KEY="sk-or-synthetic-sealed-key")
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "bot-testing-area"
        self.metaculus.posts_page = page

    def test_a_live_run_without_a_usable_key_stops_before_any_model_call(self):
        import main

        short = base64.urlsafe_b64encode(bytes(16)).decode("ascii")
        long = base64.urlsafe_b64encode(bytes(48)).decode("ascii")
        for value in (None, "", short, long, "SYNTHETIC not a key!"):
            if value is None:
                os.environ.pop(sl.KEY_ENV, None)
            else:
                self.set_env(**{sl.KEY_ENV: value})
            for argv in ([], ["--mode", "test_questions"], ["--mode", "market_pulse", "--max-questions", "1"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main.run(argv)
                message = str(caught.exception)
                self.assertTrue(message.startswith("Stopped before forecasting"), message)
                self.assertIn(sl.KEY_ENV, message)
                if value:
                    self.assertNotIn(value, message)
        # Nothing was fetched, checked, called or written.
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.metaculus.gets, [])
        self.assertEqual(self.metaculus.posts, [])
        self.assertEqual(self.openrouter_key.keys, [])
        self.assertFalse(Path("forecasts").exists())

    def test_a_dry_run_needs_no_key_and_keeps_its_full_lines(self):
        import main

        os.environ.pop(sl.KEY_ENV, None)
        with contextlib.redirect_stdout(io.StringIO()):
            reports = main.run(["--mode", "test_questions", "--dry-run", "--max-questions", "1"])
        self.assertFalse(isinstance(reports[0], BaseException))
        self.assertEqual(self.metaculus.posts, [])
        self.assertFalse(Path("forecasts/ledger.jsonl").exists())
        self.assertFalse(Path("forecasts/sealed_boxes.jsonl").exists())
        [line] = self.read_jsonl("forecasts/dry_run/ledger.jsonl")
        self.assertTrue(line["dry_run"])
        self.assertEqual(line["ledger_version"], 3)
        self.assertNotIn("sealed", line)
        self.assertNotIn("seal_sha256", line)
        # The full line, title and readable forecast included, as before.
        [question] = [
            q
            for post in load_fixture("synthetic_group_posts.json")["results"]
            for q in post["group_of_questions"]["questions"]
            if q["id"] == line["question_id"]
        ]
        self.assertEqual(line["title"], question["title"])
        self.assertEqual(line["group_option"], question["label"])
        self.assertTrue(line["final_forecast_readable"])
        self.assertTrue(line["final_forecast"]["continuous_cdf"])
        self.assertEqual(line["scheduled_close_time_utc"], question["scheduled_close_time"])


class RevealRunTests(OfflineTestCase):
    """Live runs of main.run that reveal, against the fake Metaculus."""

    def setUp(self):
        super().setUp()
        self.summary = Path(self._tmp.name) / "step_summary.md"
        self.set_env(
            OPENROUTER_API_KEY="sk-or-synthetic-reveal-key",
            GLASS_EARTH_LEDGER_KEY=FAKE_LEDGER_KEY_TEXT,
            GITHUB_STEP_SUMMARY=str(self.summary),
        )
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "bot-testing-area"
        self.metaculus.posts_page = page

    def set_status(self, question_id, status):
        for post in self.metaculus.posts_page["results"]:
            for question in post["group_of_questions"]["questions"]:
                if question["id"] == question_id:
                    question["status"] = status

    def live_run(self, **options):
        import main

        with contextlib.redirect_stdout(io.StringIO()) as console:
            reports = main.run(["--mode", "test_questions"], **options)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])
        return console.getvalue()

    def test_a_live_run_reveals_closed_questions_before_forecasting(self):
        first = self.live_run()
        self.assertIn("Ledger reveals: 0 forecast(s) revealed", first)
        sealed_first = self.read_jsonl("forecasts/ledger.jsonl")
        self.assertEqual(sorted(l["question_id"] for l in sealed_first), [90011, 90012, 90021, 90022])
        gets_before = len(self.metaculus.gets)

        # Metaculus now reports one sub-question closed.
        self.set_status(90011, "closed")
        second = self.live_run()
        self.assertIn("Ledger reveals: 1 forecast(s) revealed", second)
        lines = self.read_jsonl("forecasts/ledger.jsonl")
        reveals = [l for l in lines if l.get("revealed")]
        self.assertEqual(len(reveals), 1)
        [line] = [l for l in sealed_first if l["question_id"] == 90011]
        sl.verify_reveal(line, reveals[0])
        self.assertEqual(reveals[0]["record"]["question_id"], 90011)
        # Revealed before this run forecast anything: the reveal comes right
        # after the first run's sealed lines, before this run's.
        self.assertEqual(lines.index(reveals[0]), len(sealed_first))
        self.assertEqual(sorted(l["question_id"] for l in lines[len(sealed_first) + 1:]), [90012, 90021, 90022])
        # Asked once per post, by its own address, before this run fetched
        # the questions to forecast.
        urls = [url for url, _ in self.metaculus.gets[gets_before:]]
        self.assertEqual(
            urls[:2],
            ["https://www.metaculus.com/api/posts/9001/", "https://www.metaculus.com/api/posts/9002/"],
        )
        self.assertTrue(urls[2:])
        self.assertTrue(all(url.endswith("/api/posts/") for url in urls[2:]))

    def test_a_run_reports_a_lost_box_on_github(self):
        self.live_run()
        boxes = Path("forecasts/sealed_boxes.jsonl")
        kept = boxes.read_text(encoding="utf-8").splitlines()[1:]
        boxes.write_text("".join(line + "\n" for line in kept), encoding="utf-8", newline="\n")
        for question_id in (90011, 90012, 90021, 90022):
            self.set_status(question_id, "closed")
        self.set_env(GITHUB_ACTIONS="true")
        with self.assertLogs("main", level="WARNING") as logs:
            console = self.live_run()
        self.assertIn("Ledger reveals: 3 forecast(s) revealed", console)
        self.assertIn("1 sealed line(s) have no box in forecasts/sealed_boxes.jsonl", console)
        self.assertIn("::warning title=Sealed ledger::1 sealed line(s) have no box", console)
        self.assertTrue(any("have no box" in line for line in logs.output))
        self.assertIn("**Sealed ledger.** 1 sealed line(s) have no box", self.summary.read_text(encoding="utf-8"))

    def test_a_429_while_revealing_stops_the_reveals_not_the_run(self):
        self.live_run()
        for question_id in (90011, 90012):
            self.set_status(question_id, "closed")
        asked = []

        def rate_limited(post_id):
            asked.append(post_id)
            raise sl.RateLimited("Metaculus answered 429, too many requests")

        forecasts_before = len(self.metaculus.forecasts())
        with self.assertLogs("main", level="WARNING"):
            console = self.live_run(post_statuses=rate_limited)
        self.assertEqual(len(asked), 1)
        self.assertIn("Stopped asking Metaculus after an error (Metaculus answered 429", console)
        self.assertFalse([l for l in self.read_jsonl("forecasts/ledger.jsonl") if l.get("revealed")])
        # Forecasting went ahead: the two questions still open were forecast again (test area).
        self.assertEqual(len(self.metaculus.forecasts()) - forecasts_before, 2)

    def test_the_injected_check_decides_and_a_dry_run_asks_nothing(self):
        self.live_run()
        asked = []

        def says_closed(post_id):
            asked.append(post_id)
            return {90021: "resolved"} if post_id == 9002 else {}

        import main

        with contextlib.redirect_stdout(io.StringIO()):
            main.run(["--mode", "test_questions", "--dry-run"], post_statuses=says_closed)
        self.assertEqual(asked, [])
        console = self.live_run(post_statuses=says_closed)
        self.assertEqual(sorted(asked), [9001, 9002])
        self.assertIn("Ledger reveals: 1 forecast(s) revealed", console)
        [reveal] = [l for l in self.read_jsonl("forecasts/ledger.jsonl") if l.get("revealed")]
        self.assertEqual(reveal["record"]["question_id"], 90021)


class StatusCheckTests(OfflineTestCase):
    """sealed_ledger.metaculus_post_statuses against the fake Metaculus."""

    def test_the_check_reads_one_post_with_the_bot_token(self):
        self.metaculus.posts_page = load_fixture("synthetic_group_posts.json")
        self.assertEqual(sl.metaculus_post_statuses(9001), {90011: "open", 90012: "open", 90013: "closed"})
        self.assertEqual(self.metaculus.gets[-1][0], "https://www.metaculus.com/api/posts/9001/")
        # A post Metaculus does not find reports no status.
        self.assertEqual(sl.metaculus_post_statuses(424242), {})

    def test_a_429_or_a_failed_answer_raises_without_a_retry(self):
        import requests

        calls = []

        def answering(status=200, body=b"{}", error=None):
            def get(url, **kwargs):
                calls.append(url)
                if error is not None:
                    raise error
                response = requests.Response()
                response.status_code = status
                response._content = body
                return response

            return get

        for get, expected in (
            (answering(429), sl.RateLimited),
            (answering(503), sl.StatusCheckError),
            (answering(403), sl.StatusCheckError),
            (answering(200, b"<html>SYNTHETIC challenge page</html>"), sl.StatusCheckError),
            (answering(200, b"[1, 2]"), sl.StatusCheckError),
            (answering(error=requests.ConnectionError("SYNTHETIC")), sl.StatusCheckError),
        ):
            calls.clear()
            requests.get = get  # tearDown puts the fake Metaculus back
            with self.assertRaises(expected) as caught:
                sl.metaculus_post_statuses(9001)
            self.assertEqual(len(calls), 1)
            self.assertNotIn(FAKE_TOKEN, str(caught.exception))

    def test_without_a_token_nothing_is_asked(self):
        os.environ.pop("METACULUS_TOKEN")
        with self.assertRaises(sl.StatusCheckError):
            sl.metaculus_post_statuses(9001)
        self.assertEqual(self.metaculus.gets, [])


class ReaderTests(OfflineTestCase):
    def write_ledger(self):
        """A SYNTHETIC ledger: a full line from before sealing, sealed lines, and reveals."""
        now = datetime.now(timezone.utc)

        def at(hours_ago):
            return (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

        def sealed(question_id, recorded):
            entry = ledger_entry(question_id=question_id, final_forecast={"probability_yes": 0.4},
                                 context={"post_id": question_id // 10, "url": None}, run_mode="x")
            entry["recorded_at_utc"] = recorded
            return sl.seal_entry(entry, FAKE_LEDGER_KEY)

        older, newer, other = sealed(90011, at(200)), sealed(90011, at(30)), sealed(90021, at(50))
        lines = [
            {"question_id": 90012, "recorded_at_utc": at(300), "final_forecast": {}},
            older.line,
            "not json",
            newer.line,
            other.line,
            # A reveal comes later than its sealed line and repeats its record.
            sl.reveal_line_for(older.line, older.record, now),
            sl.reveal_line_for(other.line, other.record, now),
        ]
        Path("forecasts").mkdir(exist_ok=True)
        with open("forecasts/ledger.jsonl", "w", encoding="utf-8", newline="\n") as file:
            for line in lines:
                file.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
        return at

    def test_readers_count_each_sealed_forecast_once_and_never_a_reveal(self):
        at = self.write_ledger()
        ledger = ForecastLedger(Path("forecasts/ledger.jsonl"))
        with self.assertLogs("sealed_ledger", level="WARNING"):
            lines = ledger.forecast_lines()
        self.assertEqual(len(lines), 4)
        self.assertFalse([line for line in lines if line.get("revealed")])
        with self.assertLogs("sealed_ledger", level="WARNING"):
            self.assertEqual(ledger.question_ids(), {90011, 90012, 90021})
        with self.assertLogs("sealed_ledger", level="WARNING"):
            times = ledger.last_recorded_at()
        # The latest forecast's time, never a reveal's.
        self.assertEqual(times[90011].strftime("%Y-%m-%dT%H:%M:%SZ"), at(30))
        self.assertEqual(times[90021].strftime("%Y-%m-%dT%H:%M:%SZ"), at(50))
        self.assertEqual(times[90012].strftime("%Y-%m-%dT%H:%M:%SZ"), at(300))

    def test_the_bot_skips_and_times_questions_from_sealed_lines(self):
        self.write_ledger()
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "market-pulse-26q4"
        self.metaculus.posts_page = page
        bot = self.make_bot(publish=True, skip=True)
        with self.assertLogs("sealed_ledger", level="WARNING"):
            asyncio.run(bot.forecast_on_tournament("market-pulse-26q4", return_exceptions=True))
        forecast = sorted(f["question"] for f in self.metaculus.forecasts())
        # Market Pulse forecasts again after 7 days. 90011's latest sealed
        # line is 30 hours old and 90021's 50 hours (its reveal is not a new
        # forecast), so neither is due; 90012's full line is 300 hours old,
        # so it is due; 90022 has no line and gets its first forecast.
        self.assertEqual(forecast, [90012, 90022])


if __name__ == "__main__":
    unittest.main()
