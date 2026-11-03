"""
The learning loop (scripts/learning_loop.py), offline: SYNTHETIC ledger lines
(full, sealed and revealed), SYNTHETIC Metaculus answers from a fake API and
SYNTHETIC model answers. Nothing here touches the network, reads .env or
spends money.
"""

import contextlib
import importlib.util
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from support import FAKE_LEDGER_KEY, FAKE_TOKEN, REPO, OfflineTestCase, prompt_text  # noqa: I001  (blocks the network first)

import requests
from litellm import ModelResponse
from litellm.types.utils import Choices, Message, Usage

import forecasting_tools.ai_models.general_llm as general_llm

import sealed_ledger
from ledger import ledger_entry

SPEC = importlib.util.spec_from_file_location("learning_loop", REPO / "scripts" / "learning_loop.py")
learning_loop = importlib.util.module_from_spec(SPEC)
sys.modules["learning_loop"] = learning_loop
SPEC.loader.exec_module(learning_loop)

API = "https://www.metaculus.com/api"
EM_DASH = chr(0x2014)
CLOSE = "2026-11-01T10:00:00Z"
RESOLVED = "2026-11-02T10:00:00Z"
# 21:00 UTC is 07:00 the next day in Brisbane.
MORNING = datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)  # 4 November in Brisbane
NEXT_MORNING = datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)  # 5 November
SYNTHETIC_CAUSE = "SYNTHETIC: leaned on the status quo too hard."
BOX = "SYNTHETIC-SEALED-BOX-NEVER-OPENED"


def _response(payload, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode("utf-8")
    return response


def ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class FakeApi:
    """SYNTHETIC Metaculus answers for the learning loop's GETs, recording each one."""

    def __init__(self):
        self.posts = {}
        self.status = {}  # path -> HTTP status to answer with instead
        self.comments = {}  # post id -> SYNTHETIC comments
        self.user = {"id": 4242, "username": "SYNTHETIC-bot"}
        self.gets = []

    def add(self, *posts):
        for post in posts:
            self.posts[post["id"]] = post

    def get(self, url, params=None, **kwargs):
        headers = kwargs.get("headers") or {}
        assert headers.get("Authorization") == f"Token {FAKE_TOKEN}", "every request carries the token"
        assert url.startswith(API + "/"), url
        path = url[len(API):]
        self.gets.append((path, params))
        if path in self.status:
            return _response({"detail": "SYNTHETIC refusal"}, self.status[path])
        match = re.fullmatch(r"/posts/(\d+)/", path)
        if match:
            post = self.posts.get(int(match.group(1)))
            return _response(post) if post is not None else _response({"detail": "Not found."}, 404)
        if path == "/users/me":
            return _response(self.user)
        if path == "/comments/":
            return _response({"results": self.comments.get(params["post"], [])})
        raise AssertionError(f"Unexpected GET {url}")

    def paths(self):
        return [path for path, _ in self.gets]


def question(question_id, qtype="binary", *, status="resolved", resolution=None, title=None,
             close=CLOSE, score_data=None, history=None, **extra):
    """A SYNTHETIC question as the post detail endpoint returns it."""
    body = {
        "id": question_id,
        "title": title or f"SYNTHETIC question {question_id}",
        "type": qtype,
        "status": status,
        "resolution": resolution if status == "resolved" else None,
        "resolution_criteria": f"SYNTHETIC criteria for {question_id}.",
        "scheduled_close_time": close,
        "actual_close_time": close if status in ("closed", "resolved") else None,
        "actual_resolve_time": RESOLVED if status == "resolved" else None,
        "my_forecasts": {"history": history or [], "latest": None, "score_data": score_data or {}},
    }
    body.update(extra)
    return body


def post(post_id, *questions, group=False, tournament="fall-futureeval-2026"):
    projects = {"tournament": [{"id": 77001, "slug": tournament}]}
    if group:
        return {
            "id": post_id,
            "title": f"SYNTHETIC group {post_id}",
            "projects": projects,
            "group_of_questions": {
                "resolution_criteria": "SYNTHETIC group criteria.",
                "questions": list(questions),
            },
        }
    return {"id": post_id, "title": questions[0]["title"], "projects": projects, "question": questions[0]}


def line(question_id, post_id, forecast, *, recorded="2026-10-20T00:00:00Z", qtype="binary",
         run_mode="tournament"):
    """A SYNTHETIC full ledger line (the format before sealing), with the fields the loop reads."""
    return {
        "ledger_version": 2,
        "recorded_at_utc": recorded,
        "question_id": question_id,
        "post_id": post_id,
        "url": f"https://www.metaculus.com/questions/{post_id}",
        "type": qtype,
        "title": f"SYNTHETIC ledger title {question_id}",
        "run_mode": run_mode,
        "final_forecast": forecast,
        "final_forecast_readable": f"SYNTHETIC readable forecast {question_id}",
    }


def sealed(question_id, post_id, seal, *, recorded="2026-10-20T00:00:00Z"):
    """A SYNTHETIC sealed ledger line: IDs, link, time and cost in the clear, the forecast in a box."""
    return {
        "ledger_version": 3,
        "recorded_at_utc": recorded,
        "question_id": question_id,
        "post_id": post_id,
        "url": f"https://www.metaculus.com/questions/{post_id}",
        "forecaster_model": "anthropic/claude-opus-5.5",
        "cost_usd": "0.1",
        "sealed": True,
        "seal_sha256": seal,
        "box": BOX,
    }


def reveal(seal, question_id, post_id, forecast, *, recorded="2026-10-20T00:00:00Z", qtype="binary"):
    """A SYNTHETIC reveal line: the full entry, without title, label or readable forecast."""
    return {
        "revealed": True,
        "seal_sha256": seal,
        "revealed_at_utc": "2026-11-01T12:00:00Z",
        "record": {
            "ledger_version": 2,
            "recorded_at_utc": recorded,
            "question_id": question_id,
            "post_id": post_id,
            "url": f"https://www.metaculus.com/questions/{post_id}",
            "type": qtype,
            "run_mode": "tournament",
            "final_forecast": forecast,
        },
    }


def sealed_then_revealed(question_id, post_id, forecast, *, qtype, options=None, recorded="2026-10-20T00:00:00Z"):
    """
    A SYNTHETIC forecast sealed and then revealed by the bot's own code
    (sealed_ledger.py), so both lines are in the format a live run writes. In
    the record, each multiple choice option is its index in the question's
    option order, as text, and option_count says how many options there are.
    """
    entry = ledger_entry(
        question_id=question_id,
        final_forecast=forecast,
        context={
            "post_id": post_id,
            "url": f"https://www.metaculus.com/questions/{post_id}/",
            "type": qtype,
            "option_labels": options,
        },
        run_mode="tournament",
    )
    entry["recorded_at_utc"] = recorded
    sealed_forecast = sealed_ledger.seal_entry(entry, FAKE_LEDGER_KEY)
    revealed = sealed_ledger.reveal_line_for(
        sealed_forecast.line, sealed_forecast.record, datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    )
    return sealed_forecast.line, revealed


def binary(p):
    return {"probability_yes": p}


def linear_cdf(points=201, below=0.05, above=0.05):
    """A SYNTHETIC distribution: `below` under the range, `above` over it, even in between."""
    return [below + (1 - below - above) * i / (points - 1) for i in range(points)]


def review_response(model, answer):
    response = ModelResponse(
        id="synthetic",
        model=model,
        choices=[Choices(finish_reason="stop", index=0, message=Message(content=answer, role="assistant"))],
        usage=Usage(prompt_tokens=900, completion_tokens=30, total_tokens=930),
    )
    # SYNTHETIC reported cost: 900 tokens in and 30 out at the helper model's
    # list price (US$0.10 and US$0.50 per million).
    response._hidden_params = {"additional_headers": {"llm_provider-x-litellm-response-cost": 0.000105}}
    return response


class LearningLoopCase(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.api = FakeApi()
        requests.get = self.api.get
        self.pauses = []
        self.review_prompts = []
        self.review_answer = SYNTHETIC_CAUSE
        self.review_error = None
        synthetic = self.models.acompletion

        async def acompletion(messages, **kwargs):
            prompt = prompt_text(messages)
            if learning_loop.REVIEW_MARKER not in prompt:
                return await synthetic(messages, **kwargs)
            self.review_prompts.append(prompt)
            if self.review_error is not None:
                raise self.review_error
            return review_response(kwargs["model"], self.review_answer)

        # OfflineTestCase.tearDown puts the real one back.
        general_llm.acompletion = acompletion

    def write_ledger(self, *lines):
        Path("forecasts").mkdir(exist_ok=True)
        with open("forecasts/ledger.jsonl", "a", encoding="utf-8", newline="\n") as file:
            for entry in lines:
                file.write(json.dumps(entry) + "\n")

    def run_loop(self, now=MORNING):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = learning_loop.run(now=now, pause=self.pauses.append)
        self.printed = printed.getvalue()
        return code

    def with_model(self):
        self.set_env(OPENROUTER_API_KEY="sk-or-synthetic-learning-key")

    def state(self):
        return json.loads(Path("forecasts/learning_state.json").read_text(encoding="utf-8"))

    def lessons(self):
        return Path("LESSONS.md").read_text(encoding="utf-8")

    def written(self):
        return self.lessons() + Path("forecasts/learning_state.json").read_text(encoding="utf-8")


class ResolvedOnlyTests(LearningLoopCase):
    def test_open_and_closed_questions_never_appear_in_lessons_or_state(self):
        self.with_model()
        self.write_ledger(
            line(710001, 61001, binary(0.123456)),
            line(710002, 61002, binary(0.234567)),
            line(710003, 61003, binary(0.8)),
        )
        self.api.add(
            post(61001, question(710001, status="open")),
            post(61002, question(710002, status="closed")),
            post(61003, question(710003, resolution="no")),
        )
        self.assertEqual(self.run_loop(), 0)
        written = self.written()
        for unwritten in ("710001", "61001", "0.123456", "12.3%", "710002", "61002", "0.234567", "23.5%"):
            self.assertNotIn(unwritten, written)
        self.assertIn("[Question 710003](https://www.metaculus.com/questions/61003/)", written)
        self.assertEqual(set(self.state()["questions"]), {"710003"})
        self.assertEqual(self.state()["left_out"], {})
        # The review sees only the resolved question.
        self.assertEqual(len(self.review_prompts), 1)
        self.assertNotIn("710001", self.review_prompts[0])
        self.assertNotIn("710002", self.review_prompts[0])

    def test_open_sub_question_of_a_group_is_not_written(self):
        self.write_ledger(line(740001, 64001, binary(0.7)), line(740002, 64001, binary(0.345678)))
        self.api.add(
            post(
                64001,
                question(740001, resolution="yes", label="SYNTHETIC resolved label"),
                question(740002, status="open", label="SYNTHETIC open label"),
                group=True,
            )
        )
        self.run_loop()
        written = self.written()
        self.assertEqual(set(self.state()["questions"]), {"740001"})
        self.assertNotIn("740002", written)
        self.assertNotIn("0.345678", written)
        # The group's criteria stand in for the sub-question's own when it has none.
        self.assertEqual(
            learning_loop.resolution_criteria(self.api.posts[64001], {"id": 1}),
            "SYNTHETIC group criteria.",
        )

    def test_annulled_and_ambiguous_questions_are_left_out(self):
        self.write_ledger(
            line(720001, 62001, binary(0.3)),
            line(720002, 62002, binary(0.6)),
            line(720003, 62003, binary(0.9)),
        )
        self.api.add(
            post(62001, question(720001, resolution="annulled")),
            post(62002, question(720002, resolution="ambiguous")),
            post(62003, question(720003, resolution="no")),
        )
        self.run_loop()
        state = self.state()
        self.assertEqual(set(state["questions"]), {"720003"})
        self.assertEqual(
            state["left_out"],
            {
                "720001": {"reason": "annulled", "checked_on": "2026-11-04"},
                "720002": {"reason": "ambiguous", "checked_on": "2026-11-04"},
            },
        )
        lessons = self.lessons()
        self.assertIn("Left out: 1 ambiguous, 1 annulled.", lessons)
        self.assertIn("| Yes/no | 1 |", lessons)
        self.assertNotIn("Question 720001", lessons)
        # Never asked about again.
        self.api.gets.clear()
        self.run_loop(NEXT_MORNING)
        self.assertEqual(self.api.paths(), [])

    def test_bot_testing_area_questions_are_left_out(self):
        self.write_ledger(
            line(760001, 66001, binary(0.4), run_mode="test_questions"),
            # A sealed line does not say where it was forecast; its post does.
            sealed(760002, 66002, "a" * 64),
        )
        self.api.add(
            post(66001, question(760001, resolution="yes")),
            post(66002, question(760002, resolution="yes"), tournament="bot-testing-area"),
        )
        self.run_loop()
        self.assertEqual(self.api.paths(), ["/posts/66002/"])
        self.assertEqual(self.state()["questions"], {})
        self.assertEqual(self.state()["left_out"]["760002"]["reason"], "in the unscored bot testing area")
        self.assertIn("Nothing new resolved.", self.lessons())

    def test_last_forecast_before_close_is_the_one_scored(self):
        self.write_ledger(
            line(770001, 67001, binary(0.9), recorded="2026-10-10T00:00:00Z"),
            line(770001, 67001, binary(0.6), recorded="2026-10-25T00:00:00Z"),
            # SYNTHETIC: recorded a second after the close, so it must not count.
            line(770001, 67001, binary(0.1), recorded="2026-11-01T10:00:01Z"),
        )
        self.api.add(post(67001, question(770001, resolution="no")))
        self.run_loop()
        record = self.state()["questions"]["770001"]
        self.assertEqual(record["probability_yes"], 0.6)
        self.assertEqual(record["forecast_recorded_at_utc"], "2026-10-25T00:00:00Z")
        self.assertEqual(record["forecast_source"], "ledger")
        self.assertAlmostEqual(record["brier"], 0.36)
        self.assertAlmostEqual(record["log_score"], math.log(0.4))

    def test_no_forecast_before_close_is_left_out(self):
        self.write_ledger(line(780001, 68001, binary(0.5), recorded="2026-11-01T11:00:00Z"))
        self.api.add(post(68001, question(780001, resolution="yes")))
        self.run_loop()
        self.assertEqual(self.state()["questions"], {})
        self.assertEqual(
            self.state()["left_out"]["780001"]["reason"], "no forecast of ours stood at the close"
        )

    def test_metaculus_scores_are_observed_or_unknown(self):
        self.write_ledger(line(790001, 69001, binary(0.7)), line(790002, 69002, binary(0.7)))
        self.api.add(
            post(69001, question(790001, resolution="yes", score_data={"baseline_score": 48.5, "peer_score": -3.25})),
            post(69002, question(790002, resolution="yes")),
        )
        self.run_loop()
        questions = self.state()["questions"]
        self.assertEqual(questions["790001"]["metaculus_baseline_score"], 48.5)
        self.assertEqual(questions["790001"]["metaculus_peer_score"], -3.25)
        self.assertEqual(questions["790002"]["metaculus_baseline_score"], "Unknown")
        self.assertEqual(questions["790002"]["metaculus_peer_score"], "Unknown")
        self.assertIn("| 48.5 | -3.2 |", self.lessons())


class SealedLedgerTests(LearningLoopCase):
    def test_a_sealed_forecast_is_read_from_its_reveal_line_and_the_box_is_never_opened(self):
        seal = "b" * 64
        self.write_ledger(sealed(860001, 76001, seal), reveal(seal, 860001, 76001, binary(0.2)))
        # SYNTHETIC: Metaculus reports a different forecast, so the source shows.
        history = [{"start_time": ts("2026-10-20T00:00:00Z"), "end_time": None, "forecast_values": [0.1, 0.9]}]
        self.api.add(post(76001, question(860001, resolution="yes", history=history)))
        self.run_loop()
        record = self.state()["questions"]["860001"]
        self.assertEqual(record["probability_yes"], 0.2)
        self.assertEqual(record["forecast_source"], "ledger")
        self.assertNotIn(BOX, self.written())
        # Nothing that could open the box is even imported.
        source = (REPO / "scripts" / "learning_loop.py").read_text(encoding="utf-8")
        imports = [x for x in source.splitlines() if x.startswith(("import ", "from "))]
        for crypto in ("nacl", "cryptography", "Crypto"):
            self.assertFalse(any(crypto in x for x in imports), crypto)

    def test_a_forecast_still_sealed_is_taken_from_metaculus(self):
        self.write_ledger(sealed(860002, 76002, "c" * 64))
        history = [
            {"start_time": ts("2026-10-15T00:00:00Z"), "end_time": ts("2026-10-20T00:00:00Z"), "forecast_values": [0.2, 0.8]},
            {"start_time": ts("2026-10-20T00:00:00Z"), "end_time": None, "forecast_values": [0.7, 0.3]},
        ]
        self.api.add(post(76002, question(860002, resolution="no", history=history)))
        self.run_loop()
        record = self.state()["questions"]["860002"]
        self.assertEqual(record["probability_yes"], 0.3)
        self.assertEqual(record["forecast_source"], "metaculus")
        self.assertEqual(record["forecast_recorded_at_utc"], "2026-10-20T00:00:00Z")
        self.assertAlmostEqual(record["log_score"], math.log(0.7))

    def test_an_older_reveal_never_stands_in_for_a_later_sealed_forecast(self):
        self.write_ledger(
            sealed(860003, 76003, "d" * 64, recorded="2026-10-10T00:00:00Z"),
            reveal("d" * 64, 860003, 76003, binary(0.9), recorded="2026-10-10T00:00:00Z"),
            sealed(860003, 76003, "e" * 64, recorded="2026-10-25T00:00:00Z"),
        )
        history = [
            {"start_time": ts("2026-10-10T00:00:00Z"), "end_time": ts("2026-10-25T00:00:00Z"), "forecast_values": [0.1, 0.9]},
            {"start_time": ts("2026-10-25T00:00:00Z"), "end_time": None, "forecast_values": [0.6, 0.4]},
        ]
        self.api.add(post(76003, question(860003, resolution="yes", history=history)))
        self.run_loop()
        record = self.state()["questions"]["860003"]
        self.assertEqual(record["probability_yes"], 0.4)
        self.assertEqual(record["forecast_source"], "metaculus")

    def test_market_pulse_forecasts_revealed_after_the_close_use_the_last_before_it(self):
        self.write_ledger(
            sealed(860004, 76004, "f" * 64, recorded="2026-10-10T00:00:00Z"),
            sealed(860004, 76004, "0" * 64, recorded="2026-10-30T00:00:00Z"),
            reveal("f" * 64, 860004, 76004, binary(0.35), recorded="2026-10-10T00:00:00Z"),
            reveal("0" * 64, 860004, 76004, binary(0.55), recorded="2026-10-30T00:00:00Z"),
        )
        self.api.add(post(76004, question(860004, resolution="yes")))
        self.run_loop()
        self.assertEqual(self.state()["questions"]["860004"]["probability_yes"], 0.55)

    def test_a_sealed_forecast_neither_revealed_nor_reported_waits_for_the_next_morning(self):
        self.write_ledger(sealed(860005, 76005, "1" * 64))
        self.api.add(post(76005, question(860005, resolution="yes")))
        self.run_loop()
        self.assertEqual(self.state()["questions"], {})
        self.assertEqual(self.state()["left_out"], {})
        self.write_ledger(reveal("1" * 64, 860005, 76005, binary(0.25)))
        self.run_loop(NEXT_MORNING)
        self.assertEqual(self.state()["questions"]["860005"]["probability_yes"], 0.25)
        self.assertEqual(self.state()["mornings"]["2026-11-05"]["questions"], [860005])

    def test_a_sealed_line_without_a_readable_time_waits_rather_than_being_left_out(self):
        entry = sealed(860006, 76006, "3" * 64)
        entry["sealed_at"] = entry.pop("recorded_at_utc")  # SYNTHETIC: a time field the loop does not read
        self.write_ledger(entry)
        self.api.add(post(76006, question(860006, resolution="yes")))
        self.run_loop()
        self.assertEqual(self.state()["questions"], {})
        self.assertEqual(self.state()["left_out"], {})


class MultipleChoiceKeyTests(LearningLoopCase):
    """
    A sealed or revealed record gives each multiple choice option by its index
    in the question's option order, with option_count; an older full line
    gives each by its label. Both are scored, from the question's own options.
    """

    OPTIONS = ["SYNTHETIC OPTION AMBER", "SYNTHETIC OPTION BIRCH", "SYNTHETIC OPTION CEDAR"]
    FORECAST = {"probability_yes_per_category": dict(zip(OPTIONS, (0.5, 0.3, 0.2)))}

    def test_a_revealed_forecast_keyed_by_option_index_is_scored(self):
        sealed_line, reveal_line = sealed_then_revealed(
            880001, 78001, self.FORECAST, qtype="multiple_choice", options=self.OPTIONS
        )
        # The bot's own code wrote the record by option index, with option_count.
        self.assertEqual(
            reveal_line["record"]["final_forecast"],
            {"probability_yes_per_category": {"0": 0.5, "1": 0.3, "2": 0.2}},
        )
        self.assertEqual(reveal_line["record"]["option_count"], 3)
        self.write_ledger(sealed_line, reveal_line)
        # SYNTHETIC: Metaculus reports a different forecast, so the source shows.
        history = [{"start_time": ts("2026-10-20T00:00:00Z"), "end_time": None, "forecast_values": [0.1, 0.1, 0.8]}]
        self.api.add(post(78001, question(880001, "multiple_choice", resolution="SYNTHETIC OPTION BIRCH",
                                          options=self.OPTIONS, history=history)))
        self.run_loop()
        record = self.state()["questions"]["880001"]
        self.assertEqual(record["forecast_source"], "ledger")
        self.assertEqual(record["outcome_text"], "option 2 of 3")
        self.assertEqual(record["probability_on_outcome"], 0.3)
        self.assertEqual(record["forecast_text"], "30% on the outcome")
        self.assertAlmostEqual(record["log_score"], math.log(0.3))
        self.assertAlmostEqual(record["above_uniform"], math.log(0.3 * 3))
        self.assertIsNone(record["score_note"])
        self.assertIn("| Multiple choice | 1 | n/a | -1.204 | -0.105 |", self.lessons())
        for label in self.OPTIONS:
            self.assertNotIn(label, self.written())

    def test_an_older_full_line_keyed_by_option_label_is_still_scored(self):
        self.write_ledger(line(880002, 78002, self.FORECAST, qtype="multiple_choice"))
        self.api.add(post(78002, question(880002, "multiple_choice", resolution="SYNTHETIC OPTION CEDAR",
                                          options=self.OPTIONS)))
        self.run_loop()
        record = self.state()["questions"]["880002"]
        self.assertEqual(record["forecast_source"], "ledger")
        self.assertEqual(record["outcome_text"], "option 3 of 3")
        self.assertEqual(record["probability_on_outcome"], 0.2)
        self.assertEqual(record["forecast_text"], "20% on the outcome")
        self.assertAlmostEqual(record["log_score"], math.log(0.2))
        self.assertAlmostEqual(record["above_uniform"], math.log(0.2 * 3))
        self.assertIsNone(record["score_note"])

    def test_an_outcome_missing_from_the_options_scores_unknown_never_zero(self):
        self.write_ledger(
            *sealed_then_revealed(880003, 78003, self.FORECAST, qtype="multiple_choice", options=self.OPTIONS)
        )
        # SYNTHETIC: an outcome that is not one of the question's options.
        self.api.add(post(78003, question(880003, "multiple_choice", resolution="SYNTHETIC OPTION NOT LISTED",
                                          options=self.OPTIONS)))
        self.run_loop()
        state = self.state()
        record = state["questions"]["880003"]
        self.assertEqual(record["outcome_text"], "an option not in the question's list")
        for name in ("probability_on_outcome", "forecast_text", "log_score", "above_uniform"):
            self.assertEqual(record[name], "Unknown", name)
        self.assertIsNone(record["brier"])
        self.assertEqual(record["score_note"], "the outcome is not one of the question's options")
        # Left out of the means and of the misses: Unknown is never counted as zero.
        self.assertEqual(state["mornings"]["2026-11-04"]["worst"], [])
        lessons = self.lessons()
        self.assertIn("| Multiple choice | 0 | n/a | n/a | n/a |", lessons)
        self.assertIn("Also reported, with an Unknown score and so left out of these means: 1.", lessons)
        self.assertIn(
            "The scores of [Question 880003](https://www.metaculus.com/questions/78003/) are Unknown: "
            "the outcome is not one of the question's options.",
            lessons,
        )
        self.assertNotIn("SYNTHETIC OPTION NOT LISTED", self.written())

    def test_option_count_not_the_keys_says_a_forecast_is_keyed_by_index(self):
        # SYNTHETIC labels that look like indices: "2" is the option at index 1.
        digits = question(1, "multiple_choice", options=["1", "2", "3"])
        by_index = {"probability_yes_per_category": {"0": 0.6, "1": 0.3, "2": 0.1}}
        scores = learning_loop.scores_for("multiple_choice", "2", by_index, digits, option_count=3)
        self.assertEqual(scores["probability_on_outcome"], 0.3)
        by_label = {"probability_yes_per_category": {"1": 0.6, "2": 0.3, "3": 0.1}}
        self.assertEqual(learning_loop.scores_for("multiple_choice", "2", by_label, digits)["probability_on_outcome"], 0.3)
        # An index means nothing without the options the forecast was made over.
        for changed in (question(2, "multiple_choice", options=["1", "2", "3", "4"]), question(3, "multiple_choice")):
            with self.assertRaises(learning_loop.ScoreUnknown):
                learning_loop.scores_for("multiple_choice", "2", by_index, changed, option_count=3)


class NoQuestionTextTests(LearningLoopCase):
    """Metaculus's terms of use: no title, question text or option label is written."""

    def test_titles_labels_options_and_criteria_are_never_written(self):
        self.with_model()
        options = ["SYNTHETIC OPTION RED", "SYNTHETIC OPTION BLUE", "SYNTHETIC OPTION GREEN"]
        history = [{"start_time": ts("2026-10-20T00:00:00Z"), "end_time": None, "forecast_values": [0.6, 0.1, 0.3]}]
        self.write_ledger(
            sealed(870001, 77001, "2" * 64),
            line(870002, 77002, binary(0.1)),
            line(870003, 77003, binary(0.2)),
        )
        self.api.add(
            post(77001, question(870001, "multiple_choice", resolution="SYNTHETIC OPTION BLUE",
                                 options=options, title="SYNTHETIC SECRET MC TITLE", history=history)),
            post(77002, question(870002, resolution="yes", title="SYNTHETIC SECRET BINARY TITLE")),
            post(
                77003,
                question(870003, resolution="yes", label="SYNTHETIC SECRET LABEL",
                         title="SYNTHETIC SECRET GROUP QUESTION"),
                group=True,
            ),
        )
        self.run_loop()
        written = self.written()
        for text in options + ["SECRET", "SYNTHETIC criteria", "SYNTHETIC group", "ledger title", "readable forecast"]:
            self.assertNotIn(text, written)
            self.assertNotIn(text, self.printed)  # the run log is public too
        record = self.state()["questions"]["870001"]
        self.assertEqual(record["outcome_text"], "option 2 of 3")
        self.assertEqual(record["forecast_text"], "10% on the outcome")
        self.assertIsNone(record["outcome"])
        self.assertAlmostEqual(record["log_score"], math.log(0.1))
        self.assertIn("| [Question 870001](https://www.metaculus.com/questions/77001/) | Multiple choice | option 2 of 3 |", written)
        # The reviewing model may see the question; the files never do.
        prompts = "\n".join(self.review_prompts)
        self.assertIn("SYNTHETIC SECRET MC TITLE", prompts)
        self.assertIn("SYNTHETIC OPTION BLUE: 10%", prompts)

    def test_a_cause_that_quotes_the_question_is_withheld(self):
        self.with_model()
        self.write_ledger(line(870004, 77004, binary(0.1)))
        self.api.add(post(77004, question(870004, resolution="yes",
                                          title="SYNTHETIC: will the made up index close above its record?")))
        self.review_answer = "The bot thought the made up index close above its record was unlikely."
        self.run_loop()
        self.assertEqual(self.state()["questions"]["870004"]["cause"], "Unknown (not reviewed)")
        self.assertNotIn("made up index", self.written())
        self.assertIn("withheld because they repeated the question's own words", self.lessons())

    def test_the_quote_check(self):
        quotes = learning_loop.quotes_the_question
        title = ["Will the SYNTHETIC index close above 5,000 by March?"]
        self.assertTrue(quotes("It doubted the synthetic index close above 5,000 badly.", title, []))
        self.assertFalse(quotes("It underrated a late rally in the index.", title, []))
        self.assertTrue(quotes("Too little weight on Northwind Traders.", [], ["Northwind Traders"]))
        self.assertFalse(quotes("Too little weight on the rival firm.", [], ["Northwind Traders", "No"]))
        # A short title counts whole, from 3 words.
        self.assertTrue(quotes("Missed that gold beats silver here.", ["Gold beats silver?"], []))


class ScoreTests(LearningLoopCase):
    def test_binary_brier_and_log_score(self):
        yes = learning_loop.binary_scores(0.2, True)
        self.assertAlmostEqual(yes["brier"], 0.64)
        self.assertAlmostEqual(yes["log_score"], math.log(0.2))
        self.assertAlmostEqual(yes["above_uniform"], math.log(0.2 / 0.5))
        no = learning_loop.binary_scores(0.2, False)
        self.assertAlmostEqual(no["brier"], 0.04)
        self.assertAlmostEqual(no["log_score"], math.log(0.8))
        self.assertAlmostEqual(no["probability_on_outcome"], 0.8)
        # A forecast that ruled the outcome out is floored, never infinite.
        ruled_out = learning_loop.binary_scores(1.0, False)
        self.assertAlmostEqual(ruled_out["brier"], 1.0)
        self.assertAlmostEqual(ruled_out["log_score"], math.log(1e-6))

    def test_multiple_choice_log_score_of_the_outcome(self):
        scores = learning_loop.multiple_choice_scores({"A": 0.5, "B": 0.3, "C": 0.2}, "B")
        self.assertAlmostEqual(scores["log_score"], math.log(0.3))
        self.assertAlmostEqual(scores["above_uniform"], math.log(0.3 * 3))
        self.assertIsNone(scores["brier"])
        with self.assertRaises(learning_loop.ScoreUnknown):
            learning_loop.multiple_choice_scores({"A": 0.5, "B": 0.5}, "Z")

    def test_bins_and_the_mass_in_the_outcomes_bin(self):
        cdf = linear_cdf()  # 0.0045 in each of 200 bins, 5% beyond each bound
        inside = learning_loop.continuous_scores(cdf, 0.5, open_lower=True, open_upper=True)
        self.assertAlmostEqual(inside["log_score"], math.log(0.0045))
        self.assertAlmostEqual(inside["above_uniform"], 0.0, places=9)
        below = learning_loop.continuous_scores(cdf, -math.inf, open_lower=True, open_upper=True)
        self.assertAlmostEqual(below["log_score"], math.log(0.05))
        self.assertTrue(below["beyond_bounds"])
        above = learning_loop.continuous_scores(cdf, math.inf, open_lower=True, open_upper=True)
        self.assertAlmostEqual(above["log_score"], math.log(0.05))
        # Half the mass in bin 101, which holds locations from 0.5 to 0.505.
        peaked = [0.0] * 101 + [0.5] + [1.0] * 99
        sharp = learning_loop.continuous_scores(peaked, 0.502, open_lower=False, open_upper=False)
        self.assertAlmostEqual(sharp["log_score"], math.log(0.5))
        self.assertAlmostEqual(sharp["above_uniform"], math.log(0.5 / (1 / 200)))
        self.assertEqual(learning_loop.bucket_index(0.0, 200), 1)
        self.assertEqual(learning_loop.bucket_index(1.0, 200), 200)
        self.assertEqual(learning_loop.bucket_index(0.005, 200), 1)  # an edge counts in the lower bin
        self.assertEqual(learning_loop.bucket_index(0.0051, 200), 2)
        self.assertEqual(learning_loop.bucket_index(-0.01, 200), 0)
        self.assertEqual(learning_loop.bucket_index(1.01, 200), 201)

    def test_locations_on_linear_log_discrete_and_date_scales(self):
        self.assertAlmostEqual(learning_loop.unscaled_location(150, 100, 200, None), 0.5)
        # Log scale from 1 to 100 with zero point 0: 10 sits halfway, and back.
        self.assertAlmostEqual(learning_loop.unscaled_location(10, 1, 100, 0), 0.5)
        self.assertAlmostEqual(learning_loop.scaled_value(0.5, 1, 100, 0), 10)
        discrete = {"range_min": 117.5, "range_max": 135.5, "zero_point": None, "inbound_outcome_count": 18}
        location = learning_loop.outcome_location("121", "discrete", discrete)
        self.assertEqual(learning_loop.bucket_index(location, 18), 4)  # 120.5 to 121.5
        start = ts("2026-01-01T00:00:00Z")
        end = ts("2027-01-01T00:00:00Z")
        dates = {"range_min": start, "range_max": end, "zero_point": None}
        self.assertAlmostEqual(
            learning_loop.outcome_location("2026-07-02T12:00:00Z", "date", dates), 0.5, places=6
        )
        self.assertEqual(learning_loop.outcome_location("below_lower_bound", "numeric", {}), -math.inf)
        self.assertEqual(
            learning_loop.outcome_location("250", "numeric", {"range_min": 100, "range_max": 200}), math.inf
        )

    def test_a_distribution_of_the_wrong_size_scores_unknown(self):
        q = question(1, "discrete", scaling={"range_min": 0.5, "range_max": 3.5, "inbound_outcome_count": 3})
        with self.assertRaises(learning_loop.ScoreUnknown):
            learning_loop.scores_for("discrete", "2", {"continuous_cdf": linear_cdf(201)}, q)
        scores = learning_loop.scores_for("discrete", "2", {"continuous_cdf": [0.0, 0.2, 0.9, 1.0]}, q)
        self.assertAlmostEqual(scores["log_score"], math.log(0.7))

    def test_the_model_sees_our_distribution_as_percentiles(self):
        numeric = question(1, "numeric", scaling={"range_min": 100.0, "range_max": 200.0, "zero_point": None})
        self.assertEqual(
            learning_loop.forecast_description("numeric", {"continuous_cdf": linear_cdf()}, numeric),
            "10th percentile 105.556, median 150, 90th percentile 194.444",
        )
        self.assertEqual(
            learning_loop.forecast_description("numeric", {"continuous_cdf": linear_cdf(below=0.3, above=0.3)}, numeric),
            "10th percentile below the lower bound, median 150, 90th percentile above the upper bound",
        )
        dates = question(2, "date", scaling={"range_min": ts("2026-01-01T00:00:00Z"),
                                              "range_max": ts("2027-01-01T00:00:00Z"), "zero_point": None})
        self.assertIn("median 2026-07-02", learning_loop.forecast_description("date", {"continuous_cdf": linear_cdf()}, dates))
        self.assertEqual(learning_loop.forecast_description("numeric", {"continuous_cdf": "x"}, numeric), "not readable")

    def test_numeric_and_discrete_sub_questions_end_to_end(self):
        numeric_scaling = {"range_min": 100.0, "range_max": 200.0, "zero_point": None, "inbound_outcome_count": 200}
        discrete_scaling = {"range_min": 0.5, "range_max": 3.5, "zero_point": None, "inbound_outcome_count": 3}
        self.write_ledger(
            line(750001, 65001, {"continuous_cdf": linear_cdf()}, qtype="numeric"),
            line(750002, 65001, {"continuous_cdf": [0.0, 0.3, 0.6, 0.98]}, qtype="discrete"),
            line(750003, 65001, {"continuous_cdf": [0.0, 0.1, 0.2, 0.3]}, qtype="discrete"),
        )
        self.api.add(
            post(
                65001,
                question(750001, "numeric", resolution="150", scaling=numeric_scaling,
                         open_lower_bound=True, open_upper_bound=True, unit="SYNTHETIC units"),
                question(750002, "discrete", resolution="above_upper_bound", scaling=discrete_scaling,
                         open_lower_bound=False, open_upper_bound=True),
                question(750003, "discrete", status="open", scaling=discrete_scaling),
                group=True,
            )
        )
        self.run_loop()
        questions = self.state()["questions"]
        self.assertEqual(set(questions), {"750001", "750002"})
        self.assertAlmostEqual(questions["750001"]["log_score"], math.log(0.0045))
        self.assertEqual(questions["750001"]["outcome_text"], "150")
        self.assertEqual(questions["750001"]["forecast_text"], "0.45% in the outcome's bin")
        self.assertAlmostEqual(questions["750002"]["log_score"], math.log(0.02))
        self.assertEqual(questions["750002"]["forecast_text"], "2% beyond the bound")
        lessons = self.lessons()
        self.assertIn("| Numeric | 1 | n/a | -5.404 |", lessons)
        self.assertIn("| Discrete | 1 | n/a | -3.912 |", lessons)
        self.assertNotIn("SYNTHETIC units", self.written())


class CalibrationTests(LearningLoopCase):
    def test_ten_bins_with_count_mean_forecast_and_observed_frequency(self):
        records = [
            {"type": "binary", "probability_yes": p, "outcome": outcome}
            for p, outcome in [
                (0.05, "no"), (0.1, "yes"), (0.15, "no"), (0.3, "no"),
                (0.7, "yes"), (0.99, "yes"), (1.0, "yes"),
            ]
        ]
        records.append({"type": "numeric", "probability_yes": None, "outcome": "5"})
        bins = learning_loop.calibration_bins(records)
        self.assertEqual([b["count"] for b in bins], [1, 2, 0, 1, 0, 0, 0, 1, 0, 2])
        self.assertAlmostEqual(bins[1]["forecast_sum"], 0.25)
        self.assertEqual(bins[1]["yes"], 1)
        self.assertEqual(bins[9]["yes"], 2)

    def test_the_table_is_rewritten_from_every_morning(self):
        self.write_ledger(line(800001, 70001, binary(0.1)), line(800002, 70002, binary(0.15)))
        self.api.add(post(70001, question(800001, resolution="yes")), post(70002, question(800002, resolution="no")))
        self.run_loop()
        self.write_ledger(line(800003, 70003, binary(0.95)))
        self.api.add(post(70003, question(800003, resolution="yes")))
        self.run_loop(NEXT_MORNING)
        lessons = self.lessons()
        self.assertIn("| 10% to 20% | 2 | 12.5% | 50% |", lessons)
        self.assertIn("| 90% to 100% | 1 | 95% | 100% |", lessons)
        self.assertIn("| 0% to 10% | 0 | n/a | n/a |", lessons)
        self.assertIn("From the 3 resolved questions reported so far.", lessons)
        brier = (0.81 + 0.0225 + 0.0025) / 3
        log_score = (math.log(0.1) + math.log(0.85) + math.log(0.95)) / 3
        self.assertIn(f"| Yes/no | 3 | {brier:.3f} | {log_score:.3f} |", lessons)


class WorstMissTests(LearningLoopCase):
    def test_lowest_three_below_uniform_with_ties_broken_by_id(self):
        records = [
            {"question_id": 1, "above_uniform": -2.0},
            {"question_id": 2, "above_uniform": -0.5},
            {"question_id": 3, "above_uniform": -3.0},
            {"question_id": 4, "above_uniform": 0.4},
            {"question_id": 0, "above_uniform": -0.5},
            {"question_id": 5, "above_uniform": "Unknown"},
        ]
        self.assertEqual([r["question_id"] for r in learning_loop.worst_misses(records)], [3, 1, 0])
        self.assertEqual(learning_loop.worst_misses([{"question_id": 4, "above_uniform": 0.4}]), [])

    def test_the_three_worst_get_a_cause_and_the_rest_do_not(self):
        self.with_model()
        forecasts = {810001: 0.1, 810002: 0.3, 810003: 0.05, 810004: 0.45, 810005: 0.9}
        for n, (question_id, p) in enumerate(forecasts.items()):
            self.write_ledger(line(question_id, 71000 + n, binary(p)))
            self.api.add(post(71000 + n, question(question_id, resolution="yes")))
        self.run_loop()
        self.assertEqual(len(self.review_prompts), 3)
        state = self.state()
        self.assertEqual(state["mornings"]["2026-11-04"]["worst"], [810003, 810001, 810002])
        for question_id in (810003, 810001, 810002):
            self.assertEqual(state["questions"][str(question_id)]["cause"], SYNTHETIC_CAUSE)
        self.assertIsNone(state["questions"]["810004"]["cause"])
        self.assertIsNone(state["questions"]["810005"]["cause"])
        lessons = self.lessons()
        worst = lessons[lessons.index("\nWorst misses:\n"):]
        self.assertIn("1. [Question 810003]", worst)
        self.assertIn("2. [Question 810001]", worst)
        self.assertIn("3. [Question 810002]", worst)
        self.assertNotIn("Question 810004", worst)
        self.assertIn(f"Cause: {SYNTHETIC_CAUSE}", worst)

    def test_a_morning_without_misses_says_so_and_calls_no_model(self):
        self.with_model()
        self.write_ledger(line(820001, 72001, binary(0.8)))
        self.api.add(post(72001, question(820001, resolution="yes")))
        self.run_loop()
        self.assertEqual(self.review_prompts, [])
        self.assertIn("No misses: every forecast did at least as well as a uniform one.", self.lessons())


class ReviewTests(LearningLoopCase):
    def setUp(self):
        super().setUp()
        self.write_ledger(line(830001, 73001, binary(0.2)))
        self.api.add(post(73001, question(830001, resolution="yes", title="SYNTHETIC reviewed question")))

    def test_reviews_go_through_the_metered_path(self):
        self.with_model()
        self.set_env(GITHUB_ACTIONS="true")  # as on GitHub: the spend record the workflow commits
        self.run_loop()
        calls = [r for r in self.read_jsonl("forecasts/spend.jsonl") if r["event"] == "model_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["purpose"], "learning_review")
        self.assertEqual(calls[0]["run_mode"], "learning_loop")
        self.assertEqual(calls[0]["model"], "openai/gpt-6-luna")
        self.assertEqual(calls[0]["question_id"], 830001)
        self.assertEqual(calls[0]["cost_usd"], "0.000105")
        self.assertEqual(calls[0]["cost_source"], "openrouter_reported")
        self.assertFalse(calls[0]["dry_run"])
        prompt = self.review_prompts[0]
        for given in ("Question: SYNTHETIC reviewed question", "SYNTHETIC criteria for 830001.",
                      "Outcome: Yes", "20% chance of Yes",
                      "Probability our forecast gave to the outcome: 20%", "in your own words"):
            self.assertIn(given, prompt)

    def test_the_bots_own_reasoning_comes_from_its_latest_comment(self):
        self.with_model()
        self.api.comments[73001] = [
            {"id": 1, "author": {"id": 9999}, "text": "SYNTHETIC OTHER AUTHOR TEXT", "created_at": "2026-10-30T00:00:00Z"},
            {"id": 2, "author": {"id": 4242}, "text": "# FORECASTS\nSYNTHETIC OLD REASONING", "created_at": "2026-10-10T00:00:00Z"},
            {"id": 3, "author": {"id": 4242}, "text": "# SUMMARY\nSYNTHETIC summary\n# FORECASTS\nSYNTHETIC NEWEST REASONING", "created_at": "2026-10-20T00:00:00Z"},
        ]
        self.run_loop()
        prompt = self.review_prompts[0]
        self.assertIn("SYNTHETIC NEWEST REASONING", prompt)
        self.assertNotIn("SYNTHETIC OTHER AUTHOR TEXT", prompt)
        self.assertNotIn("SYNTHETIC OLD REASONING", prompt)
        self.assertNotIn("SYNTHETIC summary", prompt)
        self.assertIn(("/comments/", {"post": 73001, "author": 4242, "limit": 50}), self.api.gets)
        self.assertNotIn("REASONING", self.written())

    def test_reasoning_that_cannot_be_read_is_not_available(self):
        self.with_model()
        self.api.status["/comments/"] = 400
        self.assertEqual(self.run_loop(), 0)
        self.assertIn("Not available.", self.review_prompts[0])
        self.assertEqual(self.state()["questions"]["830001"]["cause"], SYNTHETIC_CAUSE)

    def test_model_failure_gives_unknown_not_reviewed(self):
        self.with_model()
        self.review_error = RuntimeError("SYNTHETIC model failure")
        self.assertEqual(self.run_loop(), 0)
        self.assertEqual(self.state()["questions"]["830001"]["cause"], "Unknown (not reviewed)")
        self.assertIn("Cause: Unknown (not reviewed)", self.lessons())
        self.assertIn("could not be reviewed", self.lessons())
        # The failed call was still metered, at its worst case.
        calls = [r for r in self.read_jsonl("forecasts/dry_run/spend.jsonl") if r["event"] == "model_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["cost_usd"], "Unknown")
        self.assertTrue(calls[0]["outcome"].startswith("failed"))

    def test_no_model_key_means_not_reviewed_and_no_call(self):
        self.run_loop()
        self.assertEqual(self.review_prompts, [])
        self.assertEqual(self.state()["questions"]["830001"]["cause"], "Unknown (not reviewed)")
        self.assertIn("not reviewed: no OPENROUTER_API_KEY is set", self.lessons())
        self.assertFalse(Path("forecasts/dry_run/spend.jsonl").exists())
        self.assertNotIn("/users/me", self.api.paths())

    def test_a_cap_without_room_means_not_reviewed_and_no_call(self):
        self.with_model()
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="0.001", GLASS_EARTH_HIGH_VALUE_RESERVE_USD="0")
        self.run_loop()
        self.assertEqual(self.review_prompts, [])
        self.assertEqual(self.state()["questions"]["830001"]["cause"], "Unknown (not reviewed)")
        self.assertIn("the daily cap had no room", self.lessons())

    def test_the_reviews_leave_the_high_value_reserve_free(self):
        self.with_model()
        # SYNTHETIC: a cap of US$0.80 with all of it reserved leaves no room.
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="0.80", GLASS_EARTH_HIGH_VALUE_RESERVE_USD="0.80")
        self.run_loop()
        self.assertEqual(self.review_prompts, [])

    def test_dashes_never_reach_the_written_files(self):
        self.with_model()
        self.review_answer = f"- Cause: overconfident {EM_DASH} ignored the base rate\nsecond line"
        self.run_loop()
        written = self.written()
        self.assertNotIn(EM_DASH, written)
        self.assertIn("Cause: overconfident - ignored the base rate", self.lessons())
        self.assertNotIn("second line", written)


class RerunTests(LearningLoopCase):
    def setUp(self):
        super().setUp()
        self.write_ledger(
            line(840001, 74001, binary(0.2)),
            line(840002, 74002, binary(0.7)),
            line(840003, 74003, binary(0.4)),
        )
        self.api.add(
            post(74001, question(840001, resolution="yes")),
            post(74002, question(840002, resolution="yes")),
            post(74003, question(840003, status="open")),
        )

    def test_a_second_run_reports_nothing_twice(self):
        self.with_model()
        self.run_loop()
        state = Path("forecasts/learning_state.json").read_bytes()
        lessons = Path("LESSONS.md").read_bytes()
        prompts = len(self.review_prompts)
        self.assertEqual(prompts, 1)
        self.api.gets.clear()
        self.run_loop()
        self.assertEqual(Path("forecasts/learning_state.json").read_bytes(), state)
        self.assertEqual(Path("LESSONS.md").read_bytes(), lessons)
        self.assertEqual(len(self.review_prompts), prompts)
        # Only the question still open is asked about again.
        self.assertEqual(self.api.paths(), ["/posts/74003/"])
        self.run_loop(NEXT_MORNING)
        mornings = self.state()["mornings"]
        self.assertEqual(mornings["2026-11-04"]["questions"], [840001, 840002])
        self.assertEqual(mornings["2026-11-05"]["questions"], [])
        lessons = self.lessons()
        self.assertIn("### 5 November 2026\n\nNothing new resolved.", lessons)
        self.assertLess(lessons.index("### 5 November 2026"), lessons.index("### 4 November 2026"))
        self.assertEqual(lessons.count("| [Question 840001]"), 1)

    def test_a_morning_with_nothing_resolved_writes_one_line(self):
        Path("forecasts/ledger.jsonl").unlink()
        self.run_loop()
        self.assertEqual(self.api.gets, [])
        lessons = self.lessons()
        section = lessons[lessons.index("### 4 November 2026"):]
        self.assertEqual(section, "### 4 November 2026\n\nNothing new resolved.\n")

    def test_questions_beyond_the_morning_limit_wait_for_the_next_morning(self):
        with mock.patch.object(learning_loop, "MAX_POSTS_PER_MORNING", 2):
            self.run_loop()
            self.assertEqual(self.api.paths(), ["/posts/74001/", "/posts/74002/"])
            self.assertIn("1 posts were not checked this morning", self.lessons())
            self.api.gets.clear()
            self.run_loop(NEXT_MORNING)
            self.assertEqual(self.api.paths(), ["/posts/74003/"])


class PoliteRequestTests(LearningLoopCase):
    def setUp(self):
        super().setUp()
        for n, recorded in enumerate(("2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z", "2026-10-03T00:00:00Z")):
            self.write_ledger(line(850001 + n, 75001 + n, binary(0.3), recorded=recorded))
            self.api.add(post(75001 + n, question(850001 + n, resolution="yes")))

    def test_requests_are_paced_and_oldest_forecast_first(self):
        self.run_loop()
        self.assertEqual(self.api.paths(), ["/posts/75001/", "/posts/75002/", "/posts/75003/"])
        self.assertEqual(self.pauses, [4.0, 4.0])

    def test_429_stops_cleanly(self):
        self.with_model()
        self.api.status["/posts/75002/"] = 429
        self.assertEqual(self.run_loop(), 0)
        # No request after the 429 and nothing retried, the reviews' included.
        self.assertEqual(self.api.paths(), ["/posts/75001/", "/posts/75002/"])
        self.assertEqual(set(self.state()["questions"]), {"850001"})
        lessons = self.lessons()
        self.assertIn("Stopped early: Metaculus answered 429 (too many requests). 1 of 3 posts were checked", lessons)
        self.assertNotIn("Question 850002", lessons)
        # The miss already read is still reviewed, without the reasoning.
        self.assertEqual(len(self.review_prompts), 1)
        self.assertIn("Not available.", self.review_prompts[0])
        # The next run picks up the rest and reports nothing twice.
        del self.api.status["/posts/75002/"]
        self.api.gets.clear()
        self.assertEqual(self.run_loop(), 0)
        self.assertEqual(self.api.paths()[:2], ["/posts/75002/", "/posts/75003/"])
        self.assertEqual(self.state()["mornings"]["2026-11-04"]["questions"], [850001, 850002, 850003])

    def test_a_refusal_stops_and_fails_the_run(self):
        self.api.status["/posts/75001/"] = 403
        self.assertEqual(self.run_loop(), 1)
        self.assertEqual(self.api.paths(), ["/posts/75001/"])
        self.assertIn("Metaculus refused the request (HTTP 403)", self.lessons())

    def test_setup_problems_stop_before_anything_is_asked_or_written(self):
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="two dollars")
        with self.assertRaises(learning_loop.SetupError):
            self.run_loop()
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="2.00", METACULUS_TOKEN="")
        with self.assertRaises(learning_loop.SetupError):
            self.run_loop()
        self.assertEqual(self.api.gets, [])
        self.assertFalse(Path("LESSONS.md").exists())
        self.assertFalse(Path("forecasts/learning_state.json").exists())


class CommittedFilesTests(LearningLoopCase):
    def test_committed_lessons_are_rendered_from_the_committed_state(self):
        state = json.loads((REPO / "forecasts" / "learning_state.json").read_text(encoding="utf-8"))
        committed = (REPO / "LESSONS.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
        self.assertEqual(learning_loop.render_lessons(state), committed)
        for rule in (
            "Resolved questions only.",
            "at most once a week",
            "capped at each question's close date",
            "AskNews is the planned one",
            "no lesson-driven change is made",
            "terms of use",
        ):
            self.assertIn(rule, committed)


if __name__ == "__main__":
    import unittest

    unittest.main()
