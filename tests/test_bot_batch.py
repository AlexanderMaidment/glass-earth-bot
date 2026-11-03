"""
A batch of 10 open binary questions through the bot as the tournament
workflow runs it (main.py with no arguments, on GitHub Actions, with no
repository variables set), and the guard that keeps questions from running
at the same time. The questions, model answers and reported costs are all
SYNTHETIC.
"""

import asyncio
import contextlib
import io
import unittest
from decimal import Decimal

import forecasting_tools.ai_models.general_llm as general_llm
from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.binary_report import BinaryReport

from spend import OverlappingBatchRefused
from support import FAKE_LEDGER_KEY_TEXT, OfflineTestCase

FUTUREEVAL = "fall-futureeval-2026"
POST_IDS = [7000 + n for n in range(1, 11)]
QUESTION_IDS = [70000 + n for n in range(1, 11)]

# SYNTHETIC costs for each call to report, in US dollars, matching the
# typical-token assumption in GLASS_EARTH.md (calculated there): research
# US$0.039, one Opus forecast US$0.076, one Fable forecast US$0.19, and a
# summary or parsing call US$0.0004.
TYPICAL_COSTS = {
    "openrouter/openai/gpt-6.1-terra:online": "0.039",
    "openrouter/anthropic/claude-opus-5.5": "0.076",
    "openrouter/anthropic/claude-fable-5.1": "0.19",
    "openrouter/openai/gpt-6-luna": "0.0004",
}


def binary_post(n: int) -> dict:
    """A SYNTHETIC open binary post shaped like one from /api/posts/."""
    title = f"SYNTHETIC: Will made-up event number {n} happen?"
    return {
        "id": 7000 + n,
        "title": title,
        "nr_forecasters": 3,
        "forecasts_count": 5,
        "published_at": "2026-10-07T20:00:00Z",
        "projects": {
            "tournament": [{"id": 77002, "slug": FUTUREEVAL, "name": "SYNTHETIC tournament"}],
            "default_project": {"id": 77002, "visibility": "normal"},
        },
        "question": {
            "id": 70000 + n,
            "title": title,
            "type": "binary",
            "status": "open",
            "description": "SYNTHETIC background.",
            "resolution_criteria": "SYNTHETIC: resolves Yes if the made-up event happens.",
            "fine_print": "SYNTHETIC fine print.",
            "open_time": "2026-10-07T20:00:00Z",
            "scheduled_close_time": "2099-01-01T10:00:00Z",
            "scheduled_resolve_time": "2099-01-01T20:00:00Z",
            "question_weight": 1.0,
            "my_forecasts": {"history": [], "latest": None},
        },
    }


def binary_question(n: int) -> BinaryQuestion:
    return BinaryQuestion(
        question_text=f"SYNTHETIC: Will made-up event number {n} happen?",
        id_of_post=7000 + n,
        id_of_question=70000 + n,
        page_url=f"https://www.metaculus.com/questions/{7000 + n}/",
        background_info="SYNTHETIC background.",
        resolution_criteria="SYNTHETIC criteria.",
        fine_print="SYNTHETIC fine print.",
        tournament_slugs=[FUTUREEVAL],
    )


class TournamentWorkflowBatchTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        # The environment run_bot_on_tournament.yaml gives main.py: GitHub
        # Actions, the secrets (the ledger key SYNTHETIC), and every
        # repository variable unset, which GitHub passes as an empty string.
        self.set_env(
            GITHUB_ACTIONS="true",
            OPENROUTER_API_KEY="sk-or-synthetic-batch-key",
            GLASS_EARTH_LEDGER_KEY=FAKE_LEDGER_KEY_TEXT,
            GLASS_EARTH_DAILY_CAP_USD="",
            GLASS_EARTH_HIGH_VALUE_RESERVE_USD="",
            GLASS_EARTH_ENSEMBLE_SIZE="",
            GLASS_EARTH_HIGH_VALUE_MODEL="",
            GLASS_EARTH_MARKET_PULSE="",
        )
        # The fake serves this page for every tournament, so the MiniBench
        # pass after the FutureEval one sees the same 10 questions again.
        self.metaculus.posts_page = {
            "next": None,
            "previous": None,
            "results": [binary_post(n) for n in range(1, 11)],
        }
        self.models.cost_by_model = dict(TYPICAL_COSTS)

    def run_workflow(self):
        """main.py exactly as the workflow's "Run bot" step calls it."""
        import main

        with self.assertLogs("main", level="INFO") as logs:
            with contextlib.redirect_stdout(io.StringIO()) as console:
                reports = main.run([])
        return reports, console.getvalue(), logs.output

    def spend_records(self):
        return self.read_jsonl("forecasts/spend.jsonl")

    def assert_questions_ran_one_after_another(self):
        # Every SYNTHETIC call yields to the event loop, so questions running
        # at the same time would interleave their calls. Each question's
        # calls form one unbroken run, in order.
        order = [c["question_id"] for c in self.models.calls]
        runs = [q for i, q in enumerate(order) if i == 0 or order[i - 1] != q]
        self.assertEqual(runs, sorted(set(order)))

    def test_ten_questions_are_each_forecast_once_with_one_comment_under_the_default_cap(self):
        reports, console, logs = self.run_workflow()
        self.assertIn("publish=yes", console)
        self.assertIn("Setup: 1 forecast per question from anthropic/claude-opus-5.5", console)
        self.assertEqual(len(reports), 10)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])

        # Forecast exactly once each, by one Opus forecast (FutureEval questions
        # are high value, but Fable is off by default). The MiniBench pass
        # skipped all 10 as already in the ledger.
        forecasts = self.metaculus.forecasts()
        self.assertEqual(sorted(f["question"] for f in forecasts), QUESTION_IDS)
        self.assertEqual(
            [c["model"] for c in self.models.forecast_calls],
            ["openrouter/anthropic/claude-opus-5.5"] * 10,
        )
        self.assertEqual(
            sorted(line["question_id"] for line in self.read_jsonl("forecasts/ledger.jsonl")),
            QUESTION_IDS,
        )

        # One comment per question, stating the forecast submitted for it.
        comments = self.metaculus.comments()
        self.assertEqual(sorted(c["on_post"] for c in comments), POST_IDS)
        text_for_post = {c["on_post"]: c["text"] for c in comments}
        for sent in forecasts:
            text = text_for_post[sent["question"] - 70000 + 7000]
            stated = BinaryReport.make_readable_prediction(sent["probability_yes"])
            self.assertIn(f"*Final Prediction*: {stated}", text)
            self.assertIn(
                "*How the final prediction was made*: 1 forecast from "
                "anthropic/claude-opus-5.5, held between 1% and 99%.",
                text,
            )
            self.assertTrue(comments[0]["is_private"])

        # The meter recorded every call, each against its own question.
        records = self.spend_records()
        calls = [r for r in records if r["event"] == "model_call"]
        self.assertEqual(len(calls), len(self.models.calls))
        self.assertEqual(sorted({r["question_id"] for r in calls}), QUESTION_IDS)
        for question_id in QUESTION_IDS:
            purposes = [r["purpose"] for r in calls if r["question_id"] == question_id]
            self.assertEqual(purposes.count("researcher"), 1)
            self.assertEqual(purposes.count("summarizer"), 1)
            self.assertEqual(purposes.count("forecaster"), 1)
            self.assertGreaterEqual(purposes.count("parser"), 1)
        self.assertTrue(all(r["daily_cap_usd"] == "2.00" for r in records))
        self.assertTrue(all(r["cost_source"] == "openrouter_reported" for r in calls))

        # The total stays under the cap: 10 x US$0.1162 (research 0.039,
        # summary 0.0004, forecast 0.076, two parses 0.0008), SYNTHETIC.
        total = sum(Decimal(r["cap_basis_usd"]) for r in records)
        self.assertEqual(total, Decimal("1.1620"))
        self.assertLessEqual(total, Decimal("2.00"))
        self.assertEqual([r for r in records if r["event"] != "model_call"], [])
        self.assertIn("US$1.1620 of the US$2.0000 daily cap", console)
        self.assertIn("0 question(s) skipped by the cap", console)
        self.assert_questions_ran_one_after_another()

    def test_a_cap_that_fits_only_some_skips_the_rest_with_a_reason(self):
        self.set_env(GLASS_EARTH_DAILY_CAP_USD="1.00")
        reports, console, logs = self.run_workflow()
        cap = Decimal("1.00")

        # Each question starts only if its reservation (US$0.27 for one Opus
        # forecast: twice the measured cost of one real question, rounded up
        # to the cent, calculated) fits in what is left. At US$0.1162 spent
        # per question (SYNTHETIC), questions 1 to 7 fit (0.6972 + 0.27 is
        # under 1.00) and question 8 does not (0.8134 + 0.27 is over).
        forecast_ids = [f["question"] for f in self.metaculus.forecasts()]
        self.assertEqual(forecast_ids, QUESTION_IDS[:7])
        self.assertEqual(len(reports), 7)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])

        records = self.spend_records()
        skips = [r for r in records if r["event"] == "question_skipped"]
        # Skipped in the FutureEval pass and again in the MiniBench pass.
        self.assertEqual(sorted({s["question_id"] for s in skips}), QUESTION_IDS[7:])
        spent = Decimal("0")
        for record in records:
            if record["event"] == "model_call":
                spent += Decimal(record["cap_basis_usd"])
            elif record["event"] == "question_skipped":
                self.assertIn("daily cap", record["reason"])
                # Skipped only because its reservation would not fit.
                self.assertEqual(record["reservation_usd"], "0.27")
                self.assertGreater(spent + Decimal(record["reservation_usd"]), cap)
                # Its calculated worst case is recorded too, for information.
                self.assertGreater(Decimal(record["worst_case_usd"]), Decimal("0.40"))
        self.assertLessEqual(spent, cap)
        # Nothing was spent on a skipped question, and no call was refused
        # halfway through a question.
        self.assertFalse(
            [r for r in records if r["event"] == "model_call" and r["question_id"] in QUESTION_IDS[7:]]
        )
        self.assertFalse([r for r in records if r["event"] == "call_refused"])
        # Each skip is logged with its reason, and counted in the summary.
        logged = [line for line in logs if "Spend cap: skipping" in line]
        self.assertEqual(len(logged), len(skips))
        self.assertTrue(all("daily cap: US$" in line for line in logged))
        self.assertIn(f"{len(skips)} question(s) skipped by the cap", console)
        self.assert_questions_ran_one_after_another()

    def test_stronger_setup_through_the_repository_variables(self):
        self.set_env(
            GLASS_EARTH_ENSEMBLE_SIZE="3",
            GLASS_EARTH_HIGH_VALUE_MODEL="true",
            GLASS_EARTH_DAILY_CAP_USD="5.00",
        )
        reports, console, logs = self.run_workflow()
        self.assertIn("Setup: 3 forecasts (median submitted) per question", console)
        self.assertEqual(len(reports), 10)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])
        self.assertEqual(sorted(f["question"] for f in self.metaculus.forecasts()), QUESTION_IDS)

        # Fable needs room for its worst case (about US$2.06 for three
        # forecasts, calculated) plus a default question's reservation
        # (US$0.81 for three Opus forecasts, calculated). At US$0.6118 per
        # Fable question (SYNTHETIC), questions 1 to 4 get Fable (1.8354 +
        # 2.06 + 0.81 is under 5.00); questions 5 to 10 fall back to Opus.
        ledger = sorted(self.sealed_records(), key=lambda line: line["question_id"])
        models = [line["forecaster_model"] for line in ledger]
        self.assertEqual(
            models, ["anthropic/claude-fable-5.1"] * 4 + ["anthropic/claude-opus-5.5"] * 6
        )
        self.assertEqual(len(self.models.forecast_calls), 30)
        text_for_post = {c["on_post"]: c["text"] for c in self.metaculus.comments()}
        for line in ledger:
            self.assertEqual(len(line["model_forecasts"]), 3)
            self.assertIn(
                f"the median of 3 independent forecasts from {line['forecaster_model']}",
                text_for_post[line["post_id"]],
            )
        total = sum(Decimal(r["cap_basis_usd"]) for r in self.spend_records())
        self.assertLessEqual(total, Decimal("5.00"))
        self.assert_questions_ran_one_after_another()


class ConcurrencyTests(OfflineTestCase):
    def test_a_second_batch_alongside_a_running_one_is_refused_before_it_spends(self):
        bot = self.make_bot(publish=False, skip=False)
        other_bot = self.make_bot(publish=False, skip=False, meter=bot.meter)
        first = [binary_question(n) for n in (1, 2, 3)]
        second = [binary_question(n) for n in (4, 5, 6)]

        async def overlapping(second_bot):
            return await asyncio.gather(
                bot.forecast_questions(first, return_exceptions=True),
                second_bot.forecast_questions(second, return_exceptions=True),
                return_exceptions=True,
            )

        # The same bot, and another bot sharing the same daily cap.
        for second_bot in (bot, other_bot):
            self.models.calls.clear()
            done, refused = asyncio.run(overlapping(second_bot))
            self.assertEqual(len(done), 3)
            self.assertFalse([r for r in done if isinstance(r, BaseException)])
            self.assertIsInstance(refused, OverlappingBatchRefused)
            self.assertEqual({c["question_id"] for c in self.models.calls}, {70001, 70002, 70003})

        # Once the first batch has finished, the next one runs.
        self.models.calls.clear()
        reports = asyncio.run(other_bot.forecast_questions(second, return_exceptions=True))
        self.assertEqual(len(reports), 3)
        self.assertEqual({c["question_id"] for c in self.models.calls}, {70004, 70005, 70006})

    def test_members_of_one_question_run_together_without_passing_the_cap(self):
        import main

        question = binary_question(1)
        probe = self.make_bot(publish=False, skip=False, ensemble_size=5)
        cap = probe._worst_case_question_cost(question, main.DEFAULT_FORECASTER) + Decimal("0.0001")
        meter = self.make_meter(cap=str(cap), reserve="0")
        bot = self.make_bot(publish=False, skip=False, meter=meter, ensemble_size=5)
        committed: list[Decimal] = []
        state = {"open": 0, "most_open": 0}
        synthetic = self.models.acompletion

        async def watched(messages, **kwargs):
            committed.append(meter.spent_today + meter.in_flight)
            state["open"] += 1
            state["most_open"] = max(state["most_open"], state["open"])
            try:
                return await synthetic(messages, **kwargs)
            finally:
                state["open"] -= 1

        general_llm.acompletion = watched  # tearDown puts the real one back
        [report] = asyncio.run(bot.forecast_questions([question], return_exceptions=True))
        self.assertFalse(isinstance(report, BaseException))
        self.assertEqual(len(self.models.forecast_calls), 5)
        # The five forecasts were in flight together, and what was spent plus
        # reserved never passed the cap.
        self.assertEqual(state["most_open"], 5)
        self.assertLessEqual(max(committed), cap)
        self.assertEqual(meter.in_flight, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
