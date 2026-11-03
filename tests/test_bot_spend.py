"""
The daily cap, cost per question and routing inside the bot, end to end on
SYNTHETIC questions and SYNTHETIC model answers.
"""

import asyncio
import json
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import spend
from spend import SpendCapReached
from support import OfflineTestCase, load_fixture


def _page_for(slug: str, keep_ids: set[int]):
    page = load_fixture("synthetic_group_posts.json")
    for post in page["results"]:
        post["projects"]["tournament"][0]["slug"] = slug
        post["group_of_questions"]["questions"] = [
            q for q in post["group_of_questions"]["questions"] if q["id"] in keep_ids
        ]
    page["results"] = [p for p in page["results"] if p["group_of_questions"]["questions"]]
    return page


class MeterIntegrationTests(OfflineTestCase):
    def test_every_call_is_metered_and_the_ledger_has_cost_per_question(self):
        self.metaculus.posts_page = _page_for("bot-testing-area", {90011, 90021})
        bot = self.make_bot(publish=True, skip=False)
        asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        calls = [r for r in self.read_jsonl("forecasts/spend.jsonl") if r["event"] == "model_call"]
        self.assertEqual(len(calls), len(self.models.calls))
        self.assertTrue(all(r["cost_source"] == "openrouter_reported" for r in calls))
        ledger = {record["question_id"]: record for record in self.sealed_records()}
        for question_id in (90011, 90021):
            line = ledger[question_id]
            question_calls = [r for r in calls if r["question_id"] == question_id]
            total = sum(Decimal(r["cost_usd"]) for r in question_calls)
            self.assertEqual(Decimal(line["cost_usd"]), total)
            self.assertEqual(line["model_calls"], len(question_calls))
            self.assertEqual(line["forecaster_model"], "anthropic/claude-opus-5.5")
            self.assertEqual(line["prices_retrieved_at_utc"], spend.PRICES_RETRIEVED_AT_UTC)
            self.assertEqual(line["ledger_version"], 3)
        purposes = {r["purpose"] for r in calls}
        self.assertEqual(purposes, {"researcher", "summarizer", "forecaster", "parser"})
        # Every call carried a max_tokens limit.
        self.assertTrue(all(c["max_tokens"] for c in self.models.calls))

    def test_comment_states_the_metered_cost(self):
        self.metaculus.posts_page = _page_for("bot-testing-area", {90011})
        bot = self.make_bot(publish=True, skip=False)
        asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        comment = self.metaculus.comments()[0]["text"]
        line = self.sealed_records()[0]
        self.assertIn(f"*Total Cost*: US${line['cost_usd']} over {line['model_calls']} model calls", comment)

    def test_unknown_costs_are_never_zero(self):
        self.models.report_cost = False
        self.metaculus.posts_page = _page_for("bot-testing-area", {90011})
        bot = self.make_bot(publish=True, skip=False)
        asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        line = self.sealed_records()[0]
        # The web-search research call has no reported cost, so the total is Unknown,
        # and the cap counted that call at its worst case.
        self.assertEqual(line["cost_usd"], "Unknown")
        self.assertGreater(Decimal(line["cost_usd_cap_basis"]), Decimal("0.05"))
        research = [
            r for r in self.read_jsonl("forecasts/spend.jsonl") if r.get("purpose") == "researcher"
        ][0]
        self.assertEqual(research["cost_usd"], "Unknown")
        self.assertEqual(research["cap_basis_usd"], research["worst_case_usd"])
        self.assertIn("Unknown", self.metaculus.comments()[0]["text"])

    def test_cap_skips_cleanly_and_spends_nothing_on_the_skipped_question(self):
        import main

        self.metaculus.posts_page = _page_for("bot-testing-area", {90011, 90012})
        meter = self.make_meter(cap="5.00", reserve="0")
        bot = self.make_bot(publish=True, skip=False, meter=meter, ensemble_size=3)
        # A cap just above one question's reservation (US$0.81 for three Opus
        # forecasts, calculated) allows the first question, and the first
        # question's SYNTHETIC spending (about US$0.01) leaves too little for
        # the second. test_bot_batch.py checks the same with the defaults.
        first = bot.metaculus_client.get_all_open_questions_from_tournament("bot-testing-area")[0]
        reservation = bot._question_reservation(first)
        self.assertEqual(reservation, Decimal("0.81"))
        self.assertGreater(bot._worst_case_question_cost(first, main.DEFAULT_FORECASTER), reservation)
        meter.cap = reservation + Decimal("0.005")
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        self.assertEqual(len(reports), 1)
        self.assertFalse(isinstance(reports[0], BaseException))
        self.assertEqual([f["question"] for f in self.metaculus.forecasts()], [90011])
        records = self.read_jsonl("forecasts/spend.jsonl")
        skips = [r for r in records if r["event"] == "question_skipped"]
        self.assertEqual([s["question_id"] for s in skips], [90012])
        self.assertIn("daily cap", skips[0]["reason"])
        self.assertFalse(any(r.get("question_id") == 90012 and r["event"] == "model_call" for r in records))
        self.assertIn("1 question(s) skipped by the cap", meter.summary_line())

    def test_cap_holds_across_runs_on_the_same_utc_day(self):
        Path("forecasts").mkdir()
        today = spend.utc_day(datetime.now(timezone.utc))
        # US$0.20 left is less than one question's reservation (US$0.27 with
        # one Opus forecast, calculated).
        earlier = [
            {"utc_day": today, "event": "model_call", "cap_basis_usd": "4.80"},
            {"utc_day": "2000-01-01", "event": "model_call", "cap_basis_usd": "99"},
        ]
        Path("forecasts/spend.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in earlier), encoding="utf-8"
        )
        meter = self.make_meter(cap="5.00", reserve="0")
        self.assertEqual(meter.spent_today, Decimal("4.80"))
        self.metaculus.posts_page = _page_for("bot-testing-area", {90011})
        bot = self.make_bot(publish=True, skip=False, meter=meter)
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        self.assertEqual(reports, [])
        self.assertEqual(self.models.calls, [])

    def test_per_call_guard_refuses_without_retrying(self):
        meter = self.make_meter(cap="0.01", reserve="0")
        bot = self.make_bot(publish=False, skip=False, meter=meter)
        llm = bot.get_llm("forecaster_default", "llm")
        with self.assertRaises(SpendCapReached):
            asyncio.run(llm.invoke("SYNTHETIC prompt: You are a professional forecaster."))
        self.assertEqual(self.models.calls, [])
        refused = [r for r in self.read_jsonl("forecasts/spend.jsonl") if r["event"] == "call_refused"]
        self.assertEqual(len(refused), 1)

    def test_failed_call_counts_its_worst_case(self):
        meter = self.make_meter()
        bot = self.make_bot(publish=False, skip=False, meter=meter)
        llm = bot.get_llm("summarizer", "llm")
        llm.allowed_tries = 1
        self.models.fail_models.add("openrouter/openai/gpt-6-luna")
        with self.assertRaises(Exception):
            asyncio.run(llm.invoke("Please summarize the following research SYNTHETIC"))
        record = [r for r in self.read_jsonl("forecasts/spend.jsonl") if r["event"] == "model_call"][0]
        self.assertEqual(record["cost_usd"], "Unknown")
        self.assertTrue(record["outcome"].startswith("failed"))
        self.assertEqual(meter.spent_this_run, Decimal(record["worst_case_usd"]))

    def test_unmetered_models_are_refused(self):
        import main
        from forecasting_tools import GeneralLlm

        with self.assertRaises(ValueError):
            main.GlassEarthBot(
                run_mode="x",
                spend_meter=self.make_meter(),
                llms={"default": GeneralLlm(model="openrouter/anthropic/claude-opus-5.5")},
            )

    def test_high_value_question_gets_one_opus_forecast_by_default(self):
        self.metaculus.posts_page = _page_for("fall-futureeval-2026", {90011})
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_on_tournament("fall-futureeval-2026", return_exceptions=True))
        self.assertEqual(
            [c["model"] for c in self.models.forecast_calls], ["openrouter/anthropic/claude-opus-5.5"]
        )
        line = self.sealed_records()[0]
        self.assertIn("high value", line["route"])
        self.assertIn("Fable is off (GLASS_EARTH_HIGH_VALUE_MODEL), so Opus", line["route"])
        self.assertTrue(line["combined_by"].startswith("1 forecast from anthropic/claude-opus-5.5"))

    def test_stronger_setup_through_the_variables_routes_to_fable(self):
        # The two repository variables, as the workflows pass them.
        self.set_env(GLASS_EARTH_ENSEMBLE_SIZE="3", GLASS_EARTH_HIGH_VALUE_MODEL="true")
        self.metaculus.posts_page = _page_for("fall-futureeval-2026", {90011})
        bot = self.make_bot(publish=True, skip=True)
        self.assertEqual((bot.ensemble_size, bot.high_value_model), (3, True))
        asyncio.run(bot.forecast_on_tournament("fall-futureeval-2026", return_exceptions=True))
        self.assertEqual(
            [c["model"] for c in self.models.forecast_calls],
            ["openrouter/anthropic/claude-fable-5.1"] * 3,
        )
        line = self.sealed_records()[0]
        self.assertEqual(line["forecaster_model"], "anthropic/claude-fable-5.1")
        self.assertIn("high value", line["route"])
        self.assertIn("the cap has room for Fable", line["route"])
        self.assertIn("3 independent forecasts from anthropic/claude-fable-5.1", line["combined_by"])
        self.assertEqual(len(line["model_forecasts"]), 3)

    def test_high_value_question_falls_back_to_opus_under_a_tight_cap(self):
        self.metaculus.posts_page = _page_for("fall-futureeval-2026", {90011})
        meter = self.make_meter(cap="2.00", reserve="0")
        bot = self.make_bot(publish=True, skip=True, meter=meter, ensemble_size=3, high_value_model=True)
        asyncio.run(bot.forecast_on_tournament("fall-futureeval-2026", return_exceptions=True))
        self.assertEqual(
            {c["model"] for c in self.models.forecast_calls}, {"openrouter/anthropic/claude-opus-5.5"}
        )
        line = self.sealed_records()[0]
        self.assertIn("no room for Fable", line["route"])

    def test_worst_case_counts_every_forecast_in_the_ensemble(self):
        import main

        self.metaculus.posts_page = _page_for("bot-testing-area", {90011})
        question = self.make_bot(publish=False, skip=False).metaculus_client.get_all_open_questions_from_tournament(
            "bot-testing-area"
        )[0]
        worst = {
            n: self.make_bot(publish=False, skip=False, ensemble_size=n)._worst_case_question_cost(
                question, main.DEFAULT_FORECASTER
            )
            for n in (1, 2, 3, 5)
        }
        # Research and summary are the same at every size; each extra
        # forecast adds the same amount (its call and its parsing).
        step = worst[2] - worst[1]
        self.assertGreater(step, Decimal("0.2"))
        self.assertEqual(worst[3] - worst[1], 2 * step)
        self.assertEqual(worst[5] - worst[1], 4 * step)
        # The reservation counts every forecast too: a cap just under five
        # forecasts' reservation (5 x US$0.27, calculated) skips the question;
        # just over, it is forecast five times.
        reservation = Decimal("1.35")
        for cap, forecasts in ((reservation - Decimal("0.0001"), 0), (reservation + Decimal("0.0001"), 5)):
            self.models.forecast_calls.clear()
            meter = self.make_meter(cap=str(cap), reserve="0")
            bot = self.make_bot(publish=False, skip=False, meter=meter, ensemble_size=5)
            self.assertEqual(bot._question_reservation(question), reservation)
            asyncio.run(bot.forecast_questions([question], return_exceptions=True))
            self.assertEqual(len(self.models.forecast_calls), forecasts, str(cap))
            self.assertEqual(len(meter.skips), 1 if forecasts == 0 else 0)

    def test_the_route_counts_the_reservation_not_the_worst_case(self):
        import main

        self.metaculus.posts_page = _page_for("bot-testing-area", {90011})
        probe = self.make_bot(publish=False, skip=False)
        question = probe.metaculus_client.get_all_open_questions_from_tournament("bot-testing-area")[0]
        reservation = probe._question_reservation(question)
        worst = probe._worst_case_question_cost(question, main.DEFAULT_FORECASTER)
        self.assertEqual(reservation, Decimal("0.27"))
        self.assertGreater(worst, Decimal("0.40"))
        # A cap between the two: the question starts and is forecast.
        meter = self.make_meter(cap=str(reservation + Decimal("0.05")), reserve="0")
        bot = self.make_bot(publish=False, skip=False, meter=meter)
        [report] = asyncio.run(bot.forecast_questions([question], return_exceptions=True))
        self.assertFalse(isinstance(report, BaseException))
        self.assertEqual(len(self.models.forecast_calls), 1)
        self.assertEqual(meter.skips, [])
        # A cap just under the reservation: skipped, with what it needed to
        # start and its calculated worst case both recorded.
        Path("forecasts/spend.jsonl").unlink()
        meter = self.make_meter(cap=str(reservation - Decimal("0.0001")), reserve="0")
        bot = self.make_bot(publish=False, skip=False, meter=meter)
        self.assertEqual(asyncio.run(bot.forecast_questions([question], return_exceptions=True)), [])
        [skip] = self.read_jsonl("forecasts/spend.jsonl")
        self.assertEqual(skip["event"], "question_skipped")
        self.assertEqual(skip["reservation_usd"], "0.27")
        self.assertEqual(Decimal(skip["worst_case_usd"]), worst)
        self.assertIn("a question needs US$0.2700 to start", skip["reason"])

    # SYNTHETIC reported costs shaped like the measured 3 November question
    # (observed there: research US$0.0875, forecast US$0.0446).
    MEASURED_SHAPE = {
        "openrouter/openai/gpt-6.1-terra:online": "0.0875",
        "openrouter/anthropic/claude-opus-5.5": "0.0446",
        "openrouter/openai/gpt-6-luna": "0.0003",
    }

    def _two_high_value_questions(self):
        self.metaculus.posts_page = _page_for("fall-futureeval-2026", {90011, 90012})
        probe = self.make_bot(publish=False, skip=False)
        return probe.metaculus_client.get_all_open_questions_from_tournament("fall-futureeval-2026")

    def test_a_question_started_with_only_its_reservation_left_finishes(self):
        import main

        self.models.cost_by_model = dict(self.MEASURED_SHAPE)
        first, second = self._two_high_value_questions()
        probe = self.make_bot(publish=False, skip=False)
        worst = probe._worst_case_question_cost(first, main.DEFAULT_FORECASTER)
        allowance = worst - Decimal("0.27")
        self.assertGreater(allowance, Decimal("0.15"))
        # The cap leaves exactly the first question's reservation (US$0.27).
        meter = self.make_meter(cap="0.27", reserve="0")
        bot = self.make_bot(publish=True, skip=False, meter=meter)
        reports = asyncio.run(bot.forecast_questions([first, second], return_exceptions=True))
        self.assertEqual(len(reports), 1)
        self.assertFalse(isinstance(reports[0], BaseException))
        self.assertEqual([f["question"] for f in self.metaculus.forecasts()], [first.id_of_question])

        records = self.read_jsonl("forecasts/spend.jsonl")
        calls = [r for r in records if r["event"] == "model_call"]
        self.assertFalse([r for r in records if r["event"] == "call_refused"])
        # After research and summary, the forecast's worst case no longer fit
        # under the cap: without the allowance it would have been refused.
        [forecast] = [r for r in calls if r["purpose"] == "forecaster"]
        before = sum(Decimal(r["cap_basis_usd"]) for r in calls[: calls.index(forecast)])
        self.assertGreater(before + Decimal(forecast["worst_case_usd"]), Decimal("0.27"))
        self.assertEqual(forecast["over_cap_allowance_usd"], spend.money_text(allowance))
        # Calls that fitted under the cap are not marked.
        research = [r for r in calls if r["purpose"] == "researcher"][0]
        self.assertNotIn("over_cap_allowance_usd", research)
        # Bounded by the cap plus the allowance, and the next question did not
        # start once the first had used the room.
        self.assertLessEqual(meter.spent_today, Decimal("0.27") + allowance)
        skips = [r for r in records if r["event"] == "question_skipped"]
        self.assertEqual([s["question_id"] for s in skips], [second.id_of_question])
        # Reset: outside a question, a forecast call that would have fitted
        # under the allowance is refused.
        self.assertEqual(meter.over_cap_allowance, Decimal("0"))
        llm = bot.get_llm("forecaster_default", "llm")
        with self.assertRaises(SpendCapReached):
            asyncio.run(llm.invoke("SYNTHETIC prompt: You are a professional forecaster."))

    def test_the_allowance_does_not_let_a_question_go_further_past_the_cap(self):
        # SYNTHETIC: a research call reported at US$0.30, more than its own
        # worst case, leaves too little even under the allowance, so the
        # forecast is refused rather than take the day further past the cap.
        self.models.cost_by_model = dict(self.MEASURED_SHAPE, **{"openrouter/openai/gpt-6.1-terra:online": "0.30"})
        first, _ = self._two_high_value_questions()
        meter = self.make_meter(cap="0.27", reserve="0")
        bot = self.make_bot(publish=True, skip=False, meter=meter)
        [report] = asyncio.run(bot.forecast_questions([first], return_exceptions=True))
        self.assertIsInstance(report, BaseException)
        self.assertEqual(self.metaculus.forecasts(), [])
        records = self.read_jsonl("forecasts/spend.jsonl")
        refused = [r for r in records if r["event"] == "call_refused"]
        self.assertTrue(refused)
        self.assertEqual({r["purpose"] for r in refused}, {"forecaster"})
        self.assertFalse([r for r in records if r["event"] == "model_call" and r["purpose"] == "forecaster"])
        self.assertEqual(meter.over_cap_allowance, Decimal("0"))

    def test_the_reservation_scales_with_the_forecasts_a_question_takes(self):
        from forecasting_tools import BinaryQuestion, ConditionalQuestion

        def binary(n):
            return BinaryQuestion(question_text=f"SYNTHETIC: will made-up event {n} happen?", id_of_question=n)

        question = binary(1)
        conditional = ConditionalQuestion(
            question_text="SYNTHETIC conditional",
            parent=binary(2),
            child=binary(3),
            question_yes=binary(4),
            question_no=binary(5),
        )
        for size in (1, 2, 3, 5):
            bot = self.make_bot(publish=False, skip=False, ensemble_size=size)
            self.assertEqual(bot._question_reservation(question), Decimal("0.27") * size)
            # A conditional question forecasts 4 linked questions per forecast.
            self.assertEqual(bot._question_reservation(conditional), Decimal("0.27") * size * 4)
        # The repository variable changes the amount per forecast, and the
        # route then counts it.
        self.set_env(GLASS_EARTH_QUESTION_RESERVATION_USD="0.35")
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        self.assertEqual(bot._question_reservation(question), Decimal("1.05"))
        self.assertEqual(bot._route_for(question).reserved_usd, Decimal("1.05"))
        with self.assertRaises(ValueError):
            self.make_bot(publish=False, skip=False, question_reservation_usd=Decimal("0"))


if __name__ == "__main__":
    unittest.main()
