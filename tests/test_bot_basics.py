"""
The ensemble median, the comment guard, the ledger and dry runs, on SYNTHETIC
questions with SYNTHETIC model answers.
"""

import asyncio
import re
import unittest
from pathlib import Path

from forecasting_tools import (
    BinaryQuestion,
    MetaculusClient,
    MultipleChoiceQuestion,
    PredictedOption,
    PredictedOptionList,
)

from support import FAKE_LEDGER_KEY, FAKE_TOKEN, OfflineTestCase

COMMON = dict(
    background_info="SYNTHETIC background.",
    resolution_criteria="SYNTHETIC criteria.",
    fine_print="SYNTHETIC fine print.",
    tournament_slugs=["bot-testing-area"],
)


def binary(post_id=101, question_id=1001, text="Will the SYNTHETIC binary event happen?"):
    return BinaryQuestion(
        question_text=text,
        id_of_post=post_id,
        id_of_question=question_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        **COMMON,
    )


def multiple_choice():
    return MultipleChoiceQuestion(
        question_text="Which SYNTHETIC colour wins?",
        id_of_post=102,
        id_of_question=1002,
        page_url="https://www.metaculus.com/questions/102/",
        options=["Red", "Green", "Blue"],
        **COMMON,
    )


class DefaultSetupTests(OfflineTestCase):
    def test_default_is_one_forecast_from_opus_stated_plainly(self):
        bot = self.make_bot(publish=False, skip=False)
        self.assertEqual((bot.ensemble_size, bot.high_value_model), (1, False))
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        # The one forecast answers 30%, and that is what is submitted.
        self.assertEqual(report.prediction, 0.3)
        self.assertEqual(len(self.models.forecast_calls), 1)
        self.assertIn("*Final Prediction*: 30.0%", report.explanation)
        self.assertIn(
            "*How the final prediction was made*: 1 forecast from anthropic/claude-opus-5.5, "
            "held between 2% and 98%.",
            report.explanation,
        )
        self.assertNotIn("median of 1", report.explanation)
        self.assertNotIn("member 1 of", report.explanation)
        self.assertIn("## R1: anthropic/claude-opus-5.5 reasoning", report.explanation)
        line = self.read_jsonl("forecasts/dry_run/ledger.jsonl")[0]
        self.assertEqual([f["model"] for f in line["model_forecasts"]], ["anthropic/claude-opus-5.5"])
        self.assertEqual(line["models_without_forecast"], [])

    def test_one_forecast_wording_for_every_question_type(self):
        import main

        bot = self.make_bot(publish=False, skip=False)
        rule = bot._combination_rule(multiple_choice())
        self.assertEqual(
            rule,
            "1 forecast from anthropic/claude-opus-5.5, its option probabilities rescaled to sum to 100%.",
        )
        bot3 = self.make_bot(publish=False, skip=False, ensemble_size=3)
        self.assertEqual(
            bot3._combination_rule(binary()),
            "the median of 3 independent forecasts from anthropic/claude-opus-5.5, "
            "each held between 2% and 98%.",
        )
        self.assertEqual(main._member_label("openrouter/x/y", 1, 1), "x/y")
        self.assertEqual(main._member_label("openrouter/x/y", 2, 3), "x/y (member 2 of 3)")

    def test_unusable_ensemble_settings_are_refused(self):
        from routing import SettingError

        for size in (0, 6, True, 2.0):
            with self.assertRaises(SettingError, msg=repr(size)):
                self.make_bot(publish=False, skip=False, ensemble_size=size)
        self.set_env(GLASS_EARTH_ENSEMBLE_SIZE="three")
        with self.assertRaises(SettingError):
            self.make_bot(publish=False, skip=False)


class EnsembleTests(OfflineTestCase):
    def test_binary_is_the_median_of_three_members(self):
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        # Members answer 30%, 35% and 42%.
        self.assertEqual(report.prediction, 0.35)
        self.assertIn("*Final Prediction*: 35.0%", report.explanation)
        self.assertEqual(len(self.models.forecast_calls), 3)
        for member in (1, 2, 3):
            self.assertIn(f"anthropic/claude-opus-5.5 (member {member} of 3)", report.explanation)

    def test_multiple_choice_is_the_per_option_median_rescaled(self):
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([multiple_choice()], return_exceptions=True))
        got = {o.option_name: o.probability for o in report.prediction.predicted_options}
        # Members: (50,30,20), (60,30,10), (40,40,20). Medians 50, 30, 20 sum to 100.
        self.assertAlmostEqual(got["Red"], 0.5, places=6)
        self.assertAlmostEqual(got["Green"], 0.3, places=6)
        self.assertAlmostEqual(got["Blue"], 0.2, places=6)
        self.assertIn("median of each option's probability", report.explanation)

    def test_median_rescales_when_medians_do_not_sum_to_one(self):
        import main

        def option_list(values):
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=n, probability=v) for n, v in zip("ABC", values)
                ]
            )

        result = main._median_of_option_lists(
            [option_list([0.6, 0.2, 0.2]), option_list([0.2, 0.6, 0.2]), option_list([0.3, 0.3, 0.4])]
        )
        values = [o.probability for o in result.predicted_options]
        # Medians 0.3, 0.3, 0.2 (sum 0.8), rescaled to 0.375, 0.375, 0.25.
        self.assertAlmostEqual(sum(values), 1.0, places=9)
        self.assertAlmostEqual(values[0], 0.375, places=6)
        self.assertAlmostEqual(values[2], 0.25, places=6)

    def test_awkward_binary_is_stated_exactly_as_submitted(self):
        self.models.forecast_answer = lambda model, prompt, n: (
            "Reference class and base rate: SYNTHETIC.\n"
            + f"Probability: {[33.33333, 35.55555, 40.0][n % 3]}%"
        )
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        self.assertEqual(report.prediction, 0.3556)
        self.assertIn("*Final Prediction*: 35.56%", report.explanation)

    def test_one_failed_member_still_gives_a_forecast(self):
        calls = {"n": 0}

        def answer(model, prompt, n):
            calls["n"] += 1
            if n == 0:
                return "SYNTHETIC answer with no probability in it"
            return f"Reference class and base rate: SYNTHETIC.\nProbability: {[0, 35, 42][n % 3]}%"

        self.models.forecast_answer = answer
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        self.assertFalse(isinstance(report, BaseException))
        line = self.read_jsonl("forecasts/dry_run/ledger.jsonl")[0]
        self.assertEqual(len(line["model_forecasts"]), 2)
        self.assertEqual(len(line["models_without_forecast"]), 1)


class CommentAndLedgerTests(OfflineTestCase):
    def test_guard_refuses_a_comment_without_the_final_prediction(self):
        import main

        bot = self.make_bot(publish=False, skip=False)
        real = main.clean_indents
        main.clean_indents = lambda text: real(text).replace("*Final Prediction*", "*Final*")
        try:
            with self.assertRaises(ValueError):
                bot._create_comment(binary(), [], 0.35, 0.0, 0.0)
        finally:
            main.clean_indents = real

    def test_dry_run_writes_only_the_ignored_dry_run_ledger(self):
        bot = self.make_bot(publish=False, skip=False)
        asyncio.run(bot.forecast_questions([binary(), multiple_choice()], return_exceptions=True))
        self.assertEqual(self.metaculus.posts, [])
        self.assertFalse(Path("forecasts/ledger.jsonl").exists())
        lines = self.read_jsonl("forecasts/dry_run/ledger.jsonl")
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line["dry_run"] for line in lines))
        # One forecast per question by default, answering 30%.
        self.assertEqual(lines[0]["final_forecast"], {"probability_yes": 0.3})

    def test_publish_records_exactly_what_was_sent(self):
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_questions([binary(), multiple_choice()], return_exceptions=True))
        # Sealed in the public ledger; the record in each box is exactly what
        # was sent, with each multiple choice option as its index in the
        # question's option order instead of its label.
        records = {r["question_id"]: r for r in self.sealed_records()}
        self.assertEqual(len(records), 2)
        options = multiple_choice().options
        for sent in self.metaculus.forecasts():
            body = {k: v for k, v in sent.items() if k not in ("question", "source")}
            if "probability_yes_per_category" in body:
                body = {
                    "probability_yes_per_category": {
                        str(options.index(label)): share
                        for label, share in body["probability_yes_per_category"].items()
                    }
                }
            self.assertEqual(records[sent["question"]]["final_forecast"], body)
        for comment in self.metaculus.comments():
            self.assertTrue(comment["is_private"])
        for path in ("forecasts/ledger.jsonl", "forecasts/sealed_boxes.jsonl"):
            raw = Path(path).read_bytes()
            self.assertNotIn(FAKE_TOKEN.encode(), raw)
            self.assertNotIn(b"\r\n", raw)
        for line in self.read_jsonl("forecasts/ledger.jsonl"):
            self.assertRegex(line["recorded_at_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_second_run_skips_questions_in_the_ledger(self):
        bot = self.make_bot(publish=True, skip=True)
        asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        before = len(self.metaculus.posts)
        bot2 = self.make_bot(publish=True, skip=True)
        reports = asyncio.run(bot2.forecast_questions([binary()], return_exceptions=True))
        self.assertEqual(reports, [])
        self.assertEqual(len(self.metaculus.posts), before)

    def test_publishing_without_a_ledger_client_is_refused(self):
        import main

        with self.assertRaisesRegex(ValueError, "LedgerClient"):
            main.GlassEarthBot(
                run_mode="x",
                spend_meter=self.make_meter(),
                ledger_key=FAKE_LEDGER_KEY,
                publish_reports_to_metaculus=True,
                metaculus_client=MetaculusClient(),
            )

    def test_question_dates_are_given_to_the_models(self):
        from datetime import datetime, timezone

        q = binary()
        q.close_time = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
        bot = self.make_bot(publish=False, skip=False)
        asyncio.run(bot.forecast_questions([q], return_exceptions=True))
        prompt = self.models.forecast_calls[0]["prompt"]
        self.assertIn("It closes for forecasting on 2026-11-20 12:00 UTC", prompt)
        self.assertTrue(re.search(r"Reference class and base rate", prompt))


if __name__ == "__main__":
    unittest.main()
