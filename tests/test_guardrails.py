"""
The forecast guardrails (guardrails.py), on their own and inside the bot, with
SYNTHETIC questions and SYNTHETIC model answers.
"""

import asyncio
import math
import unittest

from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    PredictionAffirmed,
)

from support import OfflineTestCase

COMMON = dict(
    background_info="SYNTHETIC background.",
    resolution_criteria="SYNTHETIC criteria.",
    fine_print="SYNTHETIC fine print.",
    tournament_slugs=["bot-testing-area"],
)


def binary(post_id=301, question_id=3001, text="Will the SYNTHETIC binary event happen?"):
    return BinaryQuestion(
        question_text=text,
        id_of_post=post_id,
        id_of_question=question_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        **COMMON,
    )


def conditional():
    """A SYNTHETIC conditional question whose four parts are yes/no questions."""
    parts = {
        "parent": binary(311, 3011, "Will the SYNTHETIC parent event happen?"),
        "child": binary(312, 3012, "Will the SYNTHETIC child event happen?"),
        "question_yes": binary(313, 3013, "SYNTHETIC child, if the parent resolves yes"),
        "question_no": binary(314, 3014, "SYNTHETIC child, if the parent resolves no"),
    }
    return ConditionalQuestion(
        question_text="SYNTHETIC conditional: the child event, given the parent event",
        id_of_post=310,
        id_of_question=3010,
        page_url="https://www.metaculus.com/questions/310/",
        **parts,
        **COMMON,
    )


def answer_with(percent_for_text):
    """A SYNTHETIC forecaster that answers by the question it is asked."""

    def answer(model, prompt, n):
        for text, percents in percent_for_text.items():
            if text in prompt:
                return f"Reference class and base rate: SYNTHETIC.\nProbability: {percents[n % len(percents)]}%"
        raise AssertionError(f"No SYNTHETIC answer for: {prompt[:200]}")

    return answer


class BinaryGuardrailTests(unittest.TestCase):
    def test_values_are_held_between_2_and_98_percent(self):
        from guardrails import guard_binary

        cases = {0: 0.02, 0.0: 0.02, 0.01: 0.02, 0.02: 0.02, 0.0201: 0.0201, 0.5: 0.5, 0.98: 0.98, 0.99: 0.98, 1: 0.98}
        for value, expected in cases.items():
            self.assertEqual(guard_binary(value), expected, msg=repr(value))
            self.assertIsInstance(guard_binary(value), float)

    def test_values_that_are_not_probabilities_are_refused(self):
        from guardrails import guard_binary

        for value in (-0.01, 1.01, math.nan, math.inf, True):
            with self.assertRaises(ValueError, msg=repr(value)):
                guard_binary(value)

    def test_conditional_parts_are_guarded_and_a_reaffirmed_part_is_kept(self):
        from guardrails import apply_guardrails

        guarded = apply_guardrails(
            ConditionalPrediction(
                parent=PredictionAffirmed(),
                child=0.995,
                prediction_yes=0.0,
                prediction_no=1.0,
            )
        )
        self.assertIsInstance(guarded.parent, PredictionAffirmed)
        self.assertEqual(
            (guarded.child, guarded.prediction_yes, guarded.prediction_no), (0.98, 0.02, 0.98)
        )


class BinaryGuardrailInTheBotTests(OfflineTestCase):
    def test_the_median_is_guarded_after_aggregation(self):
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        q = binary()
        # SYNTHETIC member values given straight to the aggregation step, as
        # if they had not been guarded one by one.
        for members, expected in (
            ([0.0, 0.001, 0.005], 0.02),
            ([0.01, 0.015, 0.0199], 0.02),
            ([0.995, 1.0, 0.999], 0.98),
            ([0.3, 0.35, 0.42], 0.35),
        ):
            self.assertEqual(asyncio.run(bot._aggregate_predictions(members, q)), expected, msg=members)

    def test_extreme_forecasts_are_submitted_stated_and_recorded_at_2_percent(self):
        # The members answer 0%, 0.5% and 1%.
        self.models.forecast_answer = answer_with({"SYNTHETIC binary event": [0, 0.5, 1]})
        bot = self.make_bot(publish=True, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        self.assertEqual(report.prediction, 0.02)
        self.assertIn("*Final Prediction*: 2.0%", report.explanation)
        self.assertIn(
            "*How the final prediction was made*: the median of 3 independent forecasts from "
            "anthropic/claude-opus-5.5, each held between 2% and 98%.",
            report.explanation,
        )
        [sent] = self.metaculus.forecasts()
        self.assertEqual(sent["probability_yes"], 0.02)
        [line] = self.read_jsonl("forecasts/ledger.jsonl")
        self.assertEqual(line["final_forecast"], {"probability_yes": 0.02})
        # Each forecast is shown as it was used: guarded too.
        self.assertEqual(
            [f["forecast"] for f in line["model_forecasts"]], [{"probability_yes": 0.02}] * 3
        )

    def test_one_forecast_of_100_percent_is_submitted_at_98_percent(self):
        self.models.forecast_answer = answer_with({"SYNTHETIC binary event": [100]})
        bot = self.make_bot(publish=True, skip=False)
        [report] = asyncio.run(bot.forecast_questions([binary()], return_exceptions=True))
        self.assertEqual(report.prediction, 0.98)
        self.assertIn("*Final Prediction*: 98.0%", report.explanation)
        self.assertEqual(self.metaculus.forecasts()[0]["probability_yes"], 0.98)

    def test_conditional_question_submits_guarded_yes_no_forecasts(self):
        self.models.forecast_answer = answer_with(
            {
                "SYNTHETIC parent event happen": [50],
                "SYNTHETIC child event happen": [40],
                "if the parent resolves yes": [0, 0, 1],
                "if the parent resolves no": [100, 99.5, 100],
            }
        )
        # Four forecasts per member: a SYNTHETIC cap with room for the worst case.
        meter = self.make_meter(cap="10.00", reserve="2.00")
        bot = self.make_bot(publish=True, skip=False, ensemble_size=3, meter=meter)
        [report] = asyncio.run(bot.forecast_questions([conditional()], return_exceptions=True))
        self.assertFalse(isinstance(report, BaseException), report)
        self.assertEqual(report.prediction.prediction_yes, 0.02)
        self.assertEqual(report.prediction.prediction_no, 0.98)
        sent = {f["question"]: f["probability_yes"] for f in self.metaculus.forecasts()}
        # forecasting-tools publishes only the two conditional branches.
        self.assertEqual(sent, {3013: 0.02, 3014: 0.98})
        self.assertIn("each yes/no part held between 2% and 98%", report.explanation)
        recorded = {line["question_id"]: line["final_forecast"] for line in self.read_jsonl("forecasts/ledger.jsonl")}
        self.assertEqual(recorded, {3013: {"probability_yes": 0.02}, 3014: {"probability_yes": 0.98}})


if __name__ == "__main__":
    unittest.main()
