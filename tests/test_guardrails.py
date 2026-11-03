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
    MultipleChoiceQuestion,
    PredictedOption,
    PredictedOptionList,
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


def multiple_choice(options=("Red", "Green", "Blue"), post_id=302, question_id=3002):
    return MultipleChoiceQuestion(
        question_text="Which SYNTHETIC option wins?",
        id_of_post=post_id,
        id_of_question=question_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        options=list(options),
        **COMMON,
    )


def option_list(names, values):
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=v) for n, v in zip(names, values)]
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


class MultipleChoiceGuardrailTests(unittest.TestCase):
    def assert_floored(self, values, floor):
        self.assertAlmostEqual(sum(values), 1.0, places=12)
        self.assertTrue(all(v >= floor for v in values), (floor, min(values)))

    def assert_check_changes_nothing(self, names, values):
        """forecasting-tools' PredictedOptionList check raises nothing and moves nothing."""
        checked = [o.probability for o in option_list(names, values).predicted_options]
        self.assertEqual(len(checked), len(values))
        for before, after in zip(values, checked):
            self.assertAlmostEqual(before, after, places=12)

    def test_the_floor_is_1_percent_up_to_50_options_then_half_divided_by_the_count(self):
        from guardrails import multiple_choice_floor

        self.assertEqual(multiple_choice_floor(2), 0.01)
        self.assertEqual(multiple_choice_floor(50), 0.01)
        self.assertEqual(multiple_choice_floor(51), 0.5 / 51)
        self.assertEqual(multiple_choice_floor(100), 0.005)
        # Never below forecasting-tools' own 0.1% minimum.
        self.assertEqual(multiple_choice_floor(600), 0.001)

    def test_zeros_get_the_floor_taken_in_proportion_from_the_others(self):
        from guardrails import floor_option_probabilities

        got = floor_option_probabilities([0.7, 0.3, 0.0, 0.0])
        self.assert_floored(got, 0.01)
        self.assertEqual(got[2:], [0.01, 0.01])
        # The two options above the floor give up 2% between them, 7 to 3.
        self.assertAlmostEqual(got[0], 0.7 * 0.98, places=12)
        self.assertAlmostEqual(got[1], 0.3 * 0.98, places=12)

    def test_an_option_the_rescaling_pushes_below_the_floor_is_raised_too(self):
        from guardrails import floor_option_probabilities

        # SYNTHETIC: 0.0101 would fall to 0.0098 when the three zeros are raised.
        got = floor_option_probabilities([0.96, 0.0101, 0.0, 0.0, 0.0, 0.0299])
        self.assert_floored(got, 0.01)
        self.assertEqual(got[1:5], [0.01] * 4)
        self.assertAlmostEqual(got[0] / got[5], 0.96 / 0.0299, places=9)

    def test_already_floored_values_are_kept(self):
        from guardrails import floor_option_probabilities

        for values in ([0.5, 0.3, 0.2], [0.98, 0.01, 0.01]):
            got = floor_option_probabilities(values)
            for before, after in zip(values, got):
                self.assertAlmostEqual(before, after, places=12)

    def test_twenty_options_with_fifteen_zeros_pass_the_template_check_unchanged(self):
        from guardrails import guard_option_list

        names = [f"Option {i}" for i in range(20)]
        raw = [0.96, 0.01, 0.01, 0.01, 0.01] + [0.0] * 15
        guarded = guard_option_list(names, raw)
        values = [o.probability for o in guarded.predicted_options]
        self.assertEqual([o.option_name for o in guarded.predicted_options], names)
        self.assert_floored(values, 0.01)
        # 19 options at the floor and the rest on the first one.
        self.assertAlmostEqual(values[0], 0.81, places=12)
        # Far more than the template check's 0.05 limit on a change, which is
        # why the floor is applied before the list is built.
        self.assertGreater(raw[0] - values[0], 0.05)
        self.assert_check_changes_nothing(names, values)

    def test_more_than_50_options(self):
        from guardrails import guard_option_list

        for count in (51, 60, 120, 400):
            names = [f"Option {i}" for i in range(count)]
            raw = [0.5, 0.3, 0.2] + [0.0] * (count - 3)
            values = [o.probability for o in guard_option_list(names, raw).predicted_options]
            self.assert_floored(values, 0.5 / count)
            # The zeros take half of the probability at most.
            self.assertGreaterEqual(sum(values[:3]), 0.5 - 1e-12)
            self.assert_check_changes_nothing(names, values)

    def test_option_lists_that_cannot_be_guarded_are_refused(self):
        from guardrails import floor_option_probabilities, guard_option_list

        for values in ([0.5, -0.1, 0.6], [0.0, 0.0], [0.5, math.nan], [], [0.001] * 1000):
            with self.assertRaises(ValueError, msg=repr(values[:3])):
                floor_option_probabilities(values)
        with self.assertRaises(ValueError):
            guard_option_list(["A", "B"], [1.0])

    def test_conditional_option_lists_are_guarded(self):
        from guardrails import apply_guardrails

        guarded = apply_guardrails(
            ConditionalPrediction(
                parent=PredictionAffirmed(),
                child=PredictionAffirmed(),
                prediction_yes=option_list("ABC", [0.998, 0.001, 0.001]),
                prediction_no=0.5,
            )
        )
        values = [o.probability for o in guarded.prediction_yes.predicted_options]
        self.assert_floored(values, 0.01)
        self.assertAlmostEqual(values[0], 0.98, places=12)


class MultipleChoiceGuardrailInTheBotTests(OfflineTestCase):
    def test_every_option_is_submitted_stated_and_recorded_at_1_percent_or_more(self):
        lines = {0: "Red: 100%\nGreen: 0%\nBlue: 0%", 1: "Red: 99%\nGreen: 1%\nBlue: 0%"}
        self.models.forecast_answer = lambda model, prompt, n: (
            "Reference class and base rate: SYNTHETIC.\n" + lines[n % 2]
        )
        bot = self.make_bot(publish=True, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([multiple_choice()], return_exceptions=True))
        got = {o.option_name: o.probability for o in report.prediction.predicted_options}
        self.assertAlmostEqual(sum(got.values()), 1.0, places=12)
        self.assertEqual((got["Green"], got["Blue"]), (0.01, 0.01))
        self.assertAlmostEqual(got["Red"], 0.98, places=12)
        self.assertIn("- Green: 1.0%", report.explanation)
        self.assertIn(
            "rescaled to sum to 100%, with every option at least 1%.", report.explanation
        )
        [sent] = self.metaculus.forecasts()
        self.assertEqual(sent["probability_yes_per_category"], got)
        [line] = self.read_jsonl("forecasts/ledger.jsonl")
        self.assertEqual(line["final_forecast"]["probability_yes_per_category"], got)

    def test_sixty_options_with_zero_medians_are_floored_without_an_error(self):
        names = [f"Option {i}" for i in range(60)]
        q = multiple_choice(options=names)
        members = [option_list(names, [0.6, 0.4] + [0.0] * 58)] * 3
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        guarded = asyncio.run(bot._aggregate_predictions(members, q))
        values = [o.probability for o in guarded.predicted_options]
        self.assertAlmostEqual(sum(values), 1.0, places=12)
        self.assertTrue(all(v >= 0.5 / 60 for v in values))
        self.assertIn("with every option at least 0.83%", bot._combination_rule(q))


if __name__ == "__main__":
    unittest.main()
