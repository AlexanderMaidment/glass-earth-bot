"""
The forecast guardrails (guardrails.py), on their own and inside the bot, with
SYNTHETIC questions and SYNTHETIC model answers.
"""

import asyncio
import math
import unittest
from datetime import datetime, timezone

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    DiscreteQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    NumericReport,
    Percentile,
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


BOUNDS = [(True, True), (True, False), (False, True), (False, False)]
PERCENTILES = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]


def numeric(open_lower=True, open_upper=True, *, low=0.0, high=100.0, cdf_size=201,
            zero_point=None, kind=NumericQuestion, post_id=303, question_id=3003):
    return kind(
        question_text="How large will the SYNTHETIC quantity be?",
        id_of_post=post_id,
        id_of_question=question_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        lower_bound=low,
        upper_bound=high,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        zero_point=zero_point,
        cdf_size=cdf_size,
        **COMMON,
    )


def date_question(open_lower=True, open_upper=False):
    return DateQuestion(
        question_text="When will the SYNTHETIC event happen?",
        id_of_post=304,
        id_of_question=3004,
        page_url="https://www.metaculus.com/questions/304/",
        lower_bound=datetime(2027, 1, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        **COMMON,
    )


def members_for(question, middles, width):
    """
    SYNTHETIC forecasts, one per middle: the 10th to 90th percentiles spread
    over middle +/- width, both as fractions of the question's range.
    """
    if isinstance(question, DateQuestion):
        low, high = question.lower_bound.timestamp(), question.upper_bound.timestamp()
    else:
        low, high = question.lower_bound, question.upper_bound
    offsets = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    return [
        NumericDistribution.from_question(
            [
                Percentile(percentile=p, value=low + (high - low) * (middle + width * o))
                for p, o in zip(PERCENTILES, offsets)
            ],
            question,
        )
        for middle in middles
    ]


def heights(distribution):
    return np.array([point.percentile for point in distribution.get_cdf()])


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
        # The public ledger line is sealed; its record is read from its box.
        [record] = self.sealed_records()
        self.assertEqual(record["final_forecast"], {"probability_yes": 0.02})
        # Each forecast is shown as it was used: guarded too.
        self.assertEqual(
            [f["forecast"] for f in record["model_forecasts"]], [{"probability_yes": 0.02}] * 3
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
        recorded = {record["question_id"]: record["final_forecast"] for record in self.sealed_records()}
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
        # The public ledger line is sealed. Its record gives each option by its
        # index in the question's option order (as text), never by its label.
        options = report.question.options
        [record] = self.sealed_records()
        self.assertEqual(record["option_count"], len(options))
        recorded = record["final_forecast"]["probability_yes_per_category"]
        self.assertEqual(recorded, {str(options.index(name)): share for name, share in got.items()})
        # Green and Blue are options 1 and 2: both recorded at the 1% floor.
        self.assertEqual((recorded["1"], recorded["2"]), (0.01, 0.01))

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


class NumericAssertions:
    def assert_guarded(self, cdf, open_lower, open_upper):
        """Metaculus's rules and the guardrail's floors, checked here on their own."""
        cdf = np.asarray(cdf, dtype=float)
        outcomes = len(cdf) - 1
        steps = np.diff(cdf)
        # Increasing, each step holding its share of the even 5% (Metaculus
        # asks for 1% / outcomes), and none over Metaculus's 0.2 * 200 / outcomes.
        self.assertGreaterEqual(steps.min(), 0.05 / outcomes - 1e-9)
        self.assertGreater(steps.min(), 0.01 / outcomes)
        self.assertLessEqual(steps.max(), 0.2 * 200 / outcomes)
        if open_lower:
            self.assertGreaterEqual(cdf[0], 0.01 - 1e-9)
            self.assertLessEqual(cdf[0], 0.999)
        else:
            self.assertEqual(cdf[0], 0.0)
        if open_upper:
            self.assertGreaterEqual(1 - cdf[-1], 0.01 - 1e-9)
            self.assertGreaterEqual(cdf[-1], 0.001)
        else:
            self.assertEqual(cdf[-1], 1.0)


class NumericGuardrailTests(NumericAssertions, OfflineTestCase):
    def combine(self, question, members):
        bot = self.make_bot(publish=False, skip=False, ensemble_size=len(members))
        return asyncio.run(bot._aggregate_predictions(members, question))

    def test_every_bound_combination_meets_the_rules_and_keeps_the_floors(self):
        from guardrails import GuardedNumericDistribution

        shapes = {
            "wide": ([0.5], 0.3),
            "narrow": ([0.5], 0.01),
            "three forecasts": ([0.3, 0.5, 0.7], 0.1),
            "near the lower bound": ([0.12], 0.1),
            "near the upper bound": ([0.88], 0.1),
        }
        for open_lower, open_upper in BOUNDS:
            q = numeric(open_lower, open_upper)
            for name, (middles, width) in shapes.items():
                with self.subTest(bounds=(open_lower, open_upper), shape=name):
                    members = members_for(q, middles, width)
                    combined = self.combine(q, members)
                    self.assertIsInstance(combined, GuardedNumericDistribution)
                    cdf = heights(combined)
                    self.assertEqual(len(cdf), 201)
                    self.assert_guarded(cdf, open_lower, open_upper)
                    # The guardrail's formula on the median of the forecasts' CDFs.
                    median = np.median([heights(m) for m in members], axis=0)
                    below = 0.01 if open_lower else 0.0
                    above = 0.01 if open_upper else 0.0
                    x = np.linspace(0.0, 1.0, 201)
                    expected = (1 - 0.05 - below - above) * median + 0.05 * x + below
                    self.assertTrue(np.allclose(cdf, expected, rtol=0, atol=1e-9))

    def test_the_middle_of_a_narrow_forecast_stays_far_above_its_tails(self):
        for open_lower, open_upper in BOUNDS:
            with self.subTest(bounds=(open_lower, open_upper)):
                q = numeric(open_lower, open_upper)
                members = members_for(q, [0.5], 0.01)
                before = np.diff(heights(members[0]))
                cdf = heights(self.combine(q, members))
                bins = np.diff(cdf)
                # The peak stays where the forecast put it (the template's cap
                # leaves two equal bins there), with most of its height.
                self.assertLessEqual(abs(int(bins.argmax()) - int(before.argmax())), 1)
                self.assertGreater(bins.max(), 0.9 * before.max())
                self.assertGreater(bins.max(), 0.15)
                # The bins at the ends of the range hold far less than the peak.
                edges = max(bins[0], bins[-1])
                self.assertLess(edges, 0.002)
                self.assertGreater(bins.max(), 50 * edges)
                # And beyond an open bound there is far less than at the peak.
                beyond = [m for m, is_open in ((cdf[0], open_lower), (1 - cdf[-1], open_upper)) if is_open]
                for mass in beyond:
                    self.assertLess(mass, bins.max() / 2)
                # The middle of the distribution has not moved.
                x = np.linspace(0.0, 1.0, 201)
                self.assertAlmostEqual(float(np.interp(0.5, cdf, x)), 0.5, delta=0.005)

    def test_discrete_questions(self):
        for open_lower, open_upper in BOUNDS:
            for middles, width in (([0.5], 0.02), ([0.4, 0.5, 0.6], 0.2)):
                with self.subTest(bounds=(open_lower, open_upper), width=width):
                    q = numeric(
                        open_lower, open_upper, low=117.5, high=135.5, cdf_size=19, kind=DiscreteQuestion
                    )
                    cdf = heights(self.combine(q, members_for(q, middles, width)))
                    self.assertEqual(len(cdf), 19)
                    self.assert_guarded(cdf, open_lower, open_upper)

    def test_date_questions(self):
        for open_lower, open_upper in BOUNDS:
            with self.subTest(bounds=(open_lower, open_upper)):
                q = date_question(open_lower, open_upper)
                combined = self.combine(q, members_for(q, [0.4, 0.5, 0.6], 0.05))
                self.assertTrue(combined.is_date)
                self.assert_guarded(heights(combined), open_lower, open_upper)
                self.assertIn("2027-", NumericReport.make_readable_prediction(combined))

    def test_log_scaled_questions(self):
        for open_lower, open_upper in BOUNDS:
            with self.subTest(bounds=(open_lower, open_upper)):
                q = numeric(open_lower, open_upper, low=1.0, high=1000.0, zero_point=0.0)
                cdf = heights(self.combine(q, members_for(q, [0.3, 0.5], 0.05)))
                self.assert_guarded(cdf, open_lower, open_upper)

    def test_the_guardrail_never_applies_twice(self):
        from guardrails import apply_guardrails, guard_numeric

        q = numeric()
        combined = self.combine(q, members_for(q, [0.5], 0.1))
        self.assertIs(guard_numeric(combined), combined)
        self.assertIs(apply_guardrails(combined), combined)

    def test_a_forecast_that_is_not_a_combined_cdf_is_refused(self):
        from guardrails import guard_numeric

        q = numeric()
        [member] = members_for(q, [0.5], 0.1)
        with self.assertRaises(ValueError):
            guard_numeric(member)
        combined = asyncio.run(NumericReport.aggregate_predictions([member], q))
        shifted = combined.model_copy(
            update={
                "declared_percentiles": [
                    Percentile(percentile=p.percentile, value=p.value + 0.1)
                    for p in combined.declared_percentiles
                ]
            }
        )
        with self.assertRaises(ValueError):
            guard_numeric(shifted)

    def test_widen_cdf_by_hand(self):
        from guardrails import widen_cdf

        # Both bounds open: 0.93 F + 0.05 x + 0.01.
        got = widen_cdf([0.1, 0.3, 0.5, 0.7, 0.9], open_lower=True, open_upper=True)
        self.assertTrue(np.allclose(got, [0.103, 0.3015, 0.5, 0.6985, 0.897], rtol=0, atol=1e-12))
        # Both closed: 0.95 F + 0.05 x, exactly 0 and 1 at the ends.
        got = widen_cdf([0.0, 0.2, 0.5, 0.8, 1.0], open_lower=False, open_upper=False)
        self.assertTrue(np.allclose(got, [0.0, 0.2025, 0.5, 0.7975, 1.0], rtol=0, atol=1e-12))
        self.assertEqual((got[0], got[-1]), (0.0, 1.0))
        # A closed bound with probability beyond it is first rescaled, as the
        # template does: here that leaves the even CDF itself.
        got = widen_cdf([0.1, 0.3, 0.5, 0.7, 0.9], open_lower=False, open_upper=False)
        self.assertTrue(np.allclose(got, [0.0, 0.25, 0.5, 0.75, 1.0], rtol=0, atol=1e-12))
        for bad in ([0.5], [0.2, 0.1], [0.0, 1.5], [0.0, math.nan]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                widen_cdf(bad, open_lower=True, open_upper=True)

    def test_cdfs_that_break_a_rule_are_refused(self):
        from guardrails import GuardedNumericDistribution, check_cdf

        q = numeric()
        good = heights(self.combine(q, members_for(q, [0.5], 0.1)))
        check_cdf(good, open_lower=True, open_upper=True)
        index = np.arange(201)
        bad = {
            "a step under the even share": np.where(index == 1, good[0] + 0.0001, good),
            "a step over Metaculus's limit": 0.01 + 0.77 * index / 200 + 0.21 * (index >= 100),
            "too little below an open bound": good - (good[0] - 0.005),
            "too little above an open bound": np.where(index == 200, 0.995, good),
        }
        for name, cdf in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                check_cdf(cdf, open_lower=True, open_upper=True)
        with self.assertRaises(ValueError):
            check_cdf(good, open_lower=False, open_upper=True)
        with self.assertRaises(ValueError):
            check_cdf(good, open_lower=True, open_upper=False)
        combined = self.combine(q, members_for(q, [0.5], 0.1))
        with self.assertRaises(ValueError):
            GuardedNumericDistribution(
                **{
                    **{name: getattr(combined, name) for name in NumericDistribution.model_fields},
                    "open_lower_bound": False,
                }
            )


class NumericGuardrailInTheBotTests(NumericAssertions, OfflineTestCase):
    def test_the_submitted_cdf_is_the_guarded_one_and_the_comment_reads_from_it(self):
        from guardrails import GuardedNumericDistribution

        questions = [
            numeric(open_lower, open_upper, post_id=320 + i, question_id=3020 + i)
            for i, (open_lower, open_upper) in enumerate(BOUNDS)
        ]
        bot = self.make_bot(publish=True, skip=False)
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        self.assertFalse([r for r in reports if isinstance(r, BaseException)])
        sent = {f["question"]: f["continuous_cdf"] for f in self.metaculus.forecasts()}
        recorded = {
            record["question_id"]: record["final_forecast"]["continuous_cdf"]
            for record in self.sealed_records()
        }
        for report in reports:
            q = report.question
            self.assertIsInstance(report.prediction, GuardedNumericDistribution)
            cdf = sent[q.id_of_question]
            self.assertEqual(cdf, list(heights(report.prediction)))
            self.assertEqual(recorded[q.id_of_question], cdf)
            self.assert_guarded(cdf, q.open_lower_bound, q.open_upper_bound)
            # The comment's 10th percentile is read from the CDF that was sent.
            grid = [point.value for point in report.prediction.get_cdf()]
            tenth = round(float(np.interp(0.1, cdf, grid)), 6)
            self.assertIn(f"- 10.00% chance of value below {tenth}", report.explanation)
            self.assertIn(
                "submitted as its cumulative distribution, with 5% of the probability "
                "spread evenly over the range",
                report.explanation,
            )
        rules = {r.question.id_of_question: bot._combination_rule(r.question) for r in reports}
        self.assertTrue(rules[3020].endswith("and at least 1% below it and 1% above it."))
        self.assertTrue(rules[3021].endswith("and at least 1% below it."))
        self.assertTrue(rules[3022].endswith("and at least 1% above it."))
        self.assertTrue(rules[3023].endswith("spread evenly over the range."))

    def test_a_dry_run_records_the_guarded_cdf(self):
        bot = self.make_bot(publish=False, skip=False, ensemble_size=3)
        [report] = asyncio.run(bot.forecast_questions([numeric()], return_exceptions=True))
        self.assertEqual(self.metaculus.posts, [])
        [line] = self.read_jsonl("forecasts/dry_run/ledger.jsonl")
        self.assertEqual(line["final_forecast"]["continuous_cdf"], list(heights(report.prediction)))
        self.assertIn(
            "taken point by point along their cumulative distributions, with 5% of the "
            "probability spread evenly over the range and at least 1% below it and 1% above it.",
            report.explanation,
        )


if __name__ == "__main__":
    unittest.main()
