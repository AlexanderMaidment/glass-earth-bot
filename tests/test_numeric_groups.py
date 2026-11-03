"""
Numeric and discrete group questions, end to end on SYNTHETIC fixtures: the
bot fetches a page of group posts, unpacks the open sub-questions, forecasts
each one for its own label, and posts a valid CDF of the right length to the
right sub-question.
"""

import asyncio
import unittest
from pathlib import Path

import numpy as np

from support import FAKE_TOKEN, OfflineTestCase, load_fixture


def _assert_valid_cdf(test: unittest.TestCase, cdf: list[float], size: int) -> None:
    test.assertEqual(len(cdf), size)
    steps = np.diff(cdf)
    outcomes = size - 1
    # Rules from the Metaculus API documentation, as recorded on 2026-10-09:
    # each step rises by at least 1% / outcomes and at most 0.2 * 200 / outcomes,
    # and an open bound keeps at least 0.1% of the probability outside it.
    test.assertTrue(all(step >= 0.01 / outcomes - 1e-9 for step in steps))
    test.assertTrue(all(step <= 0.2 * 200 / outcomes + 1e-9 for step in steps))
    test.assertGreaterEqual(cdf[0], 0.001 - 1e-9)
    test.assertLessEqual(cdf[-1], 0.999 + 1e-9)


class NumericGroupTests(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metaculus.posts_page = load_fixture("synthetic_group_posts.json")

    def forecast_tournament(self, bot):
        return asyncio.run(bot.forecast_on_tournament("market-pulse-26q4", return_exceptions=True))

    def test_only_open_sub_questions_are_forecast(self):
        bot = self.make_bot(publish=True)
        reports = self.forecast_tournament(bot)
        failures = [r for r in reports if isinstance(r, BaseException)]
        self.assertEqual(failures, [])
        forecast_ids = sorted(f["question"] for f in self.metaculus.forecasts())
        # 90013 is closed and 90023 is upcoming in the fixture.
        self.assertEqual(forecast_ids, [90011, 90012, 90021, 90022])
        self.assertEqual(
            {r.question.group_question_option for r in reports},
            {"Examplecorp", "Samplesoft", "Revenue (synthetic)", "Gross margin (synthetic)"},
        )

    def test_each_cdf_has_the_sub_questions_size_and_is_valid(self):
        bot = self.make_bot(publish=True)
        self.forecast_tournament(bot)
        sizes = {90011: 201, 90012: 201, 90021: 19, 90022: 42}
        for forecast in self.metaculus.forecasts():
            self.assertEqual(forecast["source"], "api")
            _assert_valid_cdf(self, forecast["continuous_cdf"], sizes[forecast["question"]])

    def test_every_prompt_names_only_its_own_sub_question(self):
        bot = self.make_bot(publish=True)
        self.forecast_tournament(bot)
        labels = ["Examplecorp", "Samplesoft", "Revenue (synthetic)", "Gross margin (synthetic)"]
        forecast_prompts = [c["prompt"] for c in self.models.forecast_calls]
        self.assertEqual(len(forecast_prompts), 4 * 3)
        for prompt in forecast_prompts:
            named = [label for label in labels if f'only about: "{label}"' in prompt]
            self.assertEqual(len(named), 1, prompt[:300])
        research = [
            c["prompt"] for c in self.models.calls if "assistant to a superforecaster" in c["prompt"]
        ]
        self.assertEqual(len(research), 4)
        for label in labels:
            self.assertEqual(sum(f'only about: "{label}"' in p for p in research), 1)

    def test_the_submitted_cdf_is_the_pointwise_median_of_the_members(self):
        from forecasting_tools import NumericDistribution, Percentile

        bot = self.make_bot(publish=False)
        reports = self.forecast_tournament(bot)
        for report in reports:
            self.assertEqual(len(report.prediction.get_cdf()), report.question.cdf_size)
        for question_id in (90011, 90021):
            question = next(r.question for r in reports if r.question.id_of_question == question_id)
            low = question.nominal_lower_bound or question.lower_bound
            high = question.nominal_upper_bound or question.upper_bound
            members = []
            for n in range(3):
                answer = self.models.default_forecast_answer(
                    "synthetic",
                    f"Percentile 10 likely not lower than {low} likely not higher than {high}",
                    n,
                )
                pairs = [line.split(": ") for line in answer.splitlines() if line.startswith("Percentile")]
                members.append(
                    NumericDistribution.from_question(
                        [Percentile(percentile=int(p.split()[1]) / 100, value=float(v)) for p, v in pairs],
                        question,
                    )
                )
            aggregated = asyncio.run(bot._aggregate_predictions(members, question))
            expected = np.median(
                np.array([[pt.percentile for pt in m.get_cdf()] for m in members]), axis=0
            )
            got = [pt.percentile for pt in aggregated.get_cdf()]
            self.assertEqual(len(got), question.cdf_size)
            self.assertTrue(np.allclose(got, expected, atol=2e-3))

    def test_comments_go_on_the_group_post_and_state_the_sub_question(self):
        bot = self.make_bot(publish=True)
        self.forecast_tournament(bot)
        comments = self.metaculus.comments()
        self.assertEqual(len(comments), 4)
        self.assertEqual(sorted({c["on_post"] for c in comments}), [9001, 9002])
        self.assertTrue(all(c["is_private"] for c in comments))
        for comment in comments:
            self.assertIn("(sub-question: ", comment["text"])
            self.assertIn("*Final Prediction*", comment["text"])

    def test_ledger_records_each_sub_question_with_its_label(self):
        bot = self.make_bot(publish=True)
        self.forecast_tournament(bot)
        lines = self.read_jsonl("forecasts/ledger.jsonl")
        self.assertEqual(sorted(line["question_id"] for line in lines), [90011, 90012, 90021, 90022])
        by_id = {line["question_id"]: line for line in lines}
        self.assertEqual(by_id[90021]["group_option"], "Revenue (synthetic)")
        self.assertEqual(by_id[90021]["type"], "discrete")
        self.assertEqual(len(by_id[90021]["final_forecast"]["continuous_cdf"]), 19)
        self.assertEqual(by_id[90011]["forecast_reason"], "first forecast")
        raw = Path("forecasts/ledger.jsonl").read_bytes()
        self.assertNotIn(FAKE_TOKEN.encode(), raw)

    def test_a_group_fetched_by_url_skips_sub_questions_that_are_not_open(self):
        bot = self.make_bot(publish=False, skip=False)
        found = bot.metaculus_client.get_question_by_url(
            "https://www.metaculus.com/questions/9002/", group_question_mode="unpack_subquestions"
        )
        self.assertEqual(len(found), 3)
        reports = asyncio.run(bot.forecast_questions(found, return_exceptions=True))
        self.assertEqual(
            sorted(r.question.id_of_question for r in reports), [90021, 90022]
        )


if __name__ == "__main__":
    unittest.main()
