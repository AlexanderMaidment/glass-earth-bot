"""
Forecast guardrails: limits on every forecast Glass Earth submits.

A log score has no floor. A yes/no forecast of 1% on something that then
happens scores ln(0.01) = -4.6, and a forecast that leaves almost nothing
where the answer lands can score far worse. These guardrails bound that
loss, at a small cost when the forecast is right. main.py applies them in
GlassEarthBot._aggregate_predictions, after the forecasts for a question are
combined, so the comment, the ledger and Metaculus all get the same forecast.

- Yes/no: the probability is held between 2% and 98%.
- Multiple choice: every option gets at least 1% (0.5 divided by the number
  of options, when there are more than 50).
- Numeric, discrete and date: 5% of the probability is spread evenly over
  the range, and at least 1% stays beyond each open bound.

A conditional question gets the guardrail of each of its parts.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from forecasting_tools import (
    ConditionalPrediction,
    NumericDistribution,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    PredictionAffirmed,
)
from pydantic import model_validator

# Yes/no questions: the submitted probability stays between these. The
# template holds each forecast between 1% and 99%.
BINARY_MIN = 0.02
BINARY_MAX = 0.98

# Multiple choice: the least probability any option gets. With more than
# MULTIPLE_CHOICE_MANY_OPTIONS options it is MULTIPLE_CHOICE_MANY_SHARE divided
# by the number of options instead, so the floors never take more than half
# of the probability. forecasting-tools' own floor (PredictedOptionList) is
# TEMPLATE_OPTION_MINIMUM, and the floor never goes below it, so that its
# check leaves the values as they are (that only matters above 500 options).
MULTIPLE_CHOICE_FLOOR = 0.01
MULTIPLE_CHOICE_MANY_OPTIONS = 50
MULTIPLE_CHOICE_MANY_SHARE = 0.5
TEMPLATE_OPTION_MINIMUM = 0.001

# Numeric, discrete and date questions: the three weights mixed into the
# submitted CDF. With x running from 0 to 1 across the question's range and F
# the combined forecast's CDF there, the submitted CDF is
#     G(x) = (1 - U - L - H) * F(x) + U * x + L
# where U is NUMERIC_UNIFORM_WEIGHT, L is NUMERIC_LOWER_TAIL_WEIGHT if the
# lower bound is open (else 0) and H is NUMERIC_UPPER_TAIL_WEIGHT if the upper
# bound is open (else 0). So the forecast keeps 93% of the weight with both
# bounds open (94% with one, 95% with none), 5% is spread evenly over the
# range, and at least 1% stays beyond each open bound. The template's weights,
# which these replace on the submitted CDF, are 0.01, 0.001 and 0.001
# (forecasting-tools NumericDistribution._standardize_cdf).
NUMERIC_UNIFORM_WEIGHT = 0.05
NUMERIC_LOWER_TAIL_WEIGHT = 0.01
NUMERIC_UPPER_TAIL_WEIGHT = 0.01

# Metaculus's rules for a submitted CDF, from its API documentation as
# recorded on 2026-10-09 (tests/test_numeric_groups.py checks them too): each
# step rises by at least METACULUS_MIN_STEP and at most METACULUS_MAX_STEP,
# both divided by the number of outcomes (the points less one), and at an
# open bound the CDF lies between METACULUS_OPEN_BOUND_MIN and 1 minus it.
METACULUS_MIN_STEP = 0.01
METACULUS_MAX_STEP = 0.2 * 200
METACULUS_OPEN_BOUND_MIN = 0.001
# Allowance for floating point rounding when checking a CDF.
CDF_TOLERANCE = 1e-9


def guard_binary(probability: float) -> float:
    """A yes/no probability, held between BINARY_MIN and BINARY_MAX."""
    if isinstance(probability, bool) or not 0 <= probability <= 1:
        # Also refuses NaN, which every comparison fails.
        raise ValueError(f"A probability must be between 0 and 1, not {probability!r}")
    return min(BINARY_MAX, max(BINARY_MIN, float(probability)))


def multiple_choice_floor(option_count: int) -> float:
    """The least probability each option gets, with option_count options."""
    if option_count < 1:
        raise ValueError("A multiple choice forecast needs at least one option")
    if option_count > MULTIPLE_CHOICE_MANY_OPTIONS:
        floor = MULTIPLE_CHOICE_MANY_SHARE / option_count
    else:
        floor = MULTIPLE_CHOICE_FLOOR
    return max(floor, TEMPLATE_OPTION_MINIMUM)


def floor_option_probabilities(probabilities: Sequence[float]) -> list[float]:
    """
    The multiple choice guardrail on option probabilities, in their order.
    They are rescaled to sum to 1. Options below the floor are raised to it,
    and the extra is taken from the options above the floor in proportion to
    their probabilities, so the sum stays 1. An option that this would push
    below the floor is raised to it as well.
    """
    count = len(probabilities)
    floor = multiple_choice_floor(count)
    if floor * count >= 1:
        raise ValueError(f"{count} options cannot each get a probability of {floor}")
    values = [float(p) for p in probabilities]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"Option probabilities must be finite and not negative: {values}")
    total = sum(values)
    if total <= 0:
        raise ValueError("The option probabilities sum to zero")
    values = [v / total for v in values]
    raised: set[int] = set()
    while True:
        # The options not raised share what the raised ones leave, in
        # proportion to their probabilities. At least one option is never
        # raised: the largest gets at least the average, which is above the
        # floor because floor * count < 1.
        left = 1.0 - floor * len(raised)
        rest = sum(v for i, v in enumerate(values) if i not in raised)
        floored = [floor if i in raised else v * left / rest for i, v in enumerate(values)]
        newly = {i for i, v in enumerate(floored) if i not in raised and v < floor}
        if not newly:
            break
        raised |= newly
    # Floating point: the largest option takes the rounding remainder, as in
    # forecasting-tools, so the probabilities sum to 1.
    largest = max(range(count), key=floored.__getitem__)
    floored[largest] = 1.0 - sum(v for i, v in enumerate(floored) if i != largest)
    return floored


def guard_option_list(
    names: Sequence[str], probabilities: Sequence[float]
) -> PredictedOptionList:
    """
    The multiple choice guardrail, as the option list that is submitted. The
    list is built from values already floored, so forecasting-tools' own check
    in PredictedOptionList (a 0.1% minimum, rescaling, and no option moved by
    more than 0.05) leaves them as they are.
    """
    if len(names) != len(probabilities):
        raise ValueError("Each option needs exactly one probability")
    floored = floor_option_probabilities(probabilities)
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=name, probability=value)
            for name, value in zip(names, floored)
        ]
    )


def widen_cdf(cdf: Sequence[float], *, open_lower: bool, open_upper: bool) -> list[float]:
    """
    The numeric guardrail on a CDF given at evenly spaced points across the
    question's range, point i at x = i / (points - 1). A closed bound first
    gets the template's treatment: the CDF is rescaled to be 0 at a closed
    lower bound and 1 at a closed upper bound. Then the weights above apply.

    Each step of the result is (1 - U - L - H), at most 0.95, times the step
    it replaces, plus U / outcomes. So every step rises by at least
    U / outcomes, five times Metaculus's minimum. The template already caps
    each step of the CDFs it combines at 95% of Metaculus's maximum, and
    0.95 of such a step plus U / outcomes stays under the maximum.
    check_cdf confirms both.
    """
    heights = np.asarray(cdf, dtype=float)
    if heights.ndim != 1 or len(heights) < 2:
        raise ValueError("A CDF needs at least two points")
    if not np.all(np.isfinite(heights)) or heights.min() < 0 or heights.max() > 1:
        raise ValueError("A CDF's values must lie between 0 and 1")
    if np.any(np.diff(heights) < 0):
        raise ValueError("A CDF must never decrease")
    start = 0.0 if open_lower else heights[0]
    end = 1.0 if open_upper else heights[-1]
    if end <= start:
        raise ValueError("The CDF puts no probability inside the range")
    below = NUMERIC_LOWER_TAIL_WEIGHT if open_lower else 0.0
    above = NUMERIC_UPPER_TAIL_WEIGHT if open_upper else 0.0
    forecast_weight = 1.0 - NUMERIC_UNIFORM_WEIGHT - below - above
    x = np.linspace(0.0, 1.0, len(heights))
    widened = (
        forecast_weight * (heights - start) / (end - start)
        + NUMERIC_UNIFORM_WEIGHT * x
        + below
    )
    if not open_lower:
        widened[0] = 0.0
    if not open_upper:
        widened[-1] = 1.0
    # As the template does, round away floating point noise.
    return np.round(widened, 10).tolist()


def check_cdf(cdf: Sequence[float], *, open_lower: bool, open_upper: bool) -> None:
    """
    Raise ValueError unless the CDF is one Metaculus accepts and it keeps the
    guardrail's floors: every step at least NUMERIC_UNIFORM_WEIGHT / outcomes
    (Metaculus asks for METACULUS_MIN_STEP / outcomes) and at most
    METACULUS_MAX_STEP / outcomes; exactly 0 at a closed lower bound and 1 at
    a closed upper bound; and at an open bound, inside Metaculus's limits
    with at least the tail weight beyond it.
    """
    values = np.asarray(cdf, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("A CDF needs at least two points")
    if not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 1:
        raise ValueError("A CDF's values must lie between 0 and 1")
    outcomes = len(values) - 1
    steps = np.diff(values)
    least = max(NUMERIC_UNIFORM_WEIGHT, METACULUS_MIN_STEP) / outcomes
    if steps.min() < least - CDF_TOLERANCE:
        raise ValueError(f"A step of {steps.min():.6g} is under the least allowed, {least:.6g}")
    most = METACULUS_MAX_STEP / outcomes
    if steps.max() > most + CDF_TOLERANCE:
        raise ValueError(f"A step of {steps.max():.6g} is over Metaculus's limit, {most:.6g}")
    if open_lower:
        lowest = max(NUMERIC_LOWER_TAIL_WEIGHT, METACULUS_OPEN_BOUND_MIN)
        if not lowest - CDF_TOLERANCE <= values[0] <= 1 - METACULUS_OPEN_BOUND_MIN:
            raise ValueError(f"The CDF at the open lower bound, {values[0]}, is outside the limits")
    elif values[0] != 0.0:
        raise ValueError(f"The CDF at a closed lower bound must be 0, not {values[0]}")
    if open_upper:
        highest = 1 - max(NUMERIC_UPPER_TAIL_WEIGHT, METACULUS_OPEN_BOUND_MIN)
        if not METACULUS_OPEN_BOUND_MIN <= values[-1] <= highest + CDF_TOLERANCE:
            raise ValueError(f"The CDF at the open upper bound, {values[-1]}, is outside the limits")
    elif values[-1] != 1.0:
        raise ValueError(f"The CDF at a closed upper bound must be 1, not {values[-1]}")


class GuardedNumericDistribution(NumericDistribution):
    """
    A numeric, discrete or date forecast after the guardrail. Its declared
    percentiles are the whole CDF that is submitted, one for each point of
    the question's grid, and get_cdf returns them as they are: a plain
    NumericDistribution would apply the template's weights to them again
    when it is published. It cannot be made unless check_cdf passes.
    """

    @model_validator(mode="after")
    def check_guarded_cdf(self) -> GuardedNumericDistribution:
        if self.cdf_size is None or len(self.declared_percentiles) != self.cdf_size:
            raise ValueError("A guarded CDF needs one point for each point of the question's grid")
        check_cdf(
            [point.percentile for point in self.declared_percentiles],
            open_lower=self.open_lower_bound,
            open_upper=self.open_upper_bound,
        )
        return self

    def get_cdf(self) -> list[Percentile]:
        return [
            Percentile(percentile=point.percentile, value=point.value)
            for point in self.declared_percentiles
        ]


def guard_numeric(distribution: NumericDistribution) -> GuardedNumericDistribution:
    """
    The numeric guardrail on a combined numeric, discrete or date forecast.
    forecasting-tools combines the forecasts into one CDF with a point for
    each point of the question's grid (each forecast's own CDF already
    carries the template's weights). Publishing would then apply the
    template's weights a second time; the guardrail's weights take the place
    of that second pass, and the result is submitted as it is. A forecast
    that is already guarded is returned unchanged, so the guardrail never
    applies twice.
    """
    if isinstance(distribution, GuardedNumericDistribution):
        return distribution
    size = distribution.cdf_size
    points = distribution.declared_percentiles
    if size is None or len(points) != size:
        raise ValueError(
            "The numeric guardrail needs a combined forecast, with one point for each point of the question's grid"
        )
    # The question's grid, as forecasting-tools lays it out for every CDF.
    grid = [distribution._cdf_location_to_nominal_location(i / (size - 1)) for i in range(size)]
    span = abs(distribution.upper_bound - distribution.lower_bound)
    if not np.allclose([point.value for point in points], grid, rtol=1e-9, atol=1e-9 * span):
        raise ValueError("The forecast's points are not on the question's grid")
    widened = widen_cdf(
        [point.percentile for point in points],
        open_lower=distribution.open_lower_bound,
        open_upper=distribution.open_upper_bound,
    )
    return GuardedNumericDistribution(
        declared_percentiles=[
            Percentile(percentile=height, value=point.value)
            for height, point in zip(widened, points)
        ],
        open_upper_bound=distribution.open_upper_bound,
        open_lower_bound=distribution.open_lower_bound,
        upper_bound=distribution.upper_bound,
        lower_bound=distribution.lower_bound,
        zero_point=distribution.zero_point,
        cdf_size=size,
        standardize_cdf=False,
        is_date=distribution.is_date,
    )


def apply_guardrails(prediction: Any) -> Any:
    """
    The guardrail for the forecast's type: a yes/no probability, an option
    list, or a numeric, discrete or date CDF. A conditional forecast gets the
    guardrail of each part's type; a part that only reaffirms an earlier
    forecast is left as it is. Any other kind of forecast is refused rather
    than submitted without a guardrail.
    """
    if isinstance(prediction, ConditionalPrediction):
        return ConditionalPrediction(
            parent=apply_guardrails(prediction.parent),
            child=apply_guardrails(prediction.child),
            prediction_yes=apply_guardrails(prediction.prediction_yes),
            prediction_no=apply_guardrails(prediction.prediction_no),
        )
    if isinstance(prediction, PredictionAffirmed):
        return prediction
    if isinstance(prediction, PredictedOptionList):
        options = prediction.predicted_options
        return guard_option_list(
            [option.option_name for option in options],
            [option.probability for option in options],
        )
    if isinstance(prediction, NumericDistribution):
        return guard_numeric(prediction)
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return guard_binary(prediction)
    raise TypeError(f"There is no guardrail for a {type(prediction).__name__} forecast")
