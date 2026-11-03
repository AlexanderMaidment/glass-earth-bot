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

A conditional question gets the guardrail of each of its parts.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from forecasting_tools import (
    ConditionalPrediction,
    PredictedOption,
    PredictedOptionList,
    PredictionAffirmed,
)

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


def apply_guardrails(prediction: Any) -> Any:
    """
    The guardrail for the forecast's type. A conditional forecast gets the
    guardrail of each part's type; a part that only reaffirms an earlier
    forecast is left as it is. Types without a guardrail yet pass unchanged.
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
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return guard_binary(prediction)
    return prediction
