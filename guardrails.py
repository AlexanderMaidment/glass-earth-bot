"""
Forecast guardrails: limits on every forecast Glass Earth submits.

A log score has no floor. A yes/no forecast of 1% on something that then
happens scores ln(0.01) = -4.6, and a forecast that leaves almost nothing
where the answer lands can score far worse. These guardrails bound that
loss, at a small cost when the forecast is right. main.py applies them in
GlassEarthBot._aggregate_predictions, after the forecasts for a question are
combined, so the comment, the ledger and Metaculus all get the same forecast.

- Yes/no: the probability is held between 2% and 98%.

A conditional question gets the guardrail of each of its parts.
"""

from __future__ import annotations

from typing import Any

from forecasting_tools import ConditionalPrediction, PredictionAffirmed

# Yes/no questions: the submitted probability stays between these. The
# template holds each forecast between 1% and 99%.
BINARY_MIN = 0.02
BINARY_MAX = 0.98


def guard_binary(probability: float) -> float:
    """A yes/no probability, held between BINARY_MIN and BINARY_MAX."""
    if isinstance(probability, bool) or not 0 <= probability <= 1:
        # Also refuses NaN, which every comparison fails.
        raise ValueError(f"A probability must be between 0 and 1, not {probability!r}")
    return min(BINARY_MAX, max(BINARY_MIN, float(probability)))


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
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return guard_binary(prediction)
    return prediction
