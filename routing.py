"""
Which model forecasts a question, and whether a question is forecast at all.

- Claude Opus 5.5 forecasts by default.
- Claude Fable 5.1 forecasts high-value questions, but only while the daily
  cap still has room for a Fable question at its worst case plus one more
  default question after it.
- A low-value question is skipped rather than spend into the reserve that is
  kept for high-value questions.
- A question that could take the day over the cap even with Opus is skipped.

High value means a prize value per question of at least
HIGH_VALUE_THRESHOLD_USD: the tournament's prize pool divided by its planned
number of questions, times the question's weight. Questions in tournaments
without prizes for bots (the bot testing area, the Metaculus Cup) have no
prize value.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from spend import ZERO, money_display

DEFAULT_FORECASTER = "openrouter/anthropic/claude-opus-5.5"
HIGH_VALUE_FORECASTER = "openrouter/anthropic/claude-fable-5.1"

# Prize pool and planned question count, as the tournament pages and the
# FutureEval announcement stated them when read on 2026-11-03 (observed):
# - Fall 2026 FutureEval: US$50,000, 300 to 400 questions (350 used).
# - Market Pulse 26Q4: US$7,500, 65 questions planned.
# - MiniBench: US$1,000 per two-week round of about 60 questions. Its
#   questions sit in a series whose slug changes each round, so they match no
#   entry here and count as low value, which is the intent.
PRIZE_PER_QUESTION_USD: dict[str, Decimal] = {
    "fall-futureeval-2026": Decimal(50000) / Decimal(350),
    "market-pulse-26q4": Decimal(7500) / Decimal(65),
}
HIGH_VALUE_THRESHOLD_USD = Decimal(100)


def prize_value_usd(tournament_slugs: list[str], question_weight: float | None) -> Decimal:
    """The prize money a question carries, before anyone's share of it."""
    base = max(
        (PRIZE_PER_QUESTION_USD[slug] for slug in tournament_slugs if slug in PRIZE_PER_QUESTION_USD),
        default=ZERO,
    )
    if question_weight is None:
        return base
    # The weight comes from Metaculus as a float; its text form is exact enough.
    return base * Decimal(repr(float(question_weight)))


def is_high_value(value_usd: Decimal) -> bool:
    return value_usd >= HIGH_VALUE_THRESHOLD_USD


@dataclass(frozen=True)
class Route:
    action: Literal["forecast", "skip"]
    tier: Literal["high_value", "default"] | None
    model: str | None
    worst_case_usd: Decimal
    reason: str


def choose_route(
    *,
    value_usd: Decimal,
    worst_case_default_usd: Decimal,
    worst_case_high_value_usd: Decimal,
    spent_today_usd: Decimal,
    cap_usd: Decimal,
    reserve_usd: Decimal,
) -> Route:
    """Pick the model for one question from what is left of today's cap."""
    for amount in (
        value_usd,
        worst_case_default_usd,
        worst_case_high_value_usd,
        spent_today_usd,
        cap_usd,
        reserve_usd,
    ):
        if not isinstance(amount, Decimal):
            raise TypeError("Money must be Decimal")
    left = cap_usd - spent_today_usd
    high = is_high_value(value_usd)
    value_text = f"prize value US${money_display(value_usd)} per question"
    if high:
        if worst_case_high_value_usd + worst_case_default_usd <= left:
            return Route(
                "forecast",
                "high_value",
                HIGH_VALUE_FORECASTER,
                worst_case_high_value_usd,
                f"high value ({value_text}); the cap has room for Fable",
            )
        if worst_case_default_usd <= left:
            return Route(
                "forecast",
                "default",
                DEFAULT_FORECASTER,
                worst_case_default_usd,
                f"high value ({value_text}), but the cap has no room for Fable, so Opus",
            )
        return Route(
            "skip",
            None,
            None,
            worst_case_default_usd,
            f"daily cap: US${money_display(left)} left, a question can cost up to "
            f"US${money_display(worst_case_default_usd)}",
        )
    if worst_case_default_usd + reserve_usd <= left:
        return Route(
            "forecast",
            "default",
            DEFAULT_FORECASTER,
            worst_case_default_usd,
            f"default ({value_text})",
        )
    reason = (
        f"daily cap: US${money_display(left)} left, a question can cost up to "
        f"US${money_display(worst_case_default_usd)}"
    )
    if reserve_usd > ZERO:
        reason += f", and US${money_display(reserve_usd)} is kept for high-value questions"
    return Route("skip", None, None, worst_case_default_usd, reason)
