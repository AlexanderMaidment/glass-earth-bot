"""
Which model forecasts a question, how many forecasts it gets, and whether a
question is forecast at all.

- Claude Opus 5.5 forecasts, once per question by default.
- Claude Fable 5.1 forecasts high-value questions only when that is switched
  on (GLASS_EARTH_HIGH_VALUE_MODEL=true), and then only while the daily cap
  still has room for a Fable question at its worst case plus one more default
  question after it.
- A low-value question is skipped rather than spend into the reserve that is
  kept for high-value questions.
- A question that could take the day over the cap even with Opus is skipped.

High value means a prize value per question of at least
HIGH_VALUE_THRESHOLD_USD: the tournament's prize pool divided by its planned
number of questions, times the question's weight. Questions in tournaments
without prizes for bots (the bot testing area, the Metaculus Cup) have no
prize value.

Both settings can be changed without editing code, through environment
variables that GitHub repository variables set (like the daily cap in
spend.py). A value the bot cannot use stops the run; it never falls back
silently.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from spend import ZERO, money_display

DEFAULT_FORECASTER = "openrouter/anthropic/claude-opus-5.5"
HIGH_VALUE_FORECASTER = "openrouter/anthropic/claude-fable-5.1"

# How many independent forecasts each question gets from its model; the
# median is submitted. One by default, which keeps a typical question near
# US$0.12 with Opus (calculated from an assumption; see GLASS_EARTH.md).
ENSEMBLE_SIZE_ENV = "GLASS_EARTH_ENSEMBLE_SIZE"
DEFAULT_ENSEMBLE_SIZE = 1
MAX_ENSEMBLE_SIZE = 5
# "true" sends high-value questions to Fable while the cap has room; off by
# default.
HIGH_VALUE_MODEL_ENV = "GLASS_EARTH_HIGH_VALUE_MODEL"
DEFAULT_HIGH_VALUE_MODEL = False


class SettingError(ValueError):
    """A setting has a value the bot cannot use, so the run stops."""


def check_ensemble_size(size: object) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise SettingError(f"The ensemble size must be a whole number, not {size!r}")
    if not 1 <= size <= MAX_ENSEMBLE_SIZE:
        raise SettingError(
            f"The ensemble size must be from 1 to {MAX_ENSEMBLE_SIZE} forecasts per question, not {size}"
        )
    return size


def ensemble_size_from_env(override: str | None = None) -> int:
    """Forecasts per question, from the environment or the default (1)."""
    raw = override if override is not None else os.getenv(ENSEMBLE_SIZE_ENV)
    # GitHub passes an unset repository variable as an empty string.
    if raw is None or raw.strip() == "":
        return DEFAULT_ENSEMBLE_SIZE
    text = raw.strip()
    if not re.fullmatch(r"[0-9]+", text) or not 1 <= int(text) <= MAX_ENSEMBLE_SIZE:
        raise SettingError(
            f"{ENSEMBLE_SIZE_ENV} must be a whole number from 1 to {MAX_ENSEMBLE_SIZE} "
            f"(forecasts per question), but it is {raw!r}. Leave it unset for the "
            f"default of {DEFAULT_ENSEMBLE_SIZE}."
        )
    return int(text)


def high_value_model_from_env(override: str | None = None) -> bool:
    """Whether high-value questions may go to Fable (default: no)."""
    raw = override if override is not None else os.getenv(HIGH_VALUE_MODEL_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_HIGH_VALUE_MODEL
    text = raw.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise SettingError(
        f"{HIGH_VALUE_MODEL_ENV} must be true or false, but it is {raw!r}. "
        "Leave it unset to keep Fable off."
    )

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
    worst_case_high_value_usd: Decimal | None,
    spent_today_usd: Decimal,
    cap_usd: Decimal,
    reserve_usd: Decimal,
    use_high_value_model: bool,
) -> Route:
    """
    Pick the model for one question from what is left of today's cap. With
    use_high_value_model off, high-value questions go to Opus, and
    worst_case_high_value_usd may be None.
    """
    amounts = [value_usd, worst_case_default_usd, spent_today_usd, cap_usd, reserve_usd]
    if worst_case_high_value_usd is not None:
        amounts.append(worst_case_high_value_usd)
    for amount in amounts:
        if not isinstance(amount, Decimal):
            raise TypeError("Money must be Decimal")
    if use_high_value_model and worst_case_high_value_usd is None:
        raise ValueError("Routing to Fable needs Fable's worst case for the question")
    left = cap_usd - spent_today_usd
    high = is_high_value(value_usd)
    value_text = f"prize value US${money_display(value_usd)} per question"
    if high:
        if use_high_value_model:
            assert worst_case_high_value_usd is not None
            if worst_case_high_value_usd + worst_case_default_usd <= left:
                return Route(
                    "forecast",
                    "high_value",
                    HIGH_VALUE_FORECASTER,
                    worst_case_high_value_usd,
                    f"high value ({value_text}); the cap has room for Fable",
                )
            fallback = f"high value ({value_text}), but the cap has no room for Fable, so Opus"
        else:
            fallback = f"high value ({value_text}); Fable is off ({HIGH_VALUE_MODEL_ENV}), so Opus"
        if worst_case_default_usd <= left:
            return Route(
                "forecast",
                "default",
                DEFAULT_FORECASTER,
                worst_case_default_usd,
                fallback,
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
