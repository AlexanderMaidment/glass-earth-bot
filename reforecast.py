"""
When the bot forecasts a question again.

The FutureEval bot tournaments and MiniBench allow one forecast per question,
so the bot never forecasts their questions twice. Market Pulse is different:
it scores the forecast standing when each question closes, and bots there
must keep their forecasts up to date. For a Market Pulse question the bot
forecasts again when its last forecast is a week old, and once more in the
two days before the question closes, so a fresh forecast is standing at the
close.

The last forecast time is the later of what Metaculus reports for the bot's
own forecast and the bot's own ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class ReforecastPolicy:
    every: timedelta
    final_window: timedelta
    final_min_gap: timedelta


MARKET_PULSE_POLICY = ReforecastPolicy(
    every=timedelta(days=7),
    final_window=timedelta(hours=48),
    final_min_gap=timedelta(hours=20),
)

REFORECAST_POLICIES: dict[str, ReforecastPolicy] = {
    "market-pulse-26q4": MARKET_PULSE_POLICY,
}

# Tournaments whose rules allow one forecast per question. A question in any
# of these is never forecast twice, whatever else it belongs to.
ONE_FORECAST_ONLY = frozenset({"fall-futureeval-2026", "minibench"})


def policy_for(tournament_slugs: list[str]) -> ReforecastPolicy | None:
    if any(slug in ONE_FORECAST_ONLY for slug in tournament_slugs):
        return None
    for slug in tournament_slugs:
        if slug in REFORECAST_POLICIES:
            return REFORECAST_POLICIES[slug]
    return None


def _utc(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def latest(*moments: datetime | None) -> datetime | None:
    known = [_utc(moment) for moment in moments if moment is not None]
    return max(known) if known else None


def forecast_due(
    *,
    now: datetime,
    close_time: datetime | None,
    last_forecast_at: datetime | None,
    policy: ReforecastPolicy,
) -> tuple[bool, str]:
    """(due, reason) for a question under a re-forecast policy."""
    now = _utc(now)
    close_time = _utc(close_time)
    last_forecast_at = _utc(last_forecast_at)
    if close_time is not None and close_time <= now:
        return False, "closed"
    if last_forecast_at is None:
        return True, "first forecast"
    age = now - last_forecast_at
    if age >= policy.every:
        return True, f"scheduled re-forecast: the last forecast is {_hours(age)} old"
    if (
        close_time is not None
        and close_time - now <= policy.final_window
        and age >= policy.final_min_gap
    ):
        return True, (
            f"final refresh before close: closes in {_hours(close_time - now)}, "
            f"the last forecast is {_hours(age)} old"
        )
    return False, f"not due: the last forecast is {_hours(age)} old"


def _hours(span: timedelta) -> str:
    return f"{span.total_seconds() / 3600:.1f} h"
