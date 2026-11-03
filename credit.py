"""
The OpenRouter key's credit, read once at the start of each run.

OpenRouter's key endpoint (GET https://openrouter.ai/api/v1/key, free) reports
the key's spending limit and what is left of it (`limit_remaining`). The bot
spends nothing on this check and never stops because of it: the key's own
limit is the hard stop on spending.

When less than WARN_BELOW_USD is left, the run says so in its log, as a
GitHub Actions warning and on the run's summary page. A run on GitHub that
can publish then ends with a failing exit code after all its forecasting is
done, at most once per UTC day, so that GitHub emails the repository owner.
forecasts/credit_warnings.jsonl records each such day, and the workflows
commit it even though the run failed.

If the check fails, the credit is Unknown: logged as Unknown, never treated
as zero, and never a reason to fail the run.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from spend import UNKNOWN, github_run_url, money_display, money_text, to_decimal, utc_day

logger = logging.getLogger(__name__)

KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
FETCH_TIMEOUT_SECONDS = 20
# Warn when less than this is left on the key.
WARN_BELOW_USD = Decimal("5.00")
WARNINGS_PATH = Path("forecasts/credit_warnings.jsonl")
WARNINGS_VERSION = 1

KeyInfoFetch = Callable[[str], dict[str, Any]]


def fetch_key_info(api_key: str) -> dict[str, Any]:
    """
    OpenRouter's report on the key (free). Numbers come back as Decimal, never
    as float. The offline tests replace this function, so they never call it.
    """
    request = urllib.request.Request(
        KEY_INFO_URL, headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"), parse_float=Decimal)


@dataclass(frozen=True)
class Credit:
    """What OpenRouter reported for the key. None means Unknown, never zero."""

    remaining_usd: Decimal | None
    limit_usd: Decimal | None
    unknown_because: str | None = None

    @property
    def low(self) -> bool:
        return self.remaining_usd is not None and self.remaining_usd < WARN_BELOW_USD

    def _limit_text(self) -> str:
        if self.limit_usd is None:
            return ""
        return f" of its US${money_text(self.limit_usd)} limit"

    def line(self) -> str:
        """The credit in one line, for the run log."""
        if self.remaining_usd is None:
            return (
                f"OpenRouter credit left on the bot's key: Unknown ({self.unknown_because}). "
                "The run carries on."
            )
        return (
            f"OpenRouter credit left on the bot's key: US${money_display(self.remaining_usd)}"
            f"{self._limit_text()} (OpenRouter's report at the start of this run)."
        )

    def warning(self) -> str:
        """The low-credit warning. Only for a known amount."""
        if self.remaining_usd is None:
            raise ValueError("An Unknown credit is never low")
        return (
            f"Low OpenRouter credit: US${money_display(self.remaining_usd)} left on the bot's "
            f"key{self._limit_text()}, under the US${money_text(WARN_BELOW_USD)} warning level. "
            "When nothing is left, OpenRouter refuses the bot's model calls and it stops "
            "forecasting."
        )


def _amount(value: Any) -> Decimal | None:
    """An amount from the key check, or None. Floats are refused as money."""
    if value is None or isinstance(value, (bool, float)):
        return None
    try:
        amount = to_decimal(value)
    except (TypeError, ValueError):
        return None
    return amount if amount.is_finite() else None


def read_credit(api_key: str | None, fetch: KeyInfoFetch) -> Credit:
    """Ask OpenRouter once. Never raises: a check that fails gives Unknown."""
    if not api_key or not api_key.strip():
        return Credit(None, None, "OPENROUTER_API_KEY is not set")
    try:
        info = fetch(api_key)
    except Exception as error:  # a failed check must never stop the run
        # Only the error's type: its text could repeat what was sent.
        return Credit(None, None, f"the key check failed: {type(error).__name__}")
    data = info.get("data") if isinstance(info, dict) else None
    if not isinstance(data, dict):
        return Credit(None, None, "the key check gave no data")
    limit = _amount(data.get("limit"))
    if data.get("limit_remaining") is None:
        why = (
            "the key has no spending limit, so OpenRouter reports no amount left"
            if data.get("limit") is None
            else "the key check gave no amount left"
        )
        return Credit(None, limit, why)
    remaining = _amount(data.get("limit_remaining"))
    if remaining is None:
        return Credit(None, limit, "the key check's amount left could not be read")
    return Credit(remaining, limit)


def _workflow_command_text(text: str) -> str:
    # GitHub's escaping for the message of a workflow command.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _add_to_summary(step_summary: str | None, text: str) -> None:
    """Add a paragraph to the GitHub Actions run's summary page, if there is one."""
    if not step_summary:
        return
    try:
        with open(step_summary, "a", encoding="utf-8") as file:
            file.write(text + "\n\n")
    except OSError as error:
        logger.warning(f"Could not write to the run's summary page: {type(error).__name__}")


def warn(
    credit: Credit,
    *,
    github_actions: bool,
    step_summary: str | None,
    out: Callable[[str], None] = print,
) -> None:
    """Say that the credit is low: in the log, as a GitHub Actions warning, and on the summary page."""
    message = credit.warning()
    logger.warning(message)
    if github_actions:
        out(f"::warning title=Low OpenRouter credit::{_workflow_command_text(message)}")
    _add_to_summary(step_summary, f"**Low OpenRouter credit.** {message}")


def failed_on_purpose(path: Path, day: str) -> bool:
    """Whether a run already ended with a failure for low credit on this UTC day."""
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # an unreadable line must not hold back the warning
        if isinstance(record, dict) and record.get("utc_day") == day:
            return True
    return False


def end_of_run_failure(
    credit: Credit,
    *,
    github_actions: bool,
    dry_run: bool,
    run_mode: str,
    path: Path = WARNINGS_PATH,
    step_summary: str | None = None,
    now: datetime | None = None,
) -> str | None:
    """
    Called after all forecasting is done. Returns the message to end the run
    with a failing exit code, or None. Only on GitHub Actions, only for a run
    that can publish (not a dry run), only when the credit is known and low,
    and only once per UTC day; it records the day in `path` first.
    """
    if not (credit.low and github_actions and not dry_run):
        return None
    moment = now if now is not None else datetime.now(timezone.utc)
    day = utc_day(moment)
    if failed_on_purpose(path, day):
        logger.info(f"Low credit: a run already failed for it on UTC day {day}, so this one does not.")
        return None
    assert credit.remaining_usd is not None
    record = {
        "credit_warning_version": WARNINGS_VERSION,
        "recorded_at_utc": moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "utc_day": day,
        "event": "low_credit_run_failed",
        "limit_remaining_usd": money_text(credit.remaining_usd),
        "limit_usd": money_text(credit.limit_usd) if credit.limit_usd is not None else UNKNOWN,
        "warn_below_usd": money_text(WARN_BELOW_USD),
        "run_mode": run_mode,
        "code_commit": os.getenv("GITHUB_SHA"),
        "run_url": github_run_url(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
    _add_to_summary(
        step_summary,
        "**This run ends with a failure on purpose,** after its forecasting finished, so that "
        "GitHub emails you about the low credit. It does this at most once per UTC day.",
    )
    return (
        f"{credit.warning()} Forecasting finished normally. This run ends with a failure on "
        "purpose, at most once per UTC day, so that GitHub emails you."
    )
