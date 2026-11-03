"""
Spending controls for glass-earth-bot.

Every model call is metered: its cost is recorded against the question it was
made for, and a hard daily cap stops the bot before a call could take the
day's spending over the cap. All money here is Decimal, in US dollars.

How a cost is known, in order of preference:
- "openrouter_reported": the cost OpenRouter returns with the response
  (litellm asks for it on every OpenRouter call).
- "calculated_from_tokens": the token counts in the response times the prices
  in MODEL_PRICES, when OpenRouter reports no cost. Not used for web-search
  calls, whose number of searches is not in the response.
- "unknown": the call failed, or its cost could not be worked out. An unknown
  cost is never treated as zero: the cap counts it at its worst case.

The day is the UTC day, because GitHub runs the schedules in UTC (10:00 in
Brisbane). Spending is read back from forecasts/spend.jsonl, which the
GitHub workflows commit after each run, so the cap holds across runs. Runs on
your own computer write to forecasts/dry_run/spend.jsonl (ignored by git) and
count both files.
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

SPEND_LOG_PATH = Path("forecasts/spend.jsonl")
LOCAL_SPEND_LOG_PATH = Path("forecasts/dry_run/spend.jsonl")
SPEND_VERSION = 1

# Defaults, both changeable without editing code (see daily_cap_from_env).
DEFAULT_DAILY_CAP_USD = Decimal("5.00")
# Low-value questions (for example MiniBench) may not spend into the last part
# of the cap, so that a high-value question later in the day can still run.
DEFAULT_HIGH_VALUE_RESERVE_USD = Decimal("2.00")

CAP_ENV = "GLASS_EARTH_DAILY_CAP_USD"
RESERVE_ENV = "GLASS_EARTH_HIGH_VALUE_RESERVE_USD"

# A conservative token estimate for text the bot sends: about 4 characters
# make a token in English, so counting 3 overestimates.
CHARS_PER_TOKEN_ESTIMATE = 3

UNKNOWN = "Unknown"
ZERO = Decimal("0")


class SpendCapReached(RuntimeError):
    """A model call was refused because it could take spending over the cap."""


class SpendRecordUnreadable(RuntimeError):
    """A spend record could not be read, so today's spending is unknown."""


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per token, as OpenRouter's model list states them."""

    prompt: Decimal
    completion: Decimal
    cache_read: Decimal
    web_search: Decimal
    # Some models charge more per token once a prompt reaches this size.
    long_prompt_from_tokens: int | None = None
    long_prompt: Decimal | None = None
    long_completion: Decimal | None = None

    def per_token(self, prompt_tokens: int) -> tuple[Decimal, Decimal]:
        if (
            self.long_prompt_from_tokens is not None
            and prompt_tokens >= self.long_prompt_from_tokens
            and self.long_prompt is not None
            and self.long_completion is not None
        ):
            return self.long_prompt, self.long_completion
        return self.prompt, self.completion


# Observed on OpenRouter's public model list, GET
# https://openrouter.ai/api/v1/models (no key needed), retrieved
# 2026-11-03T04:29:51Z, 451 models listed. Strings are copied as listed.
PRICES_RETRIEVED_AT_UTC = "2026-11-03T04:29:51Z"
MODEL_PRICES: dict[str, ModelPrice] = {
    # US$4 per million input tokens, US$20 per million output tokens.
    "anthropic/claude-opus-5.5": ModelPrice(
        prompt=Decimal("0.000004"),
        completion=Decimal("0.00002"),
        cache_read=Decimal("0.0000002"),
        web_search=Decimal("0.01"),
    ),
    # US$10 per million input tokens, US$50 per million output tokens.
    "anthropic/claude-fable-5.1": ModelPrice(
        prompt=Decimal("0.00001"),
        completion=Decimal("0.00005"),
        cache_read=Decimal("0.00000025"),
        web_search=Decimal("0.01"),
    ),
    # US$1 / US$5 per million, doubled from 272,000 prompt tokens.
    "openai/gpt-6.1-terra": ModelPrice(
        prompt=Decimal("0.000001"),
        completion=Decimal("0.000005"),
        cache_read=Decimal("0.00000005"),
        web_search=Decimal("0.01"),
        long_prompt_from_tokens=272000,
        long_prompt=Decimal("0.000002"),
        long_completion=Decimal("0.0000075"),
    ),
    # US$0.10 / US$0.50 per million, doubled from 272,000 prompt tokens.
    "openai/gpt-6-luna": ModelPrice(
        prompt=Decimal("0.0000001"),
        completion=Decimal("0.0000005"),
        cache_read=Decimal("0.00000001"),
        web_search=Decimal("0.01"),
        long_prompt_from_tokens=272000,
        long_prompt=Decimal("0.0000002"),
        long_completion=Decimal("0.00000075"),
    ),
}


def price_key(model: str) -> str:
    """The MODEL_PRICES key for a litellm model name."""
    key = model.removeprefix("openrouter/")
    return key.split(":", 1)[0]


def price_for(model: str) -> ModelPrice:
    key = price_key(model)
    if key not in MODEL_PRICES:
        raise KeyError(
            f"No price recorded for {key}. Add it to MODEL_PRICES from OpenRouter's model list first."
        )
    return MODEL_PRICES[key]


def is_web_search_model(model: str) -> bool:
    return model.endswith(":online")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 50


def to_decimal(value: Any) -> Decimal:
    """Money from a string, int or Decimal. Floats are refused."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("Money must not be a float; pass a string or Decimal")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError(f"Not an amount of money: {value!r}") from e


def money_text(amount: Decimal | str) -> str:
    """An exact amount as plain text, for records that must add up."""
    if isinstance(amount, str):
        return amount
    return format(amount, "f")


def money_display(amount: Decimal) -> str:
    """An amount rounded to a hundredth of a cent, for log lines."""
    return f"{amount.quantize(Decimal('0.0001'))}"


def worst_case_call_cost(
    model: str,
    prompt_tokens: int,
    max_output_tokens: int,
    max_searches: int = 0,
) -> Decimal:
    """The most one call can cost: every output token used, every search made."""
    price = price_for(model)
    prompt_price, completion_price = price.per_token(prompt_tokens)
    cost = Decimal(prompt_tokens) * prompt_price + Decimal(max_output_tokens) * completion_price
    if max_searches:
        cost += Decimal(max_searches) * price.web_search
    return cost


@dataclass
class CallUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_prompt_tokens: int | None
    reported_cost: Decimal | None


def usage_from_response(response: Any) -> CallUsage:
    """Read the token counts and OpenRouter's reported cost from a litellm response."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = (
        getattr(usage, "completion_tokens", None) if usage is not None else None
    )
    cached = None
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
    reported = None
    hidden = getattr(response, "_hidden_params", None) or {}
    headers = hidden.get("additional_headers") or {}
    raw_cost = headers.get("llm_provider-x-litellm-response-cost")
    if raw_cost is not None:
        try:
            # litellm stores OpenRouter's number as a float; go through its
            # shortest text form so no binary rounding is carried over.
            reported = Decimal(repr(raw_cost)) if isinstance(raw_cost, float) else to_decimal(raw_cost)
        except (ValueError, TypeError):
            reported = None
    return CallUsage(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        cached_prompt_tokens=cached if isinstance(cached, int) else None,
        reported_cost=reported,
    )


def cost_of_call(model: str, usage: CallUsage) -> tuple[Decimal | None, str]:
    """(cost, source). Cost is None when it cannot be known."""
    if usage.reported_cost is not None:
        return usage.reported_cost, "openrouter_reported"
    if is_web_search_model(model):
        # The number of searches is not in the response.
        return None, "unknown"
    if usage.prompt_tokens is None or usage.completion_tokens is None:
        return None, "unknown"
    price = price_for(model)
    prompt_price, completion_price = price.per_token(usage.prompt_tokens)
    cached = usage.cached_prompt_tokens or 0
    uncached = max(usage.prompt_tokens - cached, 0)
    cost = (
        Decimal(uncached) * prompt_price
        + Decimal(cached) * price.cache_read
        + Decimal(usage.completion_tokens) * completion_price
    )
    return cost, "calculated_from_tokens"


@dataclass
class QuestionTag:
    """Which question the current model calls are for."""

    question_id: int | None
    post_id: int | None
    url: str | None


current_question: ContextVar[QuestionTag | None] = ContextVar(
    "current_question", default=None
)


def utc_day(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def daily_cap_from_env(override: str | None = None) -> tuple[Decimal, Decimal]:
    """(cap, reserve) from the command line, the environment, or the defaults."""
    raw_cap = override if override is not None else os.getenv(CAP_ENV)
    raw_reserve = os.getenv(RESERVE_ENV)
    cap = DEFAULT_DAILY_CAP_USD if not raw_cap else to_decimal(raw_cap.strip())
    reserve = (
        DEFAULT_HIGH_VALUE_RESERVE_USD if not raw_reserve else to_decimal(raw_reserve.strip())
    )
    if cap < ZERO:
        raise ValueError("The daily cap cannot be negative")
    if reserve < ZERO:
        raise ValueError("The high-value reserve cannot be negative")
    reserve = min(reserve, cap)
    return cap, reserve


@dataclass
class Reservation:
    amount: Decimal
    model: str
    purpose: str
    tag: QuestionTag | None


@dataclass
class QuestionCost:
    known: Decimal = ZERO
    unknown_calls: int = 0
    cap_basis: Decimal = ZERO
    by_model: dict[str, Decimal] = field(default_factory=dict)
    calls: int = 0

    def as_text(self) -> str:
        if self.unknown_calls:
            return UNKNOWN
        return money_text(self.known)


class SpendMeter:
    """
    Keeps today's spending, refuses calls that could take it over the cap,
    and records every call in the spend log.
    """

    def __init__(
        self,
        *,
        cap: Decimal,
        reserve: Decimal,
        log_path: Path,
        read_paths: Iterable[Path],
        run_mode: str,
        dry_run: bool,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        if not isinstance(cap, Decimal) or not isinstance(reserve, Decimal):
            raise TypeError("The cap and reserve must be Decimal")
        self.cap = cap
        self.reserve = min(reserve, cap)
        self.log_path = log_path
        self.read_paths = list(dict.fromkeys([*read_paths, log_path]))
        self.run_mode = run_mode
        self.dry_run = dry_run
        self.clock = clock
        self.day = utc_day(clock())
        self.spent_before_run = self._read_spent_today()
        self.spent_this_run = ZERO
        self.in_flight = ZERO
        self.by_question: dict[int | None, QuestionCost] = {}
        self.skips: list[dict[str, Any]] = []

    # ------------------------------------------------------------ reading

    def _read_spent_today(self) -> Decimal:
        total = ZERO
        for path in self.read_paths:
            if not path.exists():
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("utc_day") != self.day:
                        continue
                    total += to_decimal(record.get("cap_basis_usd", "0"))
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
                    raise SpendRecordUnreadable(
                        f"Line {line_number} of {path} cannot be read, so today's spending is unknown. "
                        "Nothing more will be spent until it is fixed."
                    ) from e
        return total

    def refresh_day(self) -> None:
        today = utc_day(self.clock())
        if today != self.day:
            self.day = today
            self.spent_before_run = self._read_spent_today()
            self.spent_this_run = ZERO

    # ------------------------------------------------------------ totals

    @property
    def spent_today(self) -> Decimal:
        return self.spent_before_run + self.spent_this_run

    def remaining(self) -> Decimal:
        return self.cap - self.spent_today - self.in_flight

    def can_afford(self, amount: Decimal, keep_free: Decimal = ZERO) -> bool:
        self.refresh_day()
        return self.spent_today + self.in_flight + amount + keep_free <= self.cap

    def question_cost(self, question_id: int | None) -> QuestionCost:
        return self.by_question.setdefault(question_id, QuestionCost())

    # ------------------------------------------------------------ calls

    def ensure_affordable(self, amount: Decimal, model: str, purpose: str) -> None:
        if not self.can_afford(amount):
            self._record_refusal(amount, model, purpose)
            raise SpendCapReached(
                f"Refused a {purpose} call to {price_key(model)}: it could cost up to "
                f"US${money_display(amount)}, and US${money_display(self.spent_today)} of the "
                f"US${money_display(self.cap)} daily cap is already spent."
            )

    def reserve_call(self, amount: Decimal, model: str, purpose: str) -> Reservation:
        self.ensure_affordable(amount, model, purpose)
        self.in_flight += amount
        return Reservation(amount=amount, model=model, purpose=purpose, tag=current_question.get())

    def settle(self, reservation: Reservation, usage: CallUsage) -> Decimal | None:
        cost, source = cost_of_call(reservation.model, usage)
        self._finish(reservation, cost, source, outcome="ok", usage=usage)
        return cost

    def settle_failed(self, reservation: Reservation, error_type: str) -> None:
        self._finish(
            reservation,
            None,
            "unknown",
            outcome=f"failed ({error_type})",
            usage=CallUsage(None, None, None, None),
        )

    def _finish(
        self,
        reservation: Reservation,
        cost: Decimal | None,
        source: str,
        *,
        outcome: str,
        usage: CallUsage,
    ) -> None:
        self.in_flight -= reservation.amount
        cap_basis = cost if cost is not None else reservation.amount
        self.spent_this_run += cap_basis
        tag = reservation.tag
        question_id = tag.question_id if tag else None
        totals = self.question_cost(question_id)
        totals.calls += 1
        totals.cap_basis += cap_basis
        key = price_key(reservation.model)
        if cost is None:
            totals.unknown_calls += 1
        else:
            totals.known += cost
            totals.by_model[key] = totals.by_model.get(key, ZERO) + cost
        self._append(
            {
                "event": "model_call",
                "question_id": question_id,
                "post_id": tag.post_id if tag else None,
                "purpose": reservation.purpose,
                "model": key,
                "web_search": is_web_search_model(reservation.model),
                "outcome": outcome,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "cost_usd": money_text(cost) if cost is not None else UNKNOWN,
                "cost_source": source,
                "worst_case_usd": money_text(reservation.amount),
                "cap_basis_usd": money_text(cap_basis),
            }
        )

    def _record_refusal(self, amount: Decimal, model: str, purpose: str) -> None:
        tag = current_question.get()
        self._append(
            {
                "event": "call_refused",
                "question_id": tag.question_id if tag else None,
                "post_id": tag.post_id if tag else None,
                "purpose": purpose,
                "model": price_key(model),
                "worst_case_usd": money_text(amount),
                "cap_basis_usd": "0",
            }
        )

    def record_skip(self, tag: QuestionTag, reason: str, worst_case: Decimal) -> None:
        record = {
            "event": "question_skipped",
            "question_id": tag.question_id,
            "post_id": tag.post_id,
            "url": tag.url,
            "reason": reason,
            "worst_case_usd": money_text(worst_case),
            "cap_basis_usd": "0",
        }
        self.skips.append(record)
        self._append(record)

    def _append(self, fields: dict[str, Any]) -> None:
        server = os.getenv("GITHUB_SERVER_URL")
        repository = os.getenv("GITHUB_REPOSITORY")
        run_id = os.getenv("GITHUB_RUN_ID")
        run_url = (
            f"{server}/{repository}/actions/runs/{run_id}"
            if server and repository and run_id
            else None
        )
        moment = self.clock()
        record = {
            "spend_version": SPEND_VERSION,
            "recorded_at_utc": moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "utc_day": utc_day(moment),
            "run_mode": self.run_mode,
            "dry_run": self.dry_run,
            **fields,
            "daily_cap_usd": money_text(self.cap),
            "prices_retrieved_at_utc": PRICES_RETRIEVED_AT_UTC,
            "code_commit": os.getenv("GITHUB_SHA"),
            "run_url": run_url,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

    # ------------------------------------------------------------ summary

    def summary_line(self) -> str:
        return (
            f"Spend (UTC day {self.day}): US${money_display(self.spent_today)} of the "
            f"US${money_display(self.cap)} daily cap, US${money_display(self.spent_this_run)} in this run; "
            f"{len(self.skips)} question(s) skipped by the cap."
        )


def spend_log_paths(github_actions: bool | None = None) -> tuple[Path, list[Path]]:
    """(the log this run writes, the logs it reads)."""
    if github_actions is None:
        github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    if github_actions:
        return SPEND_LOG_PATH, [SPEND_LOG_PATH]
    return LOCAL_SPEND_LOG_PATH, [SPEND_LOG_PATH, LOCAL_SPEND_LOG_PATH]
