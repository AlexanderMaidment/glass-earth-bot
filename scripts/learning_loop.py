"""
The learning loop of glass-earth-bot: every morning, from resolved questions
only, what resolved, how our forecasts scored, the three worst misses with a
one-line cause, and a running calibration table, all in LESSONS.md.

.github/workflows/learning_loop.yaml runs it daily at 21:00 UTC (07:00 in
Brisbane), from the top of the repository:

    uv run --locked python scripts/learning_loop.py

What it does:
1. Reads the bot's own forecasts/ledger.jsonl for the questions it has
   forecast, leaving out the unscored bot testing area. While a question is
   open, its ledger line is sealed: IDs, link and time in the clear, the
   forecast in a sealed box that this script never opens. After the close,
   a later reveal line gives the forecast. Older full lines hold it in the
   clear. Either way, no forecast is read until Metaculus says its question
   has resolved.
2. Asks Metaculus about each post not reported before: read-only GETs with
   METACULUS_TOKEN, one at a time, PAUSE_SECONDS apart, never retried. A 429
   (too many requests), a refusal, an error or a failed request stops the
   requests for the morning; what was read before it is still reported, and
   the rest wait for the next morning.
3. Keeps only questions that have resolved; annulled and ambiguous ones are
   left out. For each newly resolved question: our last forecast before it
   closed (from the ledger's reveal line, or else as Metaculus reports our
   own forecast), the outcome, the scores worked out here, and Metaculus's
   own baseline and peer scores if the API gives them (else Unknown).
4. The three worst misses of the morning each get a one-line cause, in its
   own words, from the bot's cheap helper model (HELPER_MODEL in main.py)
   through the metered path in metered_llm.py, so the daily cap counts every
   call and the spend record has a line for it. Without OPENROUTER_API_KEY,
   without room under the cap, if the call fails, or if the cause repeats
   the question's words, the cause reads "Unknown (not reviewed)".
5. Writes forecasts/learning_state.json (what has been reported, so nothing
   is reported twice) and renders LESSONS.md from it.

Metaculus's terms of use do not allow republishing its site material, so
neither LESSONS.md nor the state file holds a question's title, text or
option labels: questions are named by their ID and link only. Nothing about
an open question is ever written. The loop never changes the bot: the bot
changes at most once a week, and only after a change beats the old version
on closed questions with research search capped at each question's close
date.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import requests  # noqa: E402

import main as bot  # noqa: E402  (the bot's helper model; main loads .env as the bot does)
from metered_llm import MeteredLlm  # noqa: E402
from spend import (  # noqa: E402
    QuestionTag,
    SpendCapReached,
    SpendMeter,
    SpendRecordUnreadable,
    current_question,
    daily_cap_from_env,
    money_display,
    spend_log_paths,
)

logger = logging.getLogger("learning_loop")

LEDGER_PATH = Path("forecasts/ledger.jsonl")
STATE_PATH = Path("forecasts/learning_state.json")
LESSONS_PATH = Path("LESSONS.md")

API_BASE = "https://www.metaculus.com/api"
# Seconds between two requests to Metaculus (a choice: forecasting-tools
# waits 3.5 to 4.5 s). No request is ever retried.
PAUSE_SECONDS = 4.0
REQUEST_TIMEOUT_SECONDS = 30
# At most this many posts are checked in one morning, about 13 minutes at the
# pause above; the rest wait for the next morning, oldest forecast first.
MAX_POSTS_PER_MORNING = 200

# Queensland has no daylight saving, so Brisbane is always UTC+10.
BRISBANE = timezone(timedelta(hours=10))

UNKNOWN = "Unknown"
UNREVIEWED = "Unknown (not reviewed)"
WORST_MISSES = 3
# Assumption: a probability below this counts as this in a log score, so one
# outcome a forecast all but ruled out cannot make a mean infinite.
MIN_PROBABILITY = 1e-6
# Assumption: the uniform forecast that "above uniform" compares against puts
# this much beyond each open bound of a numeric, discrete or date question.
OPEN_BOUND_UNIFORM_MASS = 0.05

REVIEW_PURPOSE = "learning_review"
REVIEW_MARKER = "You are reviewing one resolved forecast made by a forecasting bot."
TITLE_CHARS = 300
CRITERIA_CHARS = 2000
FORECAST_CHARS = 1500
REASONING_CHARS = 6000
CAUSE_CHARS = 300
# A cause that repeats this many words in a row of the question (or all of a
# shorter title, from 3 words), or any of its labels, is withheld.
QUOTE_WORDS = 5

TEST_AREA_RUN_MODE = "test_questions"
TEST_AREA_SLUG = "bot-testing-area"
TEST_AREA_ID = 32977
CANCELLED = ("annulled", "ambiguous")
CONTINUOUS = ("numeric", "discrete", "date")
TYPE_ORDER = ("binary", "multiple_choice", "numeric", "discrete", "date")
TYPE_NAMES = {
    "binary": "Yes/no",
    "multiple_choice": "Multiple choice",
    "numeric": "Numeric",
    "discrete": "Discrete",
    "date": "Date",
}
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

STATE_VERSION = 1
ABOUT = (
    "Written by scripts/learning_loop.py. The resolved questions already reported in "
    "LESSONS.md with their scores, the resolved questions left out, and what each "
    "morning reported. Questions are named by ID only, with no title, text or option "
    "label, and nothing about an open question is ever written here."
)

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


class SetupError(RuntimeError):
    """Stops the run before anything is fetched or written."""


class ScoreUnknown(ValueError):
    """A score that cannot be worked out. The message says why, without quoting the question."""


class FetchStopped(RuntimeError):
    """The requests to Metaculus stopped for this run."""

    def __init__(self, reason: str, *, fatal: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        # A refusal or an answer that is not JSON needs a look, so the run
        # ends with an error; a 429 or a passing failure just waits a day.
        self.fatal = fatal


# ---------------------------------------------------------------- text


def clean_text(value: Any, limit: int | None = None) -> str:
    """One line of plain text, with no em or en dashes."""
    text = "" if value is None else str(value)
    text = text.replace(EM_DASH, " - ").replace(EN_DASH, " - ")
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def cell(value: Any) -> str:
    """Text for a Markdown table cell."""
    return clean_text(value).replace("|", "\\|")


def pct(probability: float) -> str:
    value = probability * 100
    if value != 0 and abs(value) < 0.001:
        return "under 0.001%"
    text = f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"
    text = text.rstrip("0").rstrip(".")
    return f"{text}%"


def num(value: Any, digits: int = 3) -> str:
    """A score for display: n/a when it does not apply, Unknown when unknown."""
    if value is None:
        return "n/a"
    if _is_number(value):
        return f"{value:.{digits}f}"
    return UNKNOWN


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def parse_time(value: Any) -> datetime | None:
    """A UTC time from the API or the ledger (ISO text or seconds since 1970)."""
    if _is_number(value):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def utc_text(moment: datetime | None) -> str | None:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ") if moment is not None else None


def question_url(post_id: int) -> str:
    return f"https://www.metaculus.com/questions/{post_id}/"


def day_heading(day: str) -> str:
    moment = date.fromisoformat(day)
    return f"{moment.day} {MONTHS[moment.month - 1]} {moment.year}"


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write(text)
    os.replace(temporary, path)


# ---------------------------------------------------------------- ledger


@dataclass
class LedgerForecast:
    """One forecast in the ledger. Its payload is None while it is still sealed."""

    recorded_at: datetime | None
    payload: dict[str, Any] | None


@dataclass
class LedgerQuestion:
    question_id: int
    post_id: int
    forecasts: list[LedgerForecast] = field(default_factory=list)
    test_area: bool = False
    first_recorded: datetime | None = None

    def add(self, recorded_at: datetime | None, payload: Any, run_mode: Any) -> None:
        self.forecasts.append(LedgerForecast(recorded_at, payload if isinstance(payload, dict) else None))
        if run_mode == TEST_AREA_RUN_MODE:
            self.test_area = True
        if recorded_at is not None and (self.first_recorded is None or recorded_at < self.first_recorded):
            self.first_recorded = recorded_at


def _ids(entry: dict[str, Any]) -> tuple[int, int] | None:
    question_id = entry.get("question_id")
    post_id = entry.get("post_id")
    if not _is_int(post_id):
        match = re.search(r"/questions/(\d+)", str(entry.get("url") or ""))
        post_id = int(match.group(1)) if match else None
    if _is_int(question_id) and _is_int(post_id):
        return question_id, post_id
    return None


def read_ledger(path: Path = LEDGER_PATH) -> dict[int, LedgerQuestion]:
    """
    The questions in the bot's ledger, from its sealed lines and older full
    lines, each forecast taken from the matching reveal line once there is
    one. The sealed boxes are never opened. The forecasts stay here, unread,
    until Metaculus says their question has resolved.
    """
    if not path.exists():
        return {}
    entries: list[dict[str, Any]] = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Skipping unreadable line {number} of {path}")
            continue
        if isinstance(entry, dict) and not entry.get("dry_run"):
            entries.append(entry)
    reveals = {
        entry["seal_sha256"]: entry["record"]
        for entry in entries
        if entry.get("revealed") is True
        and isinstance(entry.get("seal_sha256"), str)
        and isinstance(entry.get("record"), dict)
    }
    found: dict[int, LedgerQuestion] = {}

    def add(entry: dict[str, Any], recorded_at: datetime | None, payload: Any, run_mode: Any) -> None:
        ids = _ids(entry)
        if ids is not None:
            found.setdefault(ids[0], LedgerQuestion(*ids)).add(recorded_at, payload, run_mode)

    matched: set[str] = set()
    for entry in entries:
        if entry.get("revealed") is True:
            continue
        if entry.get("sealed") is True:
            seal = entry.get("seal_sha256")
            record = reveals.get(seal, {}) if isinstance(seal, str) else {}
            if record:
                matched.add(seal)
            add(
                entry,
                parse_time(entry.get("recorded_at_utc")) or parse_time(record.get("recorded_at_utc")),
                record.get("final_forecast"),
                entry.get("run_mode") or record.get("run_mode"),
            )
        else:
            add(entry, parse_time(entry.get("recorded_at_utc")), entry.get("final_forecast"), entry.get("run_mode"))
    # A reveal whose sealed line is missing still names its forecast.
    for seal, record in reveals.items():
        if seal not in matched:
            add(record, parse_time(record.get("recorded_at_utc")), record.get("final_forecast"), record.get("run_mode"))
    return found


def ledger_forecast_at_close(
    forecasts: list[LedgerForecast], close_time: datetime | None
) -> LedgerForecast | None:
    """The latest ledger forecast recorded at or before the close (any, if the close is unknown)."""
    best: tuple[datetime, int, LedgerForecast] | None = None
    for index, forecast in enumerate(forecasts):
        if forecast.recorded_at is None:
            continue
        if close_time is not None and forecast.recorded_at > close_time:
            continue
        if best is None or (forecast.recorded_at, index) >= best[:2]:
            best = (forecast.recorded_at, index, forecast)
    return best[2] if best is not None else None


def payload_from_api(question: dict[str, Any], values: Any) -> dict[str, Any] | None:
    """A forecast from my_forecasts in the shape the ledger records, or None if unreadable."""
    if not isinstance(values, list) or not values or not all(_is_number(v) for v in values):
        return None
    question_type = question.get("type")
    if question_type == "binary":
        return {"probability_yes": values[1]} if len(values) == 2 else None
    if question_type == "multiple_choice":
        options = question.get("options")
        if isinstance(options, list) and len(options) == len(values):
            return {"probability_yes_per_category": dict(zip(map(str, options), values))}
        return None
    if question_type in CONTINUOUS:
        return {"continuous_cdf": values}
    return None


def api_forecast_at_close(
    question: dict[str, Any], close_time: datetime | None
) -> tuple[datetime, dict[str, Any] | None] | None:
    """
    Our own forecast standing at the close as Metaculus reports it
    (my_forecasts), for when the ledger has not revealed it: the latest one
    made at or before the close. Its payload is None if it cannot be read.
    """
    mine = question.get("my_forecasts")
    history = mine.get("history") if isinstance(mine, dict) else None
    best: tuple[datetime, int, dict[str, Any]] | None = None
    for index, entry in enumerate(history if isinstance(history, list) else []):
        if not isinstance(entry, dict):
            continue
        start = parse_time(entry.get("start_time"))
        if start is None or (close_time is not None and start > close_time):
            continue
        if best is None or (start, index) >= best[:2]:
            best = (start, index, entry)
    if best is None:
        return None
    return best[0], payload_from_api(question, best[2].get("forecast_values"))


# ---------------------------------------------------------------- state


def new_state() -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "about": ABOUT,
        "questions": {},
        "left_out": {},
        "mornings": {},
    }


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return new_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SetupError(
            f"{path} cannot be read ({type(error).__name__}). Fix it first, so that "
            "nothing already reported is lost or reported twice."
        ) from error
    if (
        not isinstance(state, dict)
        or state.get("state_version") != STATE_VERSION
        or not all(isinstance(state.get(key), dict) for key in ("questions", "left_out", "mornings"))
    ):
        raise SetupError(f"{path} is not a learning state this script can read (version {STATE_VERSION}).")
    return state


def state_text(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------- Metaculus


class MetaculusReader:
    """
    Read-only GETs to the Metaculus API with the bot's token, one at a time
    with a pause between them, never retried. The first 429, refusal, error
    or failed request stops every later request of the run.
    """

    def __init__(
        self,
        token: str,
        *,
        pause: Callable[[float], None] = time.sleep,
        pause_seconds: float = PAUSE_SECONDS,
    ) -> None:
        self._token = token
        self._pause = pause
        self.pause_seconds = pause_seconds
        self.requests_made = 0
        self.stopped: FetchStopped | None = None

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        missing_ok: tuple[int, ...] = (404,),
    ) -> Any:
        """The parsed answer, or None for a status in missing_ok."""
        if self.stopped is not None:
            raise self.stopped
        if self.requests_made:
            self._pause(self.pause_seconds)
        self.requests_made += 1
        try:
            response = requests.get(
                f"{API_BASE}{path}",
                params=params,
                headers={"Authorization": f"Token {self._token}", "Accept-Language": "en"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise self._stop(f"a request failed ({type(error).__name__})", fatal=False) from error
        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError:
                raise self._stop(
                    "Metaculus answered with a page that is not JSON, possibly an anti-bot check",
                    fatal=True,
                ) from None
        if status in missing_ok:
            return None
        if status == 429:
            raise self._stop("Metaculus answered 429 (too many requests)", fatal=False)
        if status in (401, 403):
            raise self._stop(f"Metaculus refused the request (HTTP {status})", fatal=True)
        if status >= 500:
            raise self._stop(f"Metaculus had an error (HTTP {status})", fatal=False)
        raise self._stop(f"Metaculus answered HTTP {status}", fatal=True)

    def _stop(self, reason: str, *, fatal: bool) -> FetchStopped:
        self.stopped = FetchStopped(reason, fatal=fatal)
        logger.warning(f"Stopping the requests to Metaculus: {reason}. Nothing is retried.")
        return self.stopped


def questions_in_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    """Every question of a post: a single question, a group's, or a conditional's."""
    found: list[dict[str, Any]] = []
    if isinstance(post.get("question"), dict):
        found.append(post["question"])
    group = post.get("group_of_questions")
    if isinstance(group, dict):
        found += [q for q in group.get("questions") or [] if isinstance(q, dict)]
    conditional = post.get("conditional")
    if isinstance(conditional, dict):
        for key in ("condition", "condition_child", "question_yes", "question_no"):
            if isinstance(conditional.get(key), dict):
                found.append(conditional[key])
    return found


def in_test_area(post: dict[str, Any]) -> bool:
    projects = post.get("projects")
    if not isinstance(projects, dict):
        return False
    for key in ("tournament", "question_series"):
        for project in projects.get(key) or []:
            if isinstance(project, dict) and (
                project.get("slug") == TEST_AREA_SLUG or project.get("id") == TEST_AREA_ID
            ):
                return True
    return False


def resolution_criteria(post: dict[str, Any], question: dict[str, Any]) -> str:
    text = question.get("resolution_criteria")
    if not text and isinstance(post.get("group_of_questions"), dict):
        text = post["group_of_questions"].get("resolution_criteria")
    return text if isinstance(text, str) else ""


def question_material(post: dict[str, Any], question: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The question's own words (passages and labels), which no cause may repeat."""
    group = post.get("group_of_questions") if isinstance(post.get("group_of_questions"), dict) else {}
    passages = [
        question.get("title"),
        post.get("title"),
        resolution_criteria(post, question),
        question.get("description") or group.get("description"),
        question.get("fine_print") or group.get("fine_print"),
    ]
    labels = [question.get("label")] + list(question.get("options") or [])
    return [p for p in passages if isinstance(p, str)], [x for x in labels if isinstance(x, str)]


def quotes_the_question(cause: str, passages: list[str], labels: list[str]) -> bool:
    """
    True if the cause repeats the question: QUOTE_WORDS words in a row from
    any passage (all of a shorter one, from 3 words), or any label of at
    least 3 letters or digits.
    """
    padded = " " + " ".join(_words(cause)) + " "
    for passage in passages:
        words = _words(passage)
        n = min(QUOTE_WORDS, len(words))
        if n < 3:
            continue
        for start in range(len(words) - n + 1):
            if " " + " ".join(words[start : start + n]) + " " in padded:
                return True
    for label in labels:
        words = _words(label)
        if len("".join(words)) >= 3 and " " + " ".join(words) + " " in padded:
            return True
    return False


# ---------------------------------------------------------------- scores


def _log(probability: float) -> float:
    return math.log(max(probability, MIN_PROBABILITY))


def _probability(value: Any) -> float:
    if not _is_number(value) or not 0 <= value <= 1:
        raise ScoreUnknown("the forecast holds a value that is not a probability")
    return float(value)


def binary_scores(probability_yes: float, resolved_yes: bool) -> dict[str, Any]:
    """Brier and log score of a yes/no forecast."""
    observed = 1.0 if resolved_yes else 0.0
    on_outcome = probability_yes if resolved_yes else 1.0 - probability_yes
    log_score = _log(on_outcome)
    return {
        "probability_on_outcome": on_outcome,
        "brier": (probability_yes - observed) ** 2,
        "log_score": log_score,
        "above_uniform": log_score - math.log(0.5),
    }


def multiple_choice_scores(probabilities: dict[str, float], outcome: str) -> dict[str, Any]:
    """Log score of the outcome under a multiple choice forecast."""
    if outcome not in probabilities:
        raise ScoreUnknown("the outcome is not one of the forecast's options")
    on_outcome = probabilities[outcome]
    log_score = _log(on_outcome)
    return {
        "probability_on_outcome": on_outcome,
        "brier": None,
        "log_score": log_score,
        "above_uniform": log_score - math.log(1 / len(probabilities)),
    }


def _ratio(range_min: float, range_max: float, zero_point: float | None) -> float | None:
    """The log-scale ratio of a question with a zero point, or None for a linear one."""
    if zero_point is None:
        return None
    ratio = (range_max - zero_point) / (range_min - zero_point)
    return ratio if ratio > 0 and ratio != 1 else None


def unscaled_location(value: float, range_min: float, range_max: float, zero_point: float | None) -> float:
    """
    Where a value sits on the forecast's axis: 0 at the lower bound, 1 at the
    upper. A question with a zero point is on a log scale; this is the
    formula forecasting-tools uses (NumericDistribution).
    """
    if not range_max > range_min:
        raise ScoreUnknown("the question's range is empty")
    ratio = _ratio(range_min, range_max, zero_point)
    if ratio is not None:
        return math.log((value - range_min) * (ratio - 1) / (range_max - range_min) + 1) / math.log(ratio)
    return (value - range_min) / (range_max - range_min)


def scaled_value(location: float, range_min: float, range_max: float, zero_point: float | None) -> float:
    """The inverse of unscaled_location."""
    ratio = _ratio(range_min, range_max, zero_point)
    if ratio is not None:
        return range_min + (range_max - range_min) * (ratio**location - 1) / (ratio - 1)
    return range_min + (range_max - range_min) * location


def bucket_index(location: float, bins: int) -> int:
    """
    The outcome's place in [below the range, bin 1, ..., bin N, above the
    range]. A value on the edge between two bins counts in the lower one, and
    the lower bound itself in bin 1.
    """
    if location < 0:
        return 0
    if location > 1:
        return bins + 1
    if location == 1:
        return bins
    return max(int(location * bins + 1 - 1e-10), 1)


def continuous_scores(
    cdf: list[float], location: float, *, open_lower: bool, open_upper: bool
) -> dict[str, Any]:
    """Log score of the forecast's mass in the outcome's bin, or beyond the bound."""
    if len(cdf) < 2:
        raise ScoreUnknown("the forecast's distribution has fewer than 2 points")
    bins = len(cdf) - 1
    masses = [cdf[0]] + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))] + [1.0 - cdf[-1]]
    bucket = bucket_index(location, bins)
    on_outcome = max(masses[bucket], 0.0)
    beyond = bucket in (0, bins + 1)
    if beyond:
        uniform = OPEN_BOUND_UNIFORM_MASS
    else:
        uniform = (1.0 - OPEN_BOUND_UNIFORM_MASS * (int(open_lower) + int(open_upper))) / bins
    log_score = _log(on_outcome)
    return {
        "probability_on_outcome": on_outcome,
        "brier": None,
        "log_score": log_score,
        "above_uniform": log_score - math.log(uniform),
        "beyond_bounds": beyond,
    }


def outcome_value(resolution: str, question_type: str) -> float:
    try:
        value = float(resolution)
    except ValueError:
        moment = parse_time(resolution) if question_type == "date" else None
        if moment is None:
            raise ScoreUnknown("the outcome is not a number") from None
        value = moment.timestamp()
    if not math.isfinite(value):
        raise ScoreUnknown("the outcome is not a number")
    return value


def _range(scaling: dict[str, Any]) -> tuple[float, float, float | None]:
    try:
        range_min = float(scaling["range_min"])
        range_max = float(scaling["range_max"])
        zero_point = None if scaling.get("zero_point") is None else float(scaling["zero_point"])
    except (KeyError, TypeError, ValueError):
        raise ScoreUnknown("the question has no range to score against") from None
    return range_min, range_max, zero_point


def outcome_location(resolution: str, question_type: str, scaling: dict[str, Any]) -> float:
    if resolution == "below_lower_bound":
        return -math.inf
    if resolution == "above_upper_bound":
        return math.inf
    value = outcome_value(resolution, question_type)
    range_min, range_max, zero_point = _range(scaling)
    if value < range_min:
        return -math.inf
    if value > range_max:
        return math.inf
    return unscaled_location(value, range_min, range_max, zero_point)


def _scaling(question: dict[str, Any]) -> dict[str, Any]:
    scaling = question.get("scaling")
    return scaling if isinstance(scaling, dict) else {}


def scores_for(
    question_type: str, resolution: str, forecast: Any, question: dict[str, Any]
) -> dict[str, Any]:
    """Our scores for one resolved question, from the forecast standing at its close."""
    if not isinstance(forecast, dict):
        raise ScoreUnknown("our forecast could not be read")
    if question_type == "binary":
        if resolution not in ("yes", "no"):
            raise ScoreUnknown("the outcome is not yes or no")
        probability_yes = _probability(forecast.get("probability_yes"))
        scores = binary_scores(probability_yes, resolution == "yes")
        scores["probability_yes"] = probability_yes
        scores["forecast_text"] = f"{pct(probability_yes)} Yes"
        return scores
    if question_type == "multiple_choice":
        raw = forecast.get("probability_yes_per_category")
        if not isinstance(raw, dict) or not raw:
            raise ScoreUnknown("our forecast has no option probabilities")
        scores = multiple_choice_scores({str(k): _probability(v) for k, v in raw.items()}, resolution)
        scores["forecast_text"] = f"{pct(scores['probability_on_outcome'])} on the outcome"
        return scores
    if question_type in CONTINUOUS:
        raw = forecast.get("continuous_cdf")
        if not isinstance(raw, list) or not all(_is_number(v) for v in raw):
            raise ScoreUnknown("our forecast has no distribution")
        scaling = _scaling(question)
        outcomes = scaling.get("inbound_outcome_count")
        expected = (outcomes if _is_int(outcomes) else 200) + 1
        if len(raw) != expected:
            raise ScoreUnknown(f"the forecast has {len(raw)} points where the question has {expected}")
        scores = continuous_scores(
            [float(v) for v in raw],
            outcome_location(resolution, question_type, scaling),
            open_lower=bool(question.get("open_lower_bound", scaling.get("open_lower_bound"))),
            open_upper=bool(question.get("open_upper_bound", scaling.get("open_upper_bound"))),
        )
        where = "beyond the bound" if scores.pop("beyond_bounds") else "in the outcome's bin"
        scores["forecast_text"] = f"{pct(scores['probability_on_outcome'])} {where}"
        return scores
    raise ScoreUnknown(f"the question type {clean_text(question_type, 40)} is not scored")


def outcome_text(resolution: str, question_type: str, question: dict[str, Any]) -> str:
    """The outcome for the written files: never an option label or other question text."""
    if question_type == "binary":
        return resolution.capitalize() if resolution in ("yes", "no") else "neither yes nor no"
    if question_type == "multiple_choice":
        options = question.get("options")
        if isinstance(options, list) and resolution in options:
            return f"option {options.index(resolution) + 1} of {len(options)}"
        return "an option not in the question's list"
    if question_type in CONTINUOUS:
        if resolution == "below_lower_bound":
            return "below the lower bound"
        if resolution == "above_upper_bound":
            return "above the upper bound"
        try:
            value = outcome_value(resolution, question_type)
            if question_type == "date":
                return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ScoreUnknown, OverflowError, OSError, ValueError):
            return "not a value this script reads"
        return clean_text(resolution, 40)
    return "not an outcome this script reads"


def quantile_text(cdf: list[float], q: float, question_type: str, scaling: dict[str, Any]) -> str:
    if q < cdf[0]:
        return "below the lower bound"
    if q > cdf[-1]:
        return "above the upper bound"
    location = 1.0
    for i in range(1, len(cdf)):
        if cdf[i] >= q:
            step = cdf[i] - cdf[i - 1]
            location = (i - 1 + ((q - cdf[i - 1]) / step if step > 0 else 0.0)) / (len(cdf) - 1)
            break
    value = scaled_value(location, *_range(scaling))
    if question_type == "date":
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{value:.6g}"


def forecast_description(question_type: str, payload: Any, question: dict[str, Any]) -> str:
    """Our forecast in words, for the reviewing model only. It is never written to a file."""
    try:
        if question_type == "binary":
            return f"{pct(_probability(payload['probability_yes']))} chance of Yes"
        if question_type == "multiple_choice":
            items = payload["probability_yes_per_category"].items()
            return clean_text("; ".join(f"{name}: {pct(_probability(p))}" for name, p in items), FORECAST_CHARS)
        if question_type in CONTINUOUS:
            cdf = [float(v) for v in payload["continuous_cdf"]]
            parts = [
                f"{name} {quantile_text(cdf, q, question_type, _scaling(question))}"
                for name, q in (("10th percentile", 0.1), ("median", 0.5), ("90th percentile", 0.9))
            ]
            return ", ".join(parts)
    except (KeyError, TypeError, ValueError, AttributeError, ScoreUnknown, ZeroDivisionError, OverflowError, OSError):
        pass
    return "not readable"


def metaculus_score(question: dict[str, Any], name: str) -> float | str:
    """Observed: Metaculus's own score for our forecast, if the API gives it."""
    mine = question.get("my_forecasts")
    data = mine.get("score_data") if isinstance(mine, dict) else None
    value = data.get(name) if isinstance(data, dict) else None
    return float(value) if _is_number(value) else UNKNOWN


@dataclass
class Resolved:
    """A newly resolved question: its record, and what the review may see (never written)."""

    record: dict[str, Any]
    post: dict[str, Any]
    question: dict[str, Any]
    payload: Any


def resolved_record(
    post: dict[str, Any],
    question: dict[str, Any],
    ledger_question: LedgerQuestion,
    day: str,
) -> Resolved | str | None:
    """
    The report on one question Metaculus says has resolved, the reason it is
    left out, or None to ask again next morning (our forecast is still sealed
    and Metaculus does not report it). Only here is a forecast read.
    """
    resolution = question["resolution"]
    close = parse_time(question.get("actual_close_time")) or parse_time(
        question.get("scheduled_close_time")
    )
    chosen = ledger_forecast_at_close(ledger_question.forecasts, close)
    if chosen is not None and chosen.payload is not None:
        recorded, payload, source = chosen.recorded_at, chosen.payload, "ledger"
    else:
        reported = api_forecast_at_close(question, close)
        if reported is None:
            # A forecast still sealed may be revealed later, so it waits.
            if any(forecast.payload is None for forecast in ledger_question.forecasts):
                return None
            return "no forecast of ours stood at the close"
        (recorded, payload), source = reported, "metaculus"
    question_type = str(question.get("type"))
    record: dict[str, Any] = {
        "question_id": ledger_question.question_id,
        "post_id": ledger_question.post_id,
        "type": question_type,
        # A multiple choice outcome is an option label, so only its place is kept.
        "outcome": None if question_type == "multiple_choice" else resolution,
        "outcome_text": outcome_text(resolution, question_type, question),
        "closed_at_utc": utc_text(close),
        "resolved_at_utc": utc_text(parse_time(question.get("actual_resolve_time"))),
        "forecast_recorded_at_utc": utc_text(recorded),
        "forecast_source": source,
        "probability_yes": None,
        "metaculus_baseline_score": metaculus_score(question, "baseline_score"),
        "metaculus_peer_score": metaculus_score(question, "peer_score"),
        "score_note": None,
        "cause": None,
        "reported_on": day,
    }
    try:
        record.update(scores_for(question_type, resolution, payload, question))
    except ScoreUnknown as reason:
        record.update(
            {
                "probability_on_outcome": UNKNOWN,
                "forecast_text": UNKNOWN,
                "brier": UNKNOWN if question_type == "binary" else None,
                "log_score": UNKNOWN,
                "above_uniform": UNKNOWN,
                "score_note": str(reason),
            }
        )
    return Resolved(record, post, question, payload)


# ---------------------------------------------------------------- lessons


def worst_misses(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The worst misses: the lowest above-uniform scores among the forecasts
    that did worse than a uniform one, so that question types compare fairly.
    For yes/no questions the order is the same as the plain log score.
    """
    misses = [r for r in records if _is_number(r.get("above_uniform")) and r["above_uniform"] < 0]
    misses.sort(key=lambda r: (r["above_uniform"], r["question_id"]))
    return misses[:WORST_MISSES]


def calibration_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ten 10% bins of yes/no forecasts, each from its lower edge up to its upper (the last holds 100%)."""
    bins = [{"count": 0, "forecast_sum": 0.0, "yes": 0} for _ in range(10)]
    for record in records:
        probability = record.get("probability_yes")
        if record.get("type") != "binary" or not _is_number(probability):
            continue
        if record.get("outcome") not in ("yes", "no"):
            continue
        index = min(int(Decimal(repr(float(probability))) * 10), 9)
        bins[index]["count"] += 1
        bins[index]["forecast_sum"] += float(probability)
        bins[index]["yes"] += record["outcome"] == "yes"
    return bins


def type_totals(records: list[dict[str, Any]], question_type: str) -> dict[str, Any]:
    of_type = [r for r in records if r.get("type") == question_type]
    scored = [r for r in of_type if _is_number(r.get("log_score"))]

    def mean(key: str) -> float | None:
        values = [r[key] for r in scored if _is_number(r.get(key))]
        return sum(values) / len(values) if values else None

    return {
        "count": len(scored),
        "unknown": len(of_type) - len(scored),
        "brier": mean("brier") if question_type == "binary" else None,
        "log_score": mean("log_score"),
        "above_uniform": mean("above_uniform"),
    }


LESSONS_HEADER = """# Lessons

What glass-earth-bot learns from the questions it has forecast, once they resolve. Every morning at 07:00 in Brisbane (21:00 UTC), `scripts/learning_loop.py` asks Metaculus about the questions in the bot's ledger and reports here the ones that have newly resolved. Each run rewrites this file from `forecasts/learning_state.json`.

## Rules

- Resolved questions only. The loop never learns from the bot's forecasts on open questions, and nothing about an open question is written here or in the state file.
- Annulled and ambiguous questions are left out, and so are the questions of the unscored bot testing area.
- The bot changes at most once a week, and only after the change beats the old version on closed questions, with research search capped at each question's close date.
- Until a date-capped search exists (AskNews is the planned one), no lesson-driven change is made. These lessons are notes; they change nothing by themselves.
- Questions are named by their Metaculus ID and link only. Metaculus's terms of use do not allow republishing its site material, so no question title, question text or option label is copied here or into the state file, and each cause describes the miss in its own words.

## How the numbers are worked out

- Observed: each outcome, and Metaculus's own baseline and peer scores for our forecast, from the Metaculus API. Unknown when the API does not give them. A multiple choice outcome is shown by its place in the question's list of options.
- Calculated: every other number, here, from the bot's last forecast before the question closed: as the ledger's reveal line gives it after the close, or as Metaculus reports our forecast when the ledger has not revealed it yet.
- Brier score (yes/no questions): the forecast probability of Yes minus the outcome (1 for Yes, 0 for No), squared. 0 is perfect and 1 the worst.
- Log score: the natural log of the probability the forecast gave to what happened. 0 is perfect; lower is worse. For numeric, discrete and date questions, that probability is the mass the forecast's distribution put in the outcome's bin, or beyond the bound when the outcome fell outside the range.
- Above uniform: the log score minus that of a uniform forecast, which gives 50% to each side of a yes/no question, an equal share to each option, and for numeric, discrete and date questions 5% beyond each open bound (an assumption) with the rest spread evenly over the bins. Below 0 means the forecast did worse than spreading its probability evenly.
- Worst misses: each morning, the three forecasts furthest below 0 above uniform, so that question types compare fairly. Each gets a one-line cause from the bot's cheap helper model, which sees only the resolved question, its outcome, our forecast and the bot's own reasoning when Metaculus has it. A cause that repeats the question's own words is withheld. "Unknown (not reviewed)" means no model reviewed it, or its cause was withheld.
- Assumption: a probability below 0.000001 counts as 0.000001, so that one outcome a forecast all but ruled out cannot make a mean infinite.
- Unknown means a number could not be worked out. It is left out of the means and never counted as zero.
"""


def link(record: dict[str, Any]) -> str:
    return f"[Question {record['question_id']}]({question_url(record['post_id'])})"


def render_calibration(questions: dict[str, dict[str, Any]]) -> list[str]:
    records = list(questions.values())
    if not records:
        reported = "No resolved questions reported yet."
    elif len(records) == 1:
        reported = "From the 1 resolved question reported so far."
    else:
        reported = f"From the {len(records)} resolved questions reported so far."
    lines = [
        "## Calibration so far",
        "",
        reported,
        "",
        "Yes/no questions, by our forecast of Yes. Each bin holds forecasts from its lower edge up to, not including, its upper edge; the last one includes 100%.",
        "",
        "| Forecast of Yes | Count | Mean forecast | Observed frequency of Yes |",
        "| --- | --- | --- | --- |",
    ]
    for index, item in enumerate(calibration_bins(records)):
        if item["count"]:
            mean = pct(item["forecast_sum"] / item["count"])
            frequency = pct(item["yes"] / item["count"])
        else:
            mean = frequency = "n/a"
        lines.append(f"| {index * 10}% to {index * 10 + 10}% | {item['count']} | {mean} | {frequency} |")
    lines += [
        "",
        "By question type:",
        "",
        "| Type | Count | Mean Brier | Mean log score | Mean above uniform |",
        "| --- | --- | --- | --- | --- |",
    ]
    unknown = 0
    for question_type in TYPE_ORDER:
        totals = type_totals(records, question_type)
        unknown += totals["unknown"]
        lines.append(
            f"| {TYPE_NAMES[question_type]} | {totals['count']} | {num(totals['brier'])} | "
            f"{num(totals['log_score'])} | {num(totals['above_uniform'])} |"
        )
    if unknown:
        lines += ["", f"Also reported, with an Unknown score and so left out of these means: {unknown}."]
    return lines + [""]


def render_morning(day: str, morning: dict[str, Any], questions: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"### {day_heading(day)}", ""]
    records = [questions[str(q)] for q in morning.get("questions", []) if str(q) in questions]
    if not records:
        lines.append("Nothing new resolved.")
    else:
        lines += [
            f"{len(records)} newly resolved question{'' if len(records) == 1 else 's'}.",
            "",
            "| Question | Type | Outcome | Our forecast | Brier | Log score | Above uniform | Metaculus baseline | Metaculus peer |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in records:
            lines.append(
                f"| {link(r)} | {TYPE_NAMES.get(r.get('type'), cell(r.get('type')))} | "
                f"{cell(r.get('outcome_text'))} | {cell(r.get('forecast_text'))} | {num(r.get('brier'))} | "
                f"{num(r.get('log_score'))} | {num(r.get('above_uniform'))} | "
                f"{num(r.get('metaculus_baseline_score'), 1)} | {num(r.get('metaculus_peer_score'), 1)} |"
            )
        worst = [questions[str(q)] for q in morning.get("worst", []) if str(q) in questions]
        lines.append("")
        if worst:
            lines += ["Worst misses:", ""]
            for number, r in enumerate(worst, start=1):
                lines.append(
                    f"{number}. {link(r)}: {clean_text(r.get('forecast_text'))}, outcome "
                    f"{clean_text(r.get('outcome_text'))}, log score {num(r.get('log_score'))}. "
                    f"Cause: {clean_text(r.get('cause') or UNREVIEWED)}"
                )
        else:
            lines.append("No misses: every forecast did at least as well as a uniform one.")
        for r in records:
            if r.get("score_note"):
                lines += ["", f"The scores of {link(r)} are Unknown: {clean_text(r['score_note'])}."]
    for note in morning.get("notes", []):
        lines += ["", clean_text(note)]
    return lines + [""]


def render_lessons(state: dict[str, Any]) -> str:
    questions = state["questions"]
    lines = [LESSONS_HEADER.rstrip("\n"), ""]
    lines += render_calibration(questions)
    lines += ["## Mornings", "", "Newest first.", ""]
    mornings = state["mornings"]
    if not mornings:
        lines += ["No mornings yet.", ""]
    for day in sorted(mornings, reverse=True):
        lines += render_morning(day, mornings[day], questions)
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------- reviews


def clean_cause(answer: Any) -> str:
    """The first line of the model's answer, as plain text."""
    for raw in str(answer or "").splitlines():
        line = raw.strip().strip("\"'").strip()
        for mark in ("- ", "* ", "• "):
            if line.startswith(mark):
                line = line[len(mark):].strip()
        if line.lower().startswith("cause:"):
            line = line[len("cause:"):].strip()
        line = clean_text(line, CAUSE_CHARS)
        if line:
            return line
    return ""


def review_prompt(item: Resolved, reasoning: str | None) -> str:
    """What the reviewing model sees. Sent to the model only, never written to a file."""
    question, post, record = item.question, item.post, item.record
    title = str(question.get("title") or post.get("title") or "")
    label = question.get("label")
    if isinstance(label, str) and label.strip():
        title += f" (sub-question: {label})"
    outcome = question.get("resolution") if record["type"] == "multiple_choice" else record["outcome_text"]
    return "\n".join(
        [
            REVIEW_MARKER,
            "In one line of at most 25 words, name the most likely cause of the miss, in your own "
            "words. Do not quote or repeat the question, its resolution criteria or its options, and "
            "do not name any option or sub-question label. Use only the information below. Write "
            "plain text with no dashes, no list marks and no preamble.",
            "",
            f"Question: {clean_text(title, TITLE_CHARS)}",
            f"How it resolves: {clean_text(resolution_criteria(post, question), CRITERIA_CHARS) or 'Not given.'}",
            f"Outcome: {clean_text(outcome, 200)}",
            f"Our last forecast before the question closed: {forecast_description(record['type'], item.payload, question)}",
            f"Probability our forecast gave to the outcome: {pct(record['probability_on_outcome'])}",
            "The bot's own reasoning for this question:",
            reasoning[:REASONING_CHARS] if reasoning else "Not available.",
        ]
    )


def _author_id(comment: dict[str, Any]) -> Any:
    author = comment.get("author")
    if isinstance(author, dict):
        return author.get("id")
    return author if author is not None else comment.get("author_id")


def bot_reasoning(reader: MetaculusReader, user_id: int, post_id: int) -> str | None:
    """
    The bot's own reasoning: the latest comment of the bot's account on the
    post, from its forecasts section on. None if Metaculus does not have it.
    """
    answer = reader.get_json(
        "/comments/", {"post": post_id, "author": user_id, "limit": 50}, missing_ok=(400, 404)
    )
    results = answer.get("results") if isinstance(answer, dict) else answer
    if not isinstance(results, list):
        return None
    ours = [
        c for c in results
        if isinstance(c, dict) and _author_id(c) == user_id and isinstance(c.get("text"), str)
    ]
    if not ours:
        return None
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    latest = max(enumerate(ours), key=lambda pair: (parse_time(pair[1].get("created_at")) or oldest, pair[0]))[1]
    text = latest["text"]
    start = text.find("# FORECASTS")
    if start >= 0:
        text = text[start + len("# FORECASTS"):]
    return text.strip()[:REASONING_CHARS] or None


async def _ask(llm: MeteredLlm, items: list[Resolved], prompts: list[str]) -> list[str | None]:
    """One call per miss, one at a time. A failure gives None for that miss."""
    causes: list[str | None] = []
    for item, prompt in zip(items, prompts):
        record = item.record
        token = current_question.set(
            QuestionTag(record["question_id"], record["post_id"], question_url(record["post_id"]))
        )
        try:
            causes.append(clean_cause(await llm.invoke(prompt)) or None)
        except SpendCapReached:
            causes.append(None)
        except Exception as error:  # the cause is then Unknown (not reviewed)
            logger.warning(f"The review of question {record['question_id']} failed: {type(error).__name__}")
            causes.append(None)
        finally:
            current_question.reset(token)
    return causes


def review_worst_misses(
    misses: list[dict[str, Any]],
    *,
    reader: MetaculusReader,
    resolved: dict[int, Resolved],
    notes: list[str],
) -> int:
    """
    Gives each miss not yet reviewed a one-line cause, or "Unknown (not
    reviewed)". Returns the number of model calls made.
    """
    pending = [r for r in misses if r.get("cause") is None]
    if not pending:
        return 0
    items = [resolved[r["question_id"]] for r in pending if r["question_id"] in resolved]
    for record in pending:
        if record["question_id"] not in resolved:
            record["cause"] = UNREVIEWED  # read on an earlier run, so its question is not at hand

    def not_reviewed(reason: str) -> int:
        for item in items:
            item.record["cause"] = UNREVIEWED
        notes.append(f"The worst misses were not reviewed: {reason}.")
        return 0

    if not items:
        return 0
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        return not_reviewed("no OPENROUTER_API_KEY is set")
    try:
        cap, reserve = daily_cap_from_env()
        log_path, read_paths = spend_log_paths()
        meter = SpendMeter(
            cap=cap,
            reserve=reserve,
            log_path=log_path,
            read_paths=read_paths,
            run_mode="learning_loop",
            dry_run=False,
        )
    except (ValueError, SpendRecordUnreadable) as error:
        return not_reviewed(f"today's spending could not be read ({type(error).__name__})")
    llm = MeteredLlm(
        bot.HELPER_MODEL,
        purpose=REVIEW_PURPOSE,
        meter=meter,
        max_tokens=bot.HELPER_MAX_TOKENS,
        timeout=120,
        allowed_tries=1,
    )
    # Checked first at the longest the reasoning can be, so the cap is known
    # to have room before Metaculus is asked for the reasoning. The reviews,
    # like low-value questions, leave the high-value reserve free.
    worst_case = sum(
        (llm.worst_case_for_prompt(review_prompt(item, "x" * REASONING_CHARS)) for item in items),
        start=Decimal("0"),
    )
    if not meter.can_afford(worst_case, keep_free=meter.reserve):
        return not_reviewed(
            f"the daily cap had no room for them (up to US${money_display(worst_case)}, "
            "with the high-value reserve kept free)"
        )

    reasoning: dict[int, str | None] = {}
    try:
        me = reader.get_json("/users/me", missing_ok=(400, 404))
        user_id = me.get("id") if isinstance(me, dict) else None
        if _is_int(user_id):
            for item in items:
                reasoning[item.record["question_id"]] = bot_reasoning(reader, user_id, item.record["post_id"])
    except FetchStopped:
        pass  # reviewed without the reasoning; the stop is reported with the run

    prompts = [review_prompt(item, reasoning.get(item.record["question_id"])) for item in items]
    causes = asyncio.run(_ask(llm, items, prompts))
    failed = withheld = 0
    for item, cause in zip(items, causes):
        if cause is None:
            failed += 1
            item.record["cause"] = UNREVIEWED
        elif quotes_the_question(cause, *question_material(item.post, item.question)):
            withheld += 1
            item.record["cause"] = UNREVIEWED
        else:
            item.record["cause"] = cause
    if failed:
        notes.append(
            f"{failed} of the worst misses could not be reviewed: the model call failed, "
            "gave no answer or found no room under the daily cap."
        )
    if withheld:
        notes.append(f"{withheld} of the causes were withheld because they repeated the question's own words.")
    print(meter.summary_line())
    return len(items)


# ---------------------------------------------------------------- run


def run(
    *,
    now: datetime | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> int:
    moment = now or datetime.now(timezone.utc)
    day = moment.astimezone(BRISBANE).date().isoformat()
    # As in main.py: a cap variable the bot cannot use stops the run before
    # anything is fetched or spent.
    try:
        daily_cap_from_env()
    except ValueError as error:
        raise SetupError(f"Stopped before checking anything: {error}") from error
    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        raise SetupError("METACULUS_TOKEN is not set, so Metaculus cannot be asked anything.")

    state = load_state(STATE_PATH)
    ledger = read_ledger(LEDGER_PATH)
    known = set(state["questions"]) | set(state["left_out"])
    by_post: dict[int, list[LedgerQuestion]] = {}
    for question in ledger.values():
        if str(question.question_id) not in known and not question.test_area:
            by_post.setdefault(question.post_id, []).append(question)
    never = datetime.max.replace(tzinfo=timezone.utc)

    def oldest_forecast(post_id: int) -> tuple[datetime, int]:
        return min(q.first_recorded or never for q in by_post[post_id]), post_id

    posts = sorted(by_post, key=oldest_forecast)
    to_check, unchecked = posts[:MAX_POSTS_PER_MORNING], posts[MAX_POSTS_PER_MORNING:]

    reader = MetaculusReader(token, pause=pause)
    notes: list[str] = []
    resolved: dict[int, Resolved] = {}
    left_out: dict[str, int] = {}

    def leave_out(question_id: int, reason: str) -> None:
        state["left_out"][str(question_id)] = {"reason": reason, "checked_on": day}
        left_out[reason] = left_out.get(reason, 0) + 1

    checked = 0
    for post_id in to_check:
        try:
            post = reader.get_json(f"/posts/{post_id}/")
        except FetchStopped:
            break
        checked += 1
        if not isinstance(post, dict):
            continue  # not found: asked again next morning
        found = {q.get("id"): q for q in questions_in_post(post)}
        for ledger_question in sorted(by_post[post_id], key=lambda q: q.question_id):
            question = found.get(ledger_question.question_id)
            # Open, closed, upcoming or not found: nothing is read or written.
            if question is None or question.get("status") != "resolved":
                continue
            resolution = question.get("resolution")
            if not isinstance(resolution, str) or not resolution.strip():
                continue
            if in_test_area(post):
                leave_out(ledger_question.question_id, "in the unscored bot testing area")
            elif resolution in CANCELLED:
                leave_out(ledger_question.question_id, resolution)
            else:
                outcome = resolved_record(post, question, ledger_question, day)
                if isinstance(outcome, str):
                    leave_out(ledger_question.question_id, outcome)
                elif outcome is not None:
                    state["questions"][str(ledger_question.question_id)] = outcome.record
                    resolved[ledger_question.question_id] = outcome

    if unchecked:
        notes.append(
            f"{len(unchecked)} posts were not checked this morning (at most {MAX_POSTS_PER_MORNING} "
            "a morning); they wait for the next morning."
        )
    if left_out:
        parts = [f"{count} {reason}" for reason, count in sorted(left_out.items())]
        notes.append(f"Left out: {', '.join(parts)}.")

    morning = state["mornings"].setdefault(day, {"questions": [], "worst": [], "notes": []})
    morning["questions"] += list(resolved)
    records = [state["questions"][str(q)] for q in morning["questions"] if str(q) in state["questions"]]
    worst = worst_misses(records)
    morning["worst"] = [r["question_id"] for r in worst]
    calls = review_worst_misses(worst, reader=reader, resolved=resolved, notes=notes)
    # After the reviews, so a stop while reading the bot's reasoning is noted too.
    if reader.stopped is not None:
        notes.append(
            f"Stopped early: {reader.stopped.reason}. {checked} of {len(posts)} posts were checked "
            "and nothing was retried; the rest wait for the next morning."
        )
    morning["notes"] += [note for note in notes if note not in morning["notes"]]

    write_atomically(STATE_PATH, state_text(state))
    write_atomically(LESSONS_PATH, render_lessons(state))
    print(
        f"Learning loop for the morning of {day} (Brisbane): {checked} of {len(posts)} posts checked "
        f"in {reader.requests_made} requests; {len(resolved)} newly resolved, "
        f"{sum(left_out.values())} left out; {len(worst)} worst misses, {calls} model calls."
    )
    for note in notes:
        print(note)
    return 1 if reader.stopped is not None and reader.stopped.fatal else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        return run()
    except SetupError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
