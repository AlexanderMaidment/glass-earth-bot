"""
The coverage alert of glass-earth-bot: once a day, on GitHub Actions, it names
every question in the bot's tournaments that has no forecast from us when it
should have one, so that a missed question is seen the next morning instead
of at scoring time.

.github/workflows/coverage_alert.yaml runs it daily from the top of the
repository, only while the repository variable GLASS_EARTH_COVERAGE_ALERT is
true:

    uv run --locked python scripts/coverage_alert.py

What it checks, in the tournaments the bot's tournament mode forecasts
(main.py: FutureEval, MiniBench, and Market Pulse while GLASS_EARTH_MARKET_PULSE
is true):
1. Alex's rule: an open question that still has no forecast from us 24 hours
   (ALERT_AFTER_HOURS) after it opened.
2. A question that closed in the last 36 hours (MISSED_LOOKBACK_HOURS: a day,
   plus room for GitHub starting a daily schedule hours late) with no forecast
   from us. FutureEval and MiniBench questions are open for about 3 hours, so
   rule 1 alone would never see them.
"Ours" means Metaculus lists a forecast of ours on the question (the same test
as the bot's own already-forecast check) or forecasts/ledger.jsonl has a line
for it. Annulled and ambiguous questions are left out.

How it alerts: the bot's existing path (credit.py). Each question gets a
GitHub Actions error line and a line on the run's summary page, and the run
ends with a failing exit code, so GitHub emails the owner. A check that could
not read Metaculus fails the same way, so a broken alert is seen too.

It reads Metaculus with GETs only, with the bot's METACULUS_TOKEN, one at a
time, PAUSE_SECONDS apart, never retried: the first 429, refusal, error or
page that is not JSON stops it. It never forecasts, posts, comments or spends,
and it changes no file in the repository. Metaculus's terms of use do not
allow republishing its site material, so questions are named by their ID and
link only, never by their title or text.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import requests  # noqa: E402
from forecasting_tools import MetaculusClient  # noqa: E402

from ledger import LEDGER_PATH, ForecastLedger  # noqa: E402

API_BASE = "https://www.metaculus.com/api"
PAUSE_SECONDS = 4.0
REQUEST_TIMEOUT_SECONDS = 30
PAGE_SIZE = 100
MAX_PAGES = 20
ALERT_AFTER_HOURS = 24
MISSED_LOOKBACK_HOURS = 36
# main.py's MARKET_PULSE_ENV: Market Pulse joins a tournament run only when true.
MARKET_PULSE_ENV = "GLASS_EARTH_MARKET_PULSE"
# The MetaculusClient attributes main.py's tournament mode forecasts, in order
# (a test checks the two agree), with the names this alert shows.
TOURNAMENT_ATTRIBUTES = {
    "CURRENT_AI_COMPETITION_ID": "FutureEval",
    "CURRENT_MINIBENCH_ID": "MiniBench",
    "CURRENT_MARKET_PULSE_ID": "Market Pulse",
}
NOT_SCORED = ("annulled", "ambiguous")


class CheckStopped(RuntimeError):
    """Metaculus could not be read. The message never holds the token or a reply."""


def tournaments(environ: Mapping[str, str]) -> list[tuple[str, int | str]]:
    """The tournaments the bot's tournament mode forecasts, as (name, id or slug)."""
    found = []
    for attribute, name in TOURNAMENT_ATTRIBUTES.items():
        if attribute == "CURRENT_MARKET_PULSE_ID" and environ.get(MARKET_PULSE_ENV, "").strip().lower() != "true":
            continue
        found.append((name, getattr(MetaculusClient, attribute)))
    return found


class MetaculusReader:
    """Read-only GETs to the Metaculus API, one at a time with a pause, never retried."""

    def __init__(
        self,
        token: str,
        *,
        get: Callable[..., Any] = requests.get,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._get = get
        self._pause = pause
        self.requests_made = 0

    def _page(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self.requests_made:
            self._pause(PAUSE_SECONDS)
        self.requests_made += 1
        try:
            response = self._get(
                f"{API_BASE}/posts/",
                params=params,
                headers={"Authorization": f"Token {self._token}", "Accept-Language": "en"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise CheckStopped(f"a request failed ({type(error).__name__})") from None
        status = response.status_code
        if status == 429:
            raise CheckStopped("Metaculus answered 429 (too many requests)")
        if status in (401, 403):
            raise CheckStopped(f"Metaculus refused the request (HTTP {status})")
        if status != 200:
            raise CheckStopped(f"Metaculus answered HTTP {status}")
        try:
            body = response.json()
        except ValueError:
            raise CheckStopped("Metaculus answered with a page that is not JSON, possibly an anti-bot check") from None
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise CheckStopped("Metaculus answered without a list of posts")
        return [p for p in results if isinstance(p, dict)]

    def posts(self, tournament: int | str, statuses: list[str]) -> list[dict[str, Any]]:
        """Every post in the tournament with one of the statuses: all pages, not only the first."""
        found: list[dict[str, Any]] = []
        for page in range(MAX_PAGES):
            batch = self._page(
                {
                    "tournaments": [tournament],
                    "statuses": statuses,
                    "limit": PAGE_SIZE,
                    "offset": page * PAGE_SIZE,
                    "order_by": "-published_time",
                    "with_cp": "false",
                }
            )
            found += batch
            if len(batch) < PAGE_SIZE:
                return found
        raise CheckStopped(f"more than {MAX_PAGES * PAGE_SIZE} posts in one listing, so it is incomplete")


def questions_in_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    """The questions of a post the bot forecasts: a single one, a group's, or a conditional's two."""
    found: list[dict[str, Any]] = []
    if isinstance(post.get("question"), dict):
        found.append(post["question"])
    group = post.get("group_of_questions")
    if isinstance(group, dict):
        found += [q for q in group.get("questions") or [] if isinstance(q, dict)]
    conditional = post.get("conditional")
    if isinstance(conditional, dict):
        found += [conditional[k] for k in ("question_yes", "question_no") if isinstance(conditional.get(k), dict)]
    return found


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def utc_text(moment: datetime | None) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if moment else "Unknown"


def forecast_on_metaculus(question: dict[str, Any]) -> bool:
    """The bot's own test (forecasting_tools' already_forecasted): our forecast history is not empty."""
    mine = question.get("my_forecasts")
    history = mine.get("history") if isinstance(mine, dict) else None
    return bool(history)


@dataclass(frozen=True)
class Gap:
    kind: str  # "open" (rule 1) or "missed" (rule 2)
    tournament: str
    question_id: int
    post_id: int
    opened: datetime | None
    closed: datetime | None

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"

    def line(self, now: datetime) -> str:
        if self.kind == "open":
            hours = (now - self.opened).total_seconds() / 3600 if self.opened else 0
            return (
                f"{self.tournament}: question {self.question_id} ({self.url}) opened {utc_text(self.opened)} "
                f"and still has no forecast from us after {hours:.0f} hours"
            )
        return (
            f"{self.tournament}: question {self.question_id} ({self.url}) opened {utc_text(self.opened)}, "
            f"closed {utc_text(self.closed)} with no forecast from us"
        )


def find_gaps(
    name: str,
    open_posts: list[dict[str, Any]],
    closed_posts: list[dict[str, Any]],
    ours: set[int],
    now: datetime,
) -> tuple[list[Gap], int]:
    """The gaps in one tournament, and how many open questions were checked."""
    gaps: list[Gap] = []
    checked = 0
    for post in open_posts:
        for q in questions_in_post(post):
            if q.get("status") != "open" or not isinstance(q.get("id"), int):
                continue
            checked += 1
            opened = parse_time(q.get("open_time"))
            if opened is None or now - opened < timedelta(hours=ALERT_AFTER_HOURS):
                continue
            if forecast_on_metaculus(q) or q["id"] in ours:
                continue
            gaps.append(Gap("open", name, q["id"], post.get("id"), opened, None))
    for post in closed_posts:
        for q in questions_in_post(post):
            if q.get("status") not in ("closed", "resolved") or not isinstance(q.get("id"), int):
                continue
            if str(q.get("resolution") or "").lower() in NOT_SCORED:
                continue
            closed = parse_time(q.get("actual_close_time")) or parse_time(q.get("scheduled_close_time"))
            opened = parse_time(q.get("open_time"))
            if closed is None or opened is None or not (now - timedelta(hours=MISSED_LOOKBACK_HOURS) <= closed <= now):
                continue
            if forecast_on_metaculus(q) or q["id"] in ours:
                continue
            gaps.append(Gap("missed", name, q["id"], post.get("id"), opened, closed))
    return gaps, checked


def _workflow_text(text: str) -> str:
    # GitHub's escaping for the message of a workflow command (as in credit.py).
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _summary(step_summary: str | None, lines: list[str]) -> None:
    if not step_summary:
        return
    try:
        with open(step_summary, "a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n\n")
    except OSError as error:
        print(f"Could not write to the run's summary page: {type(error).__name__}")


def run(
    *,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
    get: Callable[..., Any] = requests.get,
    pause: Callable[[float], None] = time.sleep,
    ledger_path: Path = LEDGER_PATH,
    step_summary: str | None = None,
) -> int:
    """One check. Returns the exit code: 0 with no gap, 1 with a gap or when the check failed."""
    environ = os.environ if environ is None else environ
    now = now if now is not None else datetime.now(timezone.utc)
    token = (environ.get("METACULUS_TOKEN") or "").strip()
    if not token:
        message = "METACULUS_TOKEN is not set, so Metaculus was not asked. Nothing was checked."
        print(f"::error title=Coverage check failed::{_workflow_text(message)}")
        _summary(step_summary, [f"**Coverage check failed.** {message}"])
        return 1
    ours = ForecastLedger(Path(ledger_path)).question_ids()
    reader = MetaculusReader(token, get=get, pause=pause)
    gaps: list[Gap] = []
    checked: dict[str, int] = {}
    try:
        for name, tournament in tournaments(environ):
            open_posts = reader.posts(tournament, ["open"])
            closed_posts = reader.posts(tournament, ["closed", "resolved"])
            found, count = find_gaps(name, open_posts, closed_posts, ours, now)
            gaps += found
            checked[f"{name} ({tournament})"] = count
    except CheckStopped as stop:
        message = f"Could not read Metaculus: {stop}. Nothing was checked; nothing is retried."
        print(f"::error title=Coverage check failed::{_workflow_text(message)}")
        _summary(step_summary, [f"**Coverage check failed.** {message}"])
        return 1
    where = ", ".join(f"{k}: {v} open" for k, v in checked.items())
    if not gaps:
        print(
            f"Coverage OK at {utc_text(now)}. {where}. Every open question has our forecast or opened under "
            f"{ALERT_AFTER_HOURS} hours ago, and none closed without one in the last {MISSED_LOOKBACK_HOURS} hours."
        )
        return 0
    late = [g for g in gaps if g.kind == "open"]
    missed = [g for g in gaps if g.kind == "missed"]
    for gap in gaps:
        print(f"::error title=Coverage gap::{_workflow_text(gap.line(now))}")
    lines = [f"**Coverage gap at {utc_text(now)}.** {where}."]
    if late:
        lines += ["", f"Open {ALERT_AFTER_HOURS} hours or more with no forecast from us ({len(late)}):", ""]
        lines += [f"- {g.line(now)}" for g in late]
    if missed:
        lines += ["", f"Closed in the last {MISSED_LOOKBACK_HOURS} hours with no forecast from us ({len(missed)}):", ""]
        lines += [f"- {g.line(now)}" for g in missed]
    _summary(step_summary, lines)
    print(
        f"Coverage: {len(late)} open question(s) without our forecast {ALERT_AFTER_HOURS} hours after opening, "
        f"{len(missed)} closed in the last {MISSED_LOOKBACK_HOURS} hours without one. {where}."
    )
    return 1


def main() -> int:
    return run(step_summary=os.getenv("GITHUB_STEP_SUMMARY"))


if __name__ == "__main__":
    sys.exit(main())
