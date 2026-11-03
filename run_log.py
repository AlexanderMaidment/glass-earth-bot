"""
What a run shows in its console, and where everything else goes.

The workflows run the bot in a public repository, where anyone signed in to
GitHub can read a run's log while the questions it forecast are still open.
So a live run on GitHub Actions (GITHUB_ACTIONS is "true" and the run posts
to Metaculus) shows only content-free lines, by allow-list:

- show(): the run's start and setup, the credit and cap lines, the ledger
  reveal counts, and why the run stopped, but only for the stops in
  PUBLIC_EXIT_STARTS, which the bot makes from its own settings;
- note(): for each tournament, how many questions were found, selected,
  skipped (by reason) and forecast, and for each question its ID, its
  address, its outcome (posted; failed, with the exception's type; or
  skipped, with the reason) and its cost.

An error shows as its exception's type (error_kind), never its message,
which can hold model output. Everything else that the bot, forecasting-tools,
litellm or any other code writes (research, reasoning, forecasts, the
template's report summary, warnings, tracebacks) goes only to a log file on
the runner's disk, under RUNNER_TEMP, which no workflow uploads or commits.
start_quiet() points the process's own stdout and stderr (file descriptors 1
and 2) at that file, so this holds for every writer, print() and native code
included, until the process ends.

A dry run, or any run off GitHub Actions, shows everything as before, and the
note() lines go to its log.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, TextIO

from sealed_ledger import public_url

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# The only stop messages a quiet console shows. The bot makes each from its
# own settings or the OpenRouter credit check, never from a question.
PUBLIC_EXIT_STARTS = (
    "Stopped before forecasting:",
    "--question-url is for dry runs only",
    "--max-questions must be at least 1",
    "Low OpenRouter credit:",
)
# Why a question was not forecast, in the order the counts give them.
SKIP_REASONS = ("not open", "already forecast", "not due", "daily cap")
DETAILS = "The details are only in the run's log file on the runner."

_console: TextIO | None = None


def quiet_wanted(publishing: bool, environ: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return publishing and environ.get("GITHUB_ACTIONS") == "true"


def is_quiet() -> bool:
    return _console is not None


def show(text: str) -> None:
    """An allow-listed, content-free line: in the console of every run."""
    if _console is not None:
        print(text, file=_console, flush=True)
    else:
        print(text)


def note(text: str) -> None:
    """An allow-listed, content-free line: in a quiet run's console, and in any other run's log."""
    if _console is not None:
        print(text, file=_console, flush=True)
    else:
        logger.info(text)


def error_kind(error: BaseException) -> str:
    """
    An exception by type only: its own, then the types it holds (in a group)
    or was raised from, never a message.
    """
    found: list[str] = []

    def walk(item: BaseException, depth: int) -> None:
        if isinstance(item, BaseExceptionGroup) and depth < 10:
            for inner in item.exceptions:
                walk(inner, depth + 1)
            return
        steps = 0
        while item.__cause__ is not None and steps < 10:
            item, steps = item.__cause__, steps + 1
        found.append(type(item).__name__)

    walk(error, 0)
    name = type(error).__name__
    inner = sorted(set(found) - {name})
    return f"{name} ({', '.join(inner)})" if inner else name


def skipped(reason: str) -> str:
    if reason not in SKIP_REASONS:
        raise ValueError(f"Not a reason a question is skipped: {reason!r}")
    return f"skipped ({reason})"


def failed(error: BaseException) -> str:
    return f"failed ({error_kind(error)})"


def done(publishing: bool) -> str:
    return "posted" if publishing else "forecast, not posted (dry run)"


def question_line(question_id: Any, url: Any, post_id: Any, outcome: str, cost: str | None = None) -> str:
    """One question: its ID, its address cut to the question number, its outcome and cost."""
    number = question_id if isinstance(question_id, int) and not isinstance(question_id, bool) else "?"
    text = f"Question {number} ({public_url(url, post_id) or 'no address'}): {outcome}"
    return text if cost is None else f"{text}, cost {cost}"


def counts_line(
    *, found: int, selected: int, skips: Mapping[str, int], posted: int, failures: int, publishing: bool
) -> str:
    reasons = ", ".join(f"{skips.get(reason, 0)} {reason}" for reason in SKIP_REASONS)
    verb = "posted" if publishing else "forecast in a dry run"
    return (
        f"Questions: {found} found, {selected} selected, {sum(skips.values())} skipped "
        f"({reasons}), {posted} {verb}, {failures} failed."
    )


def log_file_path(environ: Mapping[str, str] | None = None) -> Path:
    """Under RUNNER_TEMP on GitHub Actions, otherwise the system's temporary folder; never in the checkout."""
    environ = os.environ if environ is None else environ
    folder = Path(environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    try:
        inside = folder.resolve().is_relative_to(Path.cwd().resolve())
    except OSError:
        inside = True
    if inside:
        folder = Path(tempfile.gettempdir())
    return folder / f"glass-earth-run-{environ.get('GITHUB_RUN_ID') or os.getpid()}.log"


def start_quiet(environ: Mapping[str, str] | None = None) -> Path:
    """
    For the rest of this process, send everything written to stdout or
    stderr, and every log record, to the log file, and keep the console for
    show() and note() alone. Returns the log file's path.
    """
    global _console
    path = log_file_path(environ)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    log = open(path, "a", encoding="utf-8", errors="backslashreplace", buffering=1)
    console = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="backslashreplace", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = log
    sys.stderr = log
    _log_to(log)
    sys.excepthook = _uncaught
    threading.excepthook = _uncaught_in_thread
    _console = console
    return path


def _log_to(stream: TextIO) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Handlers that libraries made before now (litellm's among them) write
    # to the log file too.
    for candidate in list(logging.root.manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger):
            for existing in candidate.handlers:
                if isinstance(existing, logging.StreamHandler) and not isinstance(existing, logging.FileHandler):
                    existing.setStream(stream)
    logging.captureWarnings(True)


def _uncaught(kind: type[BaseException], error: BaseException, trace: Any) -> None:
    logger.error("The run stopped on an uncaught exception", exc_info=(kind, error, trace))
    show(f"Stopped: {error_kind(error)}. {DETAILS}")


def _uncaught_in_thread(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    logger.error(
        "A thread stopped on an uncaught exception",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    show(f"A thread stopped: {args.exc_type.__name__}. {DETAILS}")


def public_exit(stop: SystemExit) -> int:
    """The exit code for a SystemExit in a quiet run, and its line, if its message may be shown."""
    code = stop.code
    if code is None or code == 0:
        return 0
    if isinstance(code, int):
        show(f"Stopped with exit code {code}. {DETAILS}")
        return code
    text = str(code)
    logger.error(text)
    show(text if text.startswith(PUBLIC_EXIT_STARTS) else f"Stopped. {DETAILS}")
    return 1
