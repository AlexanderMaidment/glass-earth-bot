"""
Go live in two steps, each started by hand.

Step 1 (run when .env is ready):
    uv run --locked python scripts/go_live.py

  1. Checks that .env exists, is ignored by git and holds real values for
     METACULUS_TOKEN and OPENROUTER_API_KEY.
  2. Sets both as GitHub Actions secrets with `gh secret set`, passing each
     value on standard input. No value is ever printed, logged or put on a
     command line.
  3. Asks OpenRouter how much credit the key has left (a free call).
  4. Runs one paid dry run on this computer: one question from the bot
     testing area, under a US$2.00 cap, posting nothing to Metaculus.
  5. Prints that forecast, its comment and its cost, then stops.

Step 2 (only after you have read the dry run and said yes):
    uv run --locked python scripts/go_live.py --publish

  Asks you to type PUBLISH, then pushes this branch to main (fast-forward
  only) and sets the repository variable GLASS_EARTH_LIVE to true, which
  switches on the scheduled tournament workflow.

The dry run in step 1 runs here, from this branch, so it does not need the
branch pushed first. Step 2 pushes it, because GitHub runs scheduled
workflows only from the default branch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPOSITORY = "AlexanderMaidment/glass-earth-bot"
BRANCH = "engine-one-ready"
SECRET_NAMES = ("METACULUS_TOKEN", "OPENROUTER_API_KEY")
PLACEHOLDERS = {"", "1234567890", "REPLACE_ME", "your-token-here", "your-api-key-here"}
# One test-area question costs about US$0.27 with Opus (an assumption-based
# estimate) and at most about US$1 (the cap check's calculated worst case), so
# the dry run gets a US$2.00 cap and no reserve (test questions are low value).
DRY_RUN_CAP_USD = "2.00"
CONFIRM_WORD = "PUBLISH"
MASK = "[hidden]"
# Written by a good step 1, checked by step 2 (git ignores forecasts/dry_run/).
MARKER = "forecasts/dry_run/go_live_step1_ok.json"

Runner = Callable[..., subprocess.CompletedProcess]


class GoLiveError(RuntimeError):
    pass


@dataclass
class Context:
    repo: Path
    run: Runner
    out: Callable[[str], None]
    ask: Callable[[str], str]
    fetch_key_info: Callable[[str], dict[str, Any]]
    python: str


def default_fetch_key_info(api_key: str) -> dict[str, Any]:
    """OpenRouter's own report on a key: its limit and what is left. Free."""
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def mask(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, MASK)
    return text


def read_env_file(path: Path) -> dict[str, str]:
    """The two values from .env, read here and kept in memory only."""
    from dotenv import dotenv_values

    if not path.is_file():
        raise GoLiveError(f"{path.name} was not found in {path.parent}. Copy .env.example to .env first.")
    values = dotenv_values(path)
    found: dict[str, str] = {}
    for name in SECRET_NAMES:
        value = (values.get(name) or "").strip()
        if value in PLACEHOLDERS:
            raise GoLiveError(f"{name} in .env is empty or still a placeholder.")
        found[name] = value
    return found


def _git(ctx: Context, *args: str) -> subprocess.CompletedProcess:
    return ctx.run(["git", *args], cwd=ctx.repo, capture_output=True, text=True)


def check_repository(ctx: Context) -> None:
    if _git(ctx, "ls-files", "--error-unmatch", ".env").returncode == 0:
        raise GoLiveError(".env is tracked by git. Remove it from git before going further.")
    if _git(ctx, "check-ignore", "-q", ".env").returncode != 0:
        raise GoLiveError(".env is not ignored by git. Add it to .gitignore before going further.")
    remote = _git(ctx, "remote", "get-url", "origin")
    if remote.returncode != 0 or REPOSITORY not in remote.stdout:
        raise GoLiveError(f"The origin remote is not {REPOSITORY}.")
    gh = ctx.run(["gh", "--version"], capture_output=True, text=True)
    if gh.returncode != 0:
        raise GoLiveError("The GitHub CLI (gh) is not available.")


def set_secrets(ctx: Context, values: dict[str, str]) -> None:
    secrets = list(values.values())
    for name in SECRET_NAMES:
        # gh reads the value from standard input when --body is not given.
        result = ctx.run(
            ["gh", "secret", "set", name, "--repo", REPOSITORY],
            cwd=ctx.repo,
            input=values[name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = mask((result.stderr or result.stdout or "").strip(), secrets)
            raise GoLiveError(f"Setting the secret {name} failed: {detail}")
        ctx.out(f"Set the GitHub Actions secret {name} (value not shown).")


def report_credit(ctx: Context, api_key: str) -> None:
    try:
        info = ctx.fetch_key_info(api_key).get("data", {})
    except Exception as error:  # a failed check must not stop the dry run
        ctx.out(f"Could not read the OpenRouter key's credit: {type(error).__name__}.")
        return
    limit = info.get("limit")
    remaining = info.get("limit_remaining")
    usage = info.get("usage")
    ctx.out(
        "OpenRouter reports for this key: "
        f"limit {limit if limit is not None else 'none set'}, "
        f"left {remaining if remaining is not None else 'Unknown'}, "
        f"used {usage if usage is not None else 'Unknown'} (US$)."
    )


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def run_dry_run(
    ctx: Context, values: dict[str, str], question_url: str | None
) -> subprocess.CompletedProcess:
    command = [
        ctx.python,
        "main.py",
        "--mode",
        "test_questions",
        "--dry-run",
        "--max-questions",
        "1",
        "--daily-cap-usd",
        DRY_RUN_CAP_USD,
        "--high-value-reserve-usd",
        "0",
    ]
    if question_url:
        command += ["--question-url", question_url]
    env = dict(os.environ)
    env.update(values)
    # The values are already in the environment; main.py need not read .env.
    env["GLASS_EARTH_SKIP_DOTENV"] = "1"
    env.pop("GITHUB_ACTIONS", None)
    # On Windows a captured child writes in the ANSI code page, and the bot's
    # banner (an emoji) would stop the run; make the child write UTF-8.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    ctx.out(f"Running one paid dry run (cap US${DRY_RUN_CAP_USD}, nothing is posted): {' '.join(command[1:])}")
    return ctx.run(
        command,
        cwd=ctx.repo,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def show_result(
    ctx: Context, result: subprocess.CompletedProcess, values: dict[str, str], started: float
) -> bool:
    secrets = list(values.values())
    dry_run_folder = ctx.repo / "forecasts" / "dry_run"
    dry_run_folder.mkdir(parents=True, exist_ok=True)
    log = mask((result.stdout or "") + (result.stderr or ""), secrets)
    (dry_run_folder / "go_live_dry_run.log").write_text(log, encoding="utf-8")
    tail = "\n".join(log.strip().splitlines()[-12:])
    ctx.out("Last lines of the dry run log (full log: forecasts/dry_run/go_live_dry_run.log):")
    ctx.out(tail)
    # Only files this run wrote, never an older dry run's. File times can lag
    # the clock slightly on Windows, hence the two seconds.
    report_file = _newest(
        [p for p in dry_run_folder.glob("Forecasts-for-*.json") if p.stat().st_mtime >= started - 2]
    )
    ledger_file = dry_run_folder / "ledger.jsonl"
    if result.returncode != 0 or report_file is None or not ledger_file.exists():
        ctx.out("The dry run did not produce a forecast. Nothing was published. Read the log above.")
        return False
    reports = json.loads(report_file.read_text(encoding="utf-8"))
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(started)))
    lines = [
        line
        for line in (
            json.loads(l) for l in ledger_file.read_text(encoding="utf-8").splitlines() if l.strip()
        )
        if str(line.get("recorded_at_utc", "")) >= started_utc
    ]
    if not reports or not lines:
        ctx.out("The dry run did not produce a forecast. Nothing was published.")
        return False
    report, line = reports[-1], lines[-1]
    ctx.out("")
    ctx.out("=" * 72)
    ctx.out("ONE DRY-RUN FORECAST (nothing was posted to Metaculus)")
    ctx.out(f"Question: {line.get('title')} {('(' + line['group_option'] + ')') if line.get('group_option') else ''}")
    ctx.out(f"URL: {line.get('url')}")
    ctx.out(f"Forecast: {str(line.get('final_forecast_readable')).strip()}")
    ctx.out(f"Model: {line.get('forecaster_model')} ({line.get('route')})")
    ctx.out(f"Cost: US${line.get('cost_usd')} over {line.get('model_calls')} model calls "
            f"(the cap counted US${line.get('cost_usd_cap_basis')})")
    ctx.out("-" * 72)
    ctx.out("Comment the bot would post (private on Metaculus):")
    ctx.out(mask(str(report.get("explanation", "")), secrets))
    ctx.out("=" * 72)
    return True


def step_one(ctx: Context, question_url: str | None) -> int:
    check_repository(ctx)
    values = read_env_file(ctx.repo / ".env")
    ctx.out("Read METACULUS_TOKEN and OPENROUTER_API_KEY from .env (values not shown).")
    set_secrets(ctx, values)
    report_credit(ctx, values["OPENROUTER_API_KEY"])
    started = time.time()
    result = run_dry_run(ctx, values, question_url)
    ok = show_result(ctx, result, values, started)
    ctx.out("")
    ctx.out("STOPPED. Nothing has been published and the tournament workflow is still off.")
    if ok:
        commit = _git(ctx, "rev-parse", "HEAD").stdout.strip()
        marker = ctx.repo / MARKER
        marker.write_text(
            json.dumps({"dry_run_ok_at_commit": commit, "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
            encoding="utf-8",
        )
        ctx.out("If you are happy with this forecast and comment, run:")
        ctx.out("    uv run --locked python scripts/go_live.py --publish")
    return 0 if ok else 1


def step_two(ctx: Context) -> int:
    check_repository(ctx)
    branch = _git(ctx, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != BRANCH:
        raise GoLiveError(f"Check out {BRANCH} first (now on {branch}).")
    if _git(ctx, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise GoLiveError("There are uncommitted changes to tracked files. Commit or undo them first.")
    marker = ctx.repo / MARKER
    head = _git(ctx, "rev-parse", "HEAD").stdout.strip()
    if not marker.exists():
        raise GoLiveError("Run step 1 (the dry run) first.")
    if json.loads(marker.read_text(encoding="utf-8")).get("dry_run_ok_at_commit") != head:
        raise GoLiveError("The code changed after the last good dry run. Run step 1 again.")
    ctx.out("This will push this branch to main on GitHub and set GLASS_EARTH_LIVE=true,")
    ctx.out("which starts the tournament workflow every 20 minutes, spending up to the daily cap.")
    answer = ctx.ask(f"Type {CONFIRM_WORD} to go live, anything else to stop: ")
    if answer.strip() != CONFIRM_WORD:
        ctx.out("Stopped. Nothing was pushed or changed.")
        return 1
    fetch = _git(ctx, "fetch", "origin", "main")
    if fetch.returncode != 0:
        raise GoLiveError("Could not fetch origin/main.")
    if _git(ctx, "merge-base", "--is-ancestor", "origin/main", "HEAD").returncode != 0:
        raise GoLiveError("main on GitHub has commits this branch lacks. Merge them first.")
    push = _git(ctx, "push", "origin", f"{BRANCH}:main")
    if push.returncode != 0:
        raise GoLiveError(f"The push failed: {push.stderr.strip()}")
    ctx.out("Pushed this branch to main.")
    variable = ctx.run(
        ["gh", "variable", "set", "GLASS_EARTH_LIVE", "--body", "true", "--repo", REPOSITORY],
        cwd=ctx.repo,
        capture_output=True,
        text=True,
    )
    if variable.returncode != 0:
        raise GoLiveError(f"Setting GLASS_EARTH_LIVE failed: {variable.stderr.strip()}")
    ctx.out("Set GLASS_EARTH_LIVE=true. The tournament workflow now runs on its schedule.")
    ctx.out("To pause: gh variable set GLASS_EARTH_LIVE --body false --repo " + REPOSITORY)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    repo: Path | None = None,
    run: Runner = subprocess.run,
    out: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
    fetch_key_info: Callable[[str], dict[str, Any]] = default_fetch_key_info,
) -> int:
    parser = argparse.ArgumentParser(description="Go live in two steps (see the top of this file)")
    parser.add_argument("--publish", action="store_true", help="Step 2: push to main and switch on")
    parser.add_argument("--question-url", default=None, help="Dry-run this question instead of the first test question")
    args = parser.parse_args(argv)
    ctx = Context(
        repo=(repo or Path(__file__).resolve().parent.parent),
        run=run,
        out=out,
        ask=ask,
        fetch_key_info=fetch_key_info,
        python=sys.executable,
    )
    try:
        return step_two(ctx) if args.publish else step_one(ctx, args.question_url)
    except GoLiveError as error:
        out(f"Stopped: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
