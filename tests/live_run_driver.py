"""
A run of main.main() as GitHub Actions would start it, in a process of its
own, with the offline fakes from support.py: SYNTHETIC questions, model
answers, keys and costs, and no network. tests/test_run_log.py reads what
this process writes to stdout and stderr, which is what the run's log on
GitHub would show.

    python tests/live_run_driver.py <scenario> <folder> [main.py arguments ...]

Scenarios:
- batch: every question is forecast, and each model call also writes its
  prompt everywhere it could (print, stderr, the file descriptors,
  sys.__stdout__, a warning and litellm's logger), as a chatty library might;
- failure: the forecast of one question fails with model output in the
  exception's message;
- crash: the list of questions cannot be fetched, with question text in the
  exception's message;
- local: like batch, but not on GitHub Actions.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import warnings
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support  # noqa: E402  (blocks the network and keeps real keys out first)

MARKER = "SYNTHETIC-CONTENT-MARKER"
OPTIONS = ["SYNTHETIC Amberflow", "SYNTHETIC Brightwater", "SYNTHETIC Coldbrook"]


def binary_post(n: int) -> dict:
    title = f"SYNTHETIC: Will made-up event number {n} happen?"
    return {
        "id": 7000 + n,
        "title": title,
        "projects": {
            "tournament": [{"id": 77002, "slug": "fall-futureeval-2026", "name": "SYNTHETIC tournament"}],
            "default_project": {"id": 77002, "visibility": "normal"},
        },
        "question": {
            "id": 70000 + n,
            "title": title,
            "type": "binary",
            "status": "open",
            "description": "SYNTHETIC background.",
            "resolution_criteria": "SYNTHETIC: resolves Yes if the made-up event happens.",
            "fine_print": "SYNTHETIC fine print.",
            "open_time": "2026-10-07T20:00:00Z",
            "scheduled_close_time": "2099-01-01T10:00:00Z",
            "scheduled_resolve_time": "2099-01-01T20:00:00Z",
            "question_weight": 1.0,
            "my_forecasts": {"history": [], "latest": None},
        },
    }


def choice_post() -> dict:
    post = binary_post(9)
    post["title"] = post["question"]["title"] = "SYNTHETIC: Which made-up river floods first?"
    post["question"].update(type="multiple_choice", options=list(OPTIONS), group_variable="SYNTHETIC river")
    return post


def page() -> dict:
    groups = support.load_fixture("synthetic_group_posts.json")["results"]
    posts = [binary_post(n) for n in (1, 2, 3)] + [choice_post()] + groups
    return {"next": None, "previous": None, "results": posts}


def main() -> int:
    scenario, folder, *argv = sys.argv[1:]
    # As on GitHub: the checkout is the working folder, and RUNNER_TEMP is
    # outside it.
    work = Path(folder)
    (work / "checkout").mkdir()
    os.chdir(work / "checkout")
    os.environ.update(
        RUNNER_TEMP=str(work / "runner_temp"),
        GITHUB_RUN_ID="4242",
        GITHUB_STEP_SUMMARY=str(work / "step_summary.md"),
        METACULUS_TOKEN=support.FAKE_TOKEN,
        OPENROUTER_API_KEY="sk-or-synthetic-driver-key",
        GLASS_EARTH_LEDGER_KEY=support.FAKE_LEDGER_KEY_TEXT,
    )
    if scenario != "local":
        os.environ["GITHUB_ACTIONS"] = "true"

    import requests

    import credit
    import forecasting_tools.ai_models.general_llm as general_llm

    models = support.SyntheticModels()
    metaculus = support.FakeMetaculus(page())
    answer = models.acompletion

    async def acompletion(messages, **kwargs):
        prompt = support.prompt_text(messages)
        if scenario in ("batch", "local"):
            # Everything a chatty library could write, with question text in it.
            print(f"stray print: {prompt[:120]}")
            print(f"stray stderr: {prompt[:120]}", file=sys.stderr)
            os.write(1, f"stray fd 1: {prompt[:120]}\n".encode("utf-8", "replace"))
            os.write(2, f"stray fd 2: {prompt[:120]}\n".encode("utf-8", "replace"))
            sys.__stdout__.write(f"stray __stdout__: {prompt[:120]}\n")
            sys.__stdout__.flush()
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(f"stray warning: {prompt[:120]}")
            logging.getLogger("LiteLLM").warning(f"stray LiteLLM warning: {prompt[:120]}")
        if scenario == "failure" and "professional forecaster" in prompt and "number 2" in prompt:
            raise RuntimeError(f"{MARKER} model output: {prompt[:300]}")
        return await answer(messages, **kwargs)

    general_llm.acompletion = acompletion
    real_get = metaculus.get

    def get(url, params=None, **kwargs):
        if scenario == "crash" and url.endswith("/posts/"):
            raise RuntimeError(f"{MARKER} while reading SYNTHETIC: Will made-up event number 1 happen?")
        return real_get(url, params, **kwargs)

    requests.get, requests.post = get, metaculus.post
    key = support.FakeOpenRouterKey()
    key.report = {"data": {"limit": Decimal("20"), "limit_remaining": Decimal("12.50"), "usage": Decimal("7.50")}}
    credit.fetch_key_info = key

    import main

    return main.main(copy.copy(argv))


if __name__ == "__main__":
    sys.exit(main())
