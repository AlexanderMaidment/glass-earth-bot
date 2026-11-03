"""
Shared fakes for the offline tests. Nothing here touches the network, reads
.env, or spends money:
- every socket connection is refused;
- litellm's acompletion is replaced by SyntheticModels, which writes canned,
  SYNTHETIC answers and token counts;
- requests.get / requests.post (the Metaculus client) are replaced by
  FakeMetaculus, which serves SYNTHETIC fixtures and records what was posted.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import socket
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Before anything imports main.py or litellm: never read a real .env.
os.environ["GLASS_EARTH_SKIP_DOTENV"] = "1"
os.environ["LITELLM_MODE"] = "PRODUCTION"
# litellm otherwise downloads its model price list when imported.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
for _name in (
    "METACULUS_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_ACTIONS",
    "GITHUB_SHA",
    "GITHUB_SERVER_URL",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ID",
    "GLASS_EARTH_DAILY_CAP_USD",
    "GLASS_EARTH_DAILY_CAP_UNTIL_UTC",
    "GLASS_EARTH_HIGH_VALUE_RESERVE_USD",
    "GLASS_EARTH_MARKET_PULSE",
    "GLASS_EARTH_ENSEMBLE_SIZE",
    "GLASS_EARTH_HIGH_VALUE_MODEL",
):
    os.environ.pop(_name, None)


_real_connect = socket.socket.connect


def _is_loopback(address: Any) -> bool:
    host = address[0] if isinstance(address, tuple) and address else address
    return host in ("127.0.0.1", "::1", "localhost")


def _connect_loopback_only(self: socket.socket, address: Any) -> Any:
    # asyncio on Windows connects a socket pair over loopback; nothing else.
    if not _is_loopback(address):
        raise AssertionError(f"The offline tests must not open network connections ({address})")
    return _real_connect(self, address)


def _refuse_network(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("The offline tests must not open network connections")


socket.socket.connect = _connect_loopback_only  # type: ignore[assignment]
socket.create_connection = _refuse_network  # type: ignore[assignment]

import requests  # noqa: E402
from litellm import ModelResponse  # noqa: E402
from litellm.types.utils import Choices, Message, Usage  # noqa: E402

import forecasting_tools.ai_models.general_llm as general_llm  # noqa: E402
from forecasting_tools import MetaculusClient  # noqa: E402

from spend import current_question  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_TOKEN = "synthetic-test-token-not-real"

MetaculusClient._sleep_between_requests = lambda self: None  # type: ignore[assignment]


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def prompt_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    j = text.find(end, i + len(start))
    return text[i + len(start) : j] if i >= 0 and j >= 0 else text


class SyntheticModels:
    """
    Replaces litellm.acompletion. Answers by role, read from the prompt:
    research, summary, forecast or parsing. Forecast answers come from
    `forecast_answer(model, prompt, call_number)`, a function each test can
    replace. Every answer is SYNTHETIC. Each call yields to the event loop
    first, as a real call would, so calls running at the same time interleave.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.forecast_calls: list[dict[str, Any]] = []
        self.forecast_answer: Callable[[str, str, int], str] = self.default_forecast_answer
        self.report_cost = True
        # A test may set the SYNTHETIC cost each model reports, in US dollars
        # as a string, instead of the default of US$0.001 per 1,000 tokens.
        self.cost_by_model: dict[str, str] = {}
        self.fail_models: set[str] = set()

    # Synthetic token counts: 1 token per 4 characters, at least 10.
    @staticmethod
    def tokens(text: str) -> int:
        return max(10, len(text) // 4)

    @staticmethod
    def default_forecast_answer(model: str, prompt: str, n: int) -> str:
        head = "Reference class and base rate: SYNTHETIC.\nCurrent evidence: SYNTHETIC.\n"
        if "Percentile 10" in prompt:
            low, high = SyntheticModels.bounds_in(prompt)
            span = high - low
            shift = (n % 3) * 0.02 * span
            points = [0.15, 0.25, 0.42, 0.58, 0.75, 0.85]
            values = [round(low + p * span + shift, 6) for p in points]
            return head + "\n".join(
                f"Percentile {pct}: {value}" for pct, value in zip([10, 20, 40, 60, 80, 90], values)
            )
        if "Option_A" in prompt:
            options = re.findall(r"'([^']+)'", _between(prompt, "The options are: [", "]"))
            shares = [[50, 30, 20], [60, 30, 10], [40, 40, 20]][n % 3]
            return head + "\n".join(f"{o}: {p}%" for o, p in zip(options, shares))
        return head + f"Probability: {[30, 35, 42][n % 3]}%"

    @staticmethod
    def bounds_in(prompt: str) -> tuple[float, float]:
        lower = re.search(r"likely not lower than ([0-9.]+)|can not be lower than ([0-9.]+)", prompt)
        upper = re.search(r"likely not higher than ([0-9.]+)|can not be higher than ([0-9.]+)", prompt)
        low = float(next(g for g in lower.groups() if g)) if lower else 0.0
        high = float(next(g for g in upper.groups() if g)) if upper else 100.0
        return low, high

    @staticmethod
    def parse_answer(prompt: str) -> str:
        text = _between(prompt, "START TEXT <><><><><><><><><><><>", "<><><><><><><><><><><> END TEXT")
        if "BinaryPrediction" in prompt:
            found = re.findall(r"Probability: (\d+(?:\.\d+)?)%", text)
            if not found:
                return "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"
            return json.dumps({"prediction_in_decimal": float(found[-1]) / 100})
        if "PredictedOptionList" in prompt:
            options = re.findall(r"^([^:\n]+): (\d+(?:\.\d+)?)%$", text, flags=re.M)
            return json.dumps(
                {
                    "predicted_options": [
                        {"option_name": name.strip(), "probability": float(p) / 100}
                        for name, p in options
                        if not name.startswith(("Reference", "Current"))
                    ]
                }
            )
        found = re.findall(r"Percentile (\d+): (-?\d+(?:\.\d+)?)", text)
        return json.dumps([{"percentile": int(p) / 100, "value": float(v)} for p, v in found])

    async def acompletion(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        model = kwargs["model"]
        prompt = prompt_text(messages)
        tag = current_question.get()
        record = {
            "model": model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens"),
            "question_id": tag.question_id if tag else None,
        }
        self.calls.append(record)
        await asyncio.sleep(0)
        if model in self.fail_models:
            raise RuntimeError("SYNTHETIC model failure")
        if "You are an assistant to a superforecaster" in prompt:
            answer = "SYNTHETIC research: nothing decisive has happened yet."
        elif "Please summarize the following research" in prompt:
            answer = "SYNTHETIC research summary."
        elif "You are a data analyst helping to convert text into structured data" in prompt:
            answer = self.parse_answer(prompt)
        elif "You are a professional forecaster" in prompt:
            self.forecast_calls.append(record)
            answer = self.forecast_answer(model, prompt, len(self.forecast_calls) - 1)
        else:
            raise AssertionError(f"Unexpected prompt to {model}: {prompt[:200]}")
        prompt_tokens = self.tokens(prompt)
        completion_tokens = self.tokens(answer)
        response = ModelResponse(
            id="synthetic",
            model=model,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content=answer, role="assistant"),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        if self.report_cost:
            # SYNTHETIC reported cost: US$0.001 per 1,000 tokens, unless the
            # test set one for this model. litellm carries it as a float.
            if model in self.cost_by_model:
                cost = float(self.cost_by_model[model])
            else:
                cost = (prompt_tokens + completion_tokens) / 1_000_000
            response._hidden_params = {
                "additional_headers": {"llm_provider-x-litellm-response-cost": cost}
            }
        else:
            response._hidden_params = {}
        return response


class FakeMetaculus:
    """Replaces requests.get / requests.post for the Metaculus client."""

    def __init__(self, posts_page: dict[str, Any] | None = None) -> None:
        self.posts_page = posts_page or {"next": None, "previous": None, "results": []}
        self.gets: list[tuple[str, Any]] = []
        self.posts: list[tuple[str, Any]] = []

    @staticmethod
    def _response(payload: Any, status: int = 200) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode("utf-8")
        return response

    def get(self, url: str, params: Any = None, **kwargs: Any) -> requests.Response:
        self.gets.append((url, params))
        headers = (kwargs.get("headers") or {})
        assert headers.get("Authorization") == f"Token {FAKE_TOKEN}"
        if url.endswith("/posts/"):
            return self._response(copy.deepcopy(self.posts_page))
        match = re.search(r"/posts/(\d+)/$", url)
        if match:
            for post in self.posts_page["results"]:
                if post["id"] == int(match.group(1)):
                    return self._response(copy.deepcopy(post))
            return self._response({"detail": "not found"}, 404)
        raise AssertionError(f"Unexpected GET {url}")

    def post(self, url: str, json: Any = None, **kwargs: Any) -> requests.Response:
        self.posts.append((url, json))
        return self._response({}, 201)

    def forecasts(self) -> list[dict[str, Any]]:
        return [body[0] for url, body in self.posts if url.endswith("/questions/forecast/")]

    def comments(self) -> list[dict[str, Any]]:
        return [body for url, body in self.posts if url.endswith("/comments/create/")]


class OfflineTestCase(unittest.TestCase):
    """Runs each test in its own empty folder with fakes installed."""

    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        os.environ["METACULUS_TOKEN"] = FAKE_TOKEN
        self.models = SyntheticModels()
        self._real_acompletion = general_llm.acompletion
        general_llm.acompletion = self.models.acompletion  # type: ignore[assignment]
        self.metaculus = FakeMetaculus()
        self._real_get, self._real_post = requests.get, requests.post
        requests.get = self.metaculus.get  # type: ignore[assignment]
        requests.post = self.metaculus.post  # type: ignore[assignment]

    def tearDown(self) -> None:
        general_llm.acompletion = self._real_acompletion  # type: ignore[assignment]
        requests.get, requests.post = self._real_get, self._real_post  # type: ignore[assignment]
        os.environ.pop("METACULUS_TOKEN", None)
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def make_meter(self, cap: str = "5.00", reserve: str = "2.00", **kwargs: Any):
        from spend import SpendMeter

        return SpendMeter(
            cap=Decimal(cap),
            reserve=Decimal(reserve),
            log_path=Path("forecasts/spend.jsonl"),
            read_paths=[Path("forecasts/spend.jsonl")],
            run_mode=kwargs.pop("run_mode", "offline-test"),
            dry_run=kwargs.pop("dry_run", False),
            **kwargs,
        )

    def make_bot(
        self,
        *,
        publish: bool,
        skip: bool = True,
        meter=None,
        run_mode: str = "offline-test",
        ensemble_size: int | None = None,
        high_value_model: bool | None = None,
    ):
        """A bot as main.run builds it. The two settings default to the environment."""
        import main

        return main.GlassEarthBot(
            run_mode=run_mode,
            spend_meter=meter or self.make_meter(run_mode=run_mode, dry_run=not publish),
            ensemble_size=ensemble_size,
            high_value_model=high_value_model,
            research_reports_per_question=1,
            publish_reports_to_metaculus=publish,
            skip_previously_forecasted_questions=skip,
            extra_metadata_in_explanation=True,
        )

    def set_env(self, **values: str) -> None:
        """Set environment variables for this test only."""
        for name, value in values.items():
            previous = os.environ.get(name)
            os.environ[name] = value
            if previous is None:
                self.addCleanup(os.environ.pop, name, None)
            else:
                self.addCleanup(os.environ.__setitem__, name, previous)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
