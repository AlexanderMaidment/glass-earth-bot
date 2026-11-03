"""
The Glass Earth forecast ledger.

Every forecast that Metaculus accepts from this bot is appended to
forecasts/ledger.jsonl as one JSON line, at the moment Metaculus accepts it.
The workflow that ran the bot commits the file, and the daily timestamp
workflow proves when each version of it existed. A dry run writes the forecast
it would have submitted to forecasts/dry_run/ledger.jsonl instead, which git
ignores.

The ledger is public, so it never holds error messages or anything read from
the environment except the GitHub commit and run that made each forecast.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from forecasting_tools import (
    BinaryReport,
    ForecastReport,
    MetaculusClient,
    MultipleChoiceReport,
    NumericDistribution,
    NumericReport,
    PredictedOptionList,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

LEDGER_PATH = Path("forecasts/ledger.jsonl")
DRY_RUN_LEDGER_PATH = Path("forecasts/dry_run/ledger.jsonl")
LEDGER_VERSION = 2


class LedgerClient(MetaculusClient):
    """A MetaculusClient that reports each forecast Metaculus accepts."""

    on_forecast_accepted: Callable[[int, dict[str, Any]], None] | None = None

    def _post_question_prediction(
        self, question_id: int, forecast_payload: dict
    ) -> None:
        super()._post_question_prediction(question_id, forecast_payload)
        # Only reached once Metaculus has accepted the forecast.
        if self.on_forecast_accepted is not None:
            self.on_forecast_accepted(question_id, forecast_payload)


class ForecastLedger:
    """An append-only JSON Lines file of forecasts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=True) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as file:
            file.write(line)

    def question_ids(self) -> set[int]:
        if not self.path.exists():
            return set()
        question_ids: set[int] = set()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                question_id = json.loads(line).get("question_id")
            except json.JSONDecodeError:
                logger.warning(f"Skipping unreadable line {line_number} of {self.path}")
                continue
            if isinstance(question_id, int):
                question_ids.add(question_id)
        return question_ids

    def last_recorded_at(self) -> dict[int, datetime]:
        """The time of the latest line for each question in the ledger."""
        if not self.path.exists():
            return {}
        latest: dict[int, datetime] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                question_id = entry.get("question_id")
                recorded = datetime.strptime(
                    entry.get("recorded_at_utc", ""), "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                continue
            if isinstance(question_id, int) and (
                question_id not in latest or recorded > latest[question_id]
            ):
                latest[question_id] = recorded
        return latest


def forecast_as_json(prediction: Any) -> Any:
    """A forecast in the same shape that Metaculus is sent."""
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return {"probability_yes": float(prediction)}
    if isinstance(prediction, PredictedOptionList):
        return {"probability_yes_per_category": prediction.to_dict()}
    if isinstance(prediction, NumericDistribution):
        return {
            "percentiles": [
                {"percentile": percentile.percentile, "value": percentile.value}
                for percentile in prediction.declared_percentiles
            ]
        }
    if isinstance(prediction, BaseModel):
        return prediction.model_dump(mode="json")
    return str(prediction)


def payload_for(report: ForecastReport) -> dict[str, Any] | None:
    """
    The forecast that publishing this report would send to Metaculus. This
    mirrors publish_report_to_metaculus in forecasting-tools and is only used
    for dry runs. Real submissions record what was actually sent.
    """
    if isinstance(report, BinaryReport):
        return {"probability_yes": report.prediction}
    if isinstance(report, MultipleChoiceReport):
        return {
            "probability_yes_per_category": {
                option.option_name: option.probability
                for option in report.prediction.predicted_options
            }
        }
    if isinstance(report, NumericReport):
        prediction = report.prediction
        if prediction.cdf_size is None:
            prediction = NumericDistribution.from_question(
                prediction.declared_percentiles, report.question
            )
        return {
            "continuous_cdf": [percentile.percentile for percentile in prediction.get_cdf()]
        }
    return None


def ledger_entry(
    *,
    question_id: int,
    final_forecast: dict[str, Any],
    context: dict[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    """
    One ledger line. `context` describes the question and the model forecasts;
    `final_forecast` is exactly what Metaculus was sent.
    """
    server = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    run_url = (
        f"{server}/{repository}/actions/runs/{run_id}"
        if server and repository and run_id
        else None
    )
    return {
        "ledger_version": LEDGER_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question_id": question_id,
        "post_id": context.get("post_id"),
        "url": context.get("url"),
        "type": context.get("type"),
        "group_option": context.get("group_option"),
        "title": context.get("title"),
        "tournaments": context.get("tournaments"),
        "run_mode": run_mode,
        "final_forecast": final_forecast,
        "final_forecast_readable": context.get("final_forecast_readable"),
        "combined_by": context.get("combined_by"),
        "model_forecasts": context.get("model_forecasts"),
        "models_without_forecast": context.get("models_without_forecast"),
        "forecaster_model": context.get("forecaster_model"),
        "route": context.get("route"),
        "forecast_reason": context.get("forecast_reason"),
        "cost_usd": context.get("cost_usd"),
        "cost_usd_cap_basis": context.get("cost_usd_cap_basis"),
        "cost_by_model_usd": context.get("cost_by_model_usd"),
        "model_calls": context.get("model_calls"),
        "prices_retrieved_at_utc": context.get("prices_retrieved_at_utc"),
        "code_commit": os.getenv("GITHUB_SHA"),
        "run_url": run_url,
    }
