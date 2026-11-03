"""
The Glass Earth forecast ledger.

Every forecast that Metaculus accepts from this bot is recorded at the moment
Metaculus accepts it, sealed until its question closes (sealed_ledger.py):
its encrypted record goes in forecasts/sealed_boxes.jsonl, then a sealed
line in forecasts/ledger.jsonl. Once Metaculus reports the question closed
or resolved, a later live run appends the record to the ledger. The workflow
that ran the bot commits both files, and the daily timestamp workflow proves
when each version of the ledger existed. A dry run writes the full line it
would have written, unsealed, to forecasts/dry_run/ledger.jsonl instead,
which git ignores.

The ledger is public, so it never holds error messages, text copied from
Metaculus, or anything read from the environment except the GitHub commit
and run that made each forecast.
"""

from __future__ import annotations

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

from sealed_ledger import BOXES_FILE, LEDGER_FILE, append_json_line, is_reveal_line, read_json_lines

logger = logging.getLogger(__name__)

LEDGER_PATH = Path(LEDGER_FILE)
SEALED_BOXES_PATH = Path(BOXES_FILE)
DRY_RUN_LEDGER_PATH = Path("forecasts/dry_run/ledger.jsonl")
# Version 3 adds scheduled_close_time_utc. From version 3 the public ledger
# holds sealed lines and their reveals, never full lines.
LEDGER_VERSION = 3


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
        append_json_line(self.path, entry)

    def forecast_lines(self) -> list[dict[str, Any]]:
        """
        One line per forecast, in order: sealed lines, and full lines (a dry
        run's, or any written before sealing). A reveal line repeats the
        forecast of an earlier sealed line, so it is left out.
        """
        return [line for line in read_json_lines(self.path) if not is_reveal_line(line)]

    def question_ids(self) -> set[int]:
        return {
            line["question_id"]
            for line in self.forecast_lines()
            if isinstance(line.get("question_id"), int)
        }

    def last_recorded_at(self) -> dict[int, datetime]:
        """The time of the latest forecast for each question in the ledger."""
        latest: dict[int, datetime] = {}
        for line in self.forecast_lines():
            question_id = line.get("question_id")
            try:
                recorded = datetime.strptime(
                    line.get("recorded_at_utc", ""), "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
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
    One full ledger line. `context` describes the question and the model
    forecasts; `final_forecast` is exactly what Metaculus was sent. A dry run
    writes this line as it is; a live run seals it (sealed_ledger.seal_entry),
    and every field here must be listed in sealed_ledger.RECORD_FIELDS or
    sealed_ledger.WITHHELD_FIELDS.
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
        "scheduled_close_time_utc": context.get("scheduled_close_time_utc"),
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
