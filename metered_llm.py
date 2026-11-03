"""
A GeneralLlm whose every call goes through the SpendMeter (see spend.py).

Before each call it works out the most the call could cost and asks the meter
for room under the daily cap; the meter refuses (SpendCapReached) if there is
none. After the call it records the cost against the current question.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from forecasting_tools import GeneralLlm

from spend import (
    SpendMeter,
    estimate_tokens,
    price_for,
    usage_from_response,
    worst_case_call_cost,
)


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    try:
        return json.dumps(prompt, default=str)
    except (TypeError, ValueError):
        return str(prompt)


class MeteredLlm(GeneralLlm):
    def __init__(
        self,
        model: str,
        *,
        purpose: str,
        meter: SpendMeter,
        max_tokens: int,
        max_searches: int = 0,
        search_context_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("Every metered model needs a positive max_tokens")
        price_for(model)  # fail at start-up, not mid-run, for an unpriced model
        super().__init__(model=model, max_tokens=max_tokens, **kwargs)
        self.purpose = purpose
        self.meter = meter
        self.max_output_tokens = max_tokens
        self.max_searches = max_searches
        self.search_context_tokens = search_context_tokens

    def worst_case_for_prompt(self, prompt: Any) -> Decimal:
        prompt_tokens = estimate_tokens(_prompt_text(prompt)) + self.search_context_tokens
        return worst_case_call_cost(
            self.model, prompt_tokens, self.max_output_tokens, self.max_searches
        )

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:
        # Checked here as well as per attempt, so a refusal is immediate rather
        # than waiting through the retry back-off.
        full_prompt = prompt if system_prompt is None else [system_prompt, prompt]
        self.meter.ensure_affordable(
            self.worst_case_for_prompt(full_prompt), self.model, self.purpose
        )
        return await super().invoke(prompt, system_prompt)

    async def _call_litellm(self, prompt: Any) -> Any:
        reservation = self.meter.reserve_call(
            self.worst_case_for_prompt(prompt), self.model, self.purpose
        )
        try:
            response = await super()._call_litellm(prompt)
        except BaseException as error:
            self.meter.settle_failed(reservation, type(error).__name__)
            raise
        self.meter.settle(reservation, usage_from_response(response))
        return response
