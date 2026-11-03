import argparse
import asyncio
import logging
import math
import os
import statistics
import sys
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Literal, Sequence

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    QuestionState,
    ReasonedPrediction,
    SmartSearcher,
    DataOrganizer,
    ForecastReport,
    clean_indents,
    structure_output,
)
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools.data_models.markdown_tree import MarkdownTree

# Imported as a module: run() looks up credit.fetch_key_info when it runs, so
# the offline tests can replace it and never reach the network.
import credit
from guardrails import (
    BINARY_MAX,
    BINARY_MIN,
    NUMERIC_LOWER_TAIL_WEIGHT,
    NUMERIC_UNIFORM_WEIGHT,
    NUMERIC_UPPER_TAIL_WEIGHT,
    apply_guardrails,
    guard_binary,
    guard_option_list,
    multiple_choice_floor,
)
from ledger import (
    DRY_RUN_LEDGER_PATH,
    LEDGER_PATH,
    SEALED_BOXES_PATH,
    ForecastLedger,
    LedgerClient,
    forecast_as_json,
    ledger_entry,
    payload_for,
)
from metered_llm import MeteredLlm
from reforecast import forecast_due, latest, policy_for
from routing import (
    DEFAULT_FORECASTER,
    ENSEMBLE_SIZE_ENV,
    HIGH_VALUE_FORECASTER,
    HIGH_VALUE_MODEL_ENV,
    Route,
    check_ensemble_size,
    choose_route,
    ensemble_size_from_env,
    high_value_model_from_env,
    prize_value_usd,
)
from run_log import (
    DETAILS,
    LOG_FORMAT,
    counts_line,
    done,
    error_kind,
    failed,
    note,
    public_exit,
    question_line,
    quiet_wanted,
    show,
    skipped,
    start_quiet,
)
from sealed_ledger import (
    KEY_ENV,
    LedgerKeyError,
    PostStatuses,
    append_json_line,
    check_ledger_key,
    ledger_key_from_env,
    metaculus_post_statuses,
    reveal_closed_questions,
    seal_entry,
    utc_text,
)
from spend import (
    PRICES_RETRIEVED_AT_UTC,
    QUESTION_RESERVATION_ENV,
    ZERO,
    QuestionTag,
    SpendMeter,
    current_question,
    daily_cap_setting_from_env,
    estimate_tokens,
    money_display,
    money_text,
    question_reservation_from_env,
    spend_log_paths,
    worst_case_call_cost,
)

# The offline tests set GLASS_EARTH_SKIP_DOTENV=1 so that they never read the
# real keys in .env.
if os.getenv("GLASS_EARTH_SKIP_DOTENV") != "1":
    dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Glass Earth forecasters. Each question gets the bot's ensemble size of
# independent forecasts from one model, and the median is submitted; with one
# forecast (the default), that forecast is submitted. routing.py picks the
# model: Claude Opus 5.5, or Claude Fable 5.1 for high-value questions when
# Fable is switched on and the daily cap allows (DEFAULT_FORECASTER and
# HIGH_VALUE_FORECASTER). GLASS_EARTH_ENSEMBLE_SIZE (1 to 5, default 1) and
# GLASS_EARTH_HIGH_VALUE_MODEL (true or false, default false) change both
# without editing code; see routing.py.
# Model IDs and prices are from OpenRouter's public model list
# (https://openrouter.ai/api/v1/models), read on 2026-11-03; see spend.py.
# The most a forecaster may write per call, its thinking included. This also
# bounds the worst-case cost that the daily cap counts.
FORECASTER_MAX_TOKENS = 10000
# ":online" adds OpenRouter's web search, which uses OpenAI's own search for
# OpenAI models. The template's default researcher is no longer on OpenRouter.
RESEARCH_MODEL = "openrouter/openai/gpt-6.1-terra:online"
RESEARCH_MAX_TOKENS = 6000
# Assumptions for the worst case only, since OpenRouter reports neither: at
# most 5 searches per research call, adding at most 20,000 prompt tokens.
RESEARCH_MAX_SEARCHES = 5
RESEARCH_SEARCH_CONTEXT_TOKENS = 20000
# A cheap model that summarises the research and reads the final numbers out
# of each forecast.
HELPER_MODEL = "openrouter/openai/gpt-6-luna"
HELPER_MAX_TOKENS = 4000
# Worst-case planning: the prompt text around the research in a forecast
# prompt, and how many tries structure_output may make per sample.
FORECAST_TEMPLATE_TOKENS = 1500
STRUCTURE_OUTPUT_TRIES = 3

# The ensemble member that the current forecast task belongs to. Each member
# runs in its own asyncio task, so setting this affects only that member.
_ensemble_member: ContextVar[int] = ContextVar("ensemble_member", default=1)
# The route (model) chosen for the question being forecast, and why it is
# being forecast now.
_current_route: ContextVar[Route | None] = ContextVar("current_route", default=None)
_forecast_reason: ContextVar[str | None] = ContextVar("forecast_reason", default=None)
_MODEL_MARKER = "Model: "


def _model_label(model: str) -> str:
    return model.removeprefix("openrouter/")


def _percent(fraction: float) -> str:
    """0.02 as "2%", 0.008333 as "0.83%", for the comment."""
    return f"{round(fraction * 100, 2):g}%"


def _numeric_guardrail_text(open_lower: bool, open_upper: bool) -> str:
    """The numeric guardrail (guardrails.py) in words, for the comment."""
    text = f", with {_percent(NUMERIC_UNIFORM_WEIGHT)} of the probability spread evenly over the range"
    tails = []
    if open_lower:
        tails.append(f"{_percent(NUMERIC_LOWER_TAIL_WEIGHT)} below it")
    if open_upper:
        tails.append(f"{_percent(NUMERIC_UPPER_TAIL_WEIGHT)} above it")
    if tails:
        text += " and at least " + " and ".join(tails)
    return text


def _member_label(model: str, member: int, ensemble_size: int) -> str:
    if ensemble_size == 1:
        return _model_label(model)
    return f"{_model_label(model)} (member {member} of {ensemble_size})"


def setup_text(ensemble_size: int, high_value_model: bool) -> str:
    """The forecasting setup in one line, for the run log."""
    forecasts = "1 forecast" if ensemble_size == 1 else f"{ensemble_size} forecasts (median submitted)"
    fable = "on, while the cap has room" if high_value_model else "off"
    return (
        f"Setup: {forecasts} per question from {_model_label(DEFAULT_FORECASTER)} "
        f"({ENSEMBLE_SIZE_ENV}); {_model_label(HIGH_VALUE_FORECASTER)} for high-value "
        f"questions is {fable} ({HIGH_VALUE_MODEL_ENV})."
    )


def reservation_text(question_reservation_usd: Decimal, ensemble_size: int) -> str:
    """What an Opus question needs left of the cap to start, in one line for the run log."""
    return (
        f"An Opus question needs US${money_text(question_reservation_usd * ensemble_size)} left "
        f"of the cap to start: US${money_text(question_reservation_usd)} per forecast "
        f"({QUESTION_RESERVATION_ENV}), 4 times that for a conditional question."
    )


def _question_for_prompt(question: MetaculusQuestion) -> str:
    """
    The question as the models see it. A group sub-question carries its own
    label (a company, a date range, a measure), which its title may not
    mention, so the label is stated here.
    """
    text = question.question_text
    option = question.group_question_option
    if option:
        text += (
            "\n\nThis is one sub-question of a group of related questions. "
            f'This sub-question is only about: "{option}". '
            f'Forecast "{option}" alone, not the other sub-questions of the group.'
        )
    dates = []
    if question.close_time is not None:
        dates.append(f"It closes for forecasting on {question.close_time.strftime('%Y-%m-%d %H:%M')} UTC")
    if question.scheduled_resolution_time is not None:
        dates.append(
            f"it is scheduled to resolve on {question.scheduled_resolution_time.strftime('%Y-%m-%d %H:%M')} UTC"
        )
    if dates:
        text += "\n\n" + ", and ".join(dates) + "."
    return text


def _question_tag(question: MetaculusQuestion) -> QuestionTag:
    return QuestionTag(
        question_id=question.id_of_question,
        post_id=question.id_of_post,
        url=question.page_url,
    )


def _last_forecast_on_metaculus(question: MetaculusQuestion) -> datetime | None:
    try:
        return question.timestamp_of_my_last_forecast
    except ValueError:
        return None


def _median_of_option_lists(
    predictions: list[PredictedOptionList],
) -> PredictedOptionList:
    """
    Each option's median across the forecasts, rescaled to sum to 1, with
    the multiple choice guardrail (guardrails.py) applied before the option
    list is built.
    """
    names = [option.option_name for option in predictions[0].predicted_options]
    for prediction in predictions:
        if {option.option_name for option in prediction.predicted_options} != set(names):
            raise ValueError("All forecasts must give the same options")
    medians = []
    for name in names:
        values = [
            option.probability
            for prediction in predictions
            for option in prediction.predicted_options
            if option.option_name == name
        ]
        medians.append(statistics.median(values))
    if sum(medians) <= 0:
        raise ValueError("The option medians sum to zero")
    return guard_option_list(names, medians)


def _split_model_line(reasoning: str) -> tuple[str, str]:
    """Separate the "Model: ..." line that each ensemble forecast starts with."""
    first_line, _, rest = reasoning.partition("\n")
    if not first_line.startswith(_MODEL_MARKER):
        return "unknown model", reasoning
    return first_line.removeprefix(_MODEL_MARKER), rest.lstrip("\n")


def _question_ids_submitted_for(question: MetaculusQuestion) -> list[int]:
    """The Metaculus question IDs that forecasting this question posts to."""
    if isinstance(question, ConditionalQuestion):
        parts = [question.parent, question.child, question.question_yes, question.question_no]
    else:
        parts = [question]
    return [part.id_of_question for part in parts if part.id_of_question is not None]


class GlassEarthBot(ForecastBot):
    """
    Glass Earth v0, built on FallTemplateBot2026, the Metaculus template bot for
    the Fall 2026 FutureEval Bot Tournament. Changes from the template:
    - Each question gets ensemble_size forecasts (GLASS_EARTH_ENSEMBLE_SIZE,
      1 by default) from the model routing.py picks for it, and the median
      is submitted.
    - Every model call is metered against a hard daily cap (spend.py); a
      question is skipped when what it needs to start (its reservation, from
      the measured cost of a real question) does not fit in what is left of
      the cap, and each question's cost goes in the ledger.
    - Questions are forecast one at a time, so the cap check before each one
      sees everything spent so far, and a second batch that would run
      alongside one already running against the same cap is refused.
    - Market Pulse questions are forecast again on a schedule
      (reforecast.py); FutureEval and MiniBench questions are forecast once.
    - Each model states a base rate from a reference class first, then adjusts
      it for the current evidence.
    - Guardrails (guardrails.py) on every submitted forecast: yes/no held
      between 2% and 98%, every multiple choice option at least 1%, and
      wider tails for numeric, discrete and date questions.
    - The comment states the submitted forecast and how it was combined, and
      the bot will not publish a comment that leaves it out.
    - Every forecast Metaculus accepts is recorded in forecasts/ledger.jsonl,
      sealed until its question closes (see ledger.py and sealed_ledger.py),
      and questions already in the ledger are skipped.

    Notes from the template:
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(
        self,
        *,
        run_mode: str = "unknown",
        spend_meter: SpendMeter | None = None,
        ensemble_size: int | None = None,
        high_value_model: bool | None = None,
        question_reservation_usd: Decimal | None = None,
        ledger_key: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        # A publishing bot seals each forecast in the public ledger until its
        # question closes, so it needs the ledger key (unless given, from
        # GLASS_EARTH_LEDGER_KEY); without one it stops here. A dry run
        # writes full lines to its own ignored ledger and needs no key.
        self._ledger_key: bytes | None = None
        if kwargs.get("publish_reports_to_metaculus", False):
            self._ledger_key = (
                check_ledger_key(ledger_key) if ledger_key is not None else ledger_key_from_env()
            )
        # Unless given, these come from the environment (routing.py and
        # spend.py), and a value the bot cannot use stops it here, before
        # anything is spent.
        self.ensemble_size = check_ensemble_size(
            ensemble_size if ensemble_size is not None else ensemble_size_from_env()
        )
        if high_value_model is None:
            high_value_model = high_value_model_from_env()
        if not isinstance(high_value_model, bool):
            raise TypeError("high_value_model must be True or False")
        self.high_value_model = high_value_model
        if question_reservation_usd is None:
            question_reservation_usd = question_reservation_from_env()
        if not isinstance(question_reservation_usd, Decimal):
            raise TypeError("question_reservation_usd must be Decimal")
        if not question_reservation_usd.is_finite() or question_reservation_usd <= ZERO:
            raise ValueError("question_reservation_usd must be more than zero")
        self.question_reservation_usd = question_reservation_usd
        # ForecastBot counts its expected predictions from these two, and the
        # cap check assumes one research report per question.
        if kwargs.setdefault("predictions_per_research_report", self.ensemble_size) != self.ensemble_size:
            raise ValueError("predictions_per_research_report must equal the ensemble size")
        if kwargs.setdefault("research_reports_per_question", 1) != 1:
            raise ValueError("Glass Earth runs one research report per question")
        if kwargs.get("metaculus_client") is None:
            kwargs["metaculus_client"] = LedgerClient()
        if kwargs.get("llms") is not None:
            raise ValueError(
                "GlassEarthBot builds its own metered models, so that every call counts against the cap"
            )
        if spend_meter is None:
            log_path, read_paths = spend_log_paths()
            spend_meter = SpendMeter(
                cap_setting=daily_cap_setting_from_env(),
                log_path=log_path,
                read_paths=read_paths,
                run_mode=run_mode,
                dry_run=not kwargs.get("publish_reports_to_metaculus", False),
            )
        self.meter = spend_meter
        kwargs["llms"] = self._metered_llms(spend_meter)
        super().__init__(**kwargs)
        unmetered = [
            purpose for purpose, llm in self._llms.items() if not isinstance(llm, MeteredLlm)
        ]
        if unmetered:
            raise ValueError(f"These models are not metered: {unmetered}")
        self.run_mode = run_mode
        self.ledger = ForecastLedger(LEDGER_PATH)
        self.sealed_boxes_path = SEALED_BOXES_PATH
        self.dry_run_ledger = ForecastLedger(DRY_RUN_LEDGER_PATH)
        self._ledger_contexts: dict[int, dict[str, Any]] = {}
        self.skipped_not_due: list[str] = []
        if isinstance(self.metaculus_client, LedgerClient):
            self.metaculus_client.on_forecast_accepted = self._record_submitted_forecast
        elif self.publish_reports_to_metaculus:
            raise ValueError(
                "Publishing needs a LedgerClient, so that every submitted forecast is recorded"
            )

    ##################################### ENSEMBLE #####################################

    _PURPOSES = (
        "default",
        "forecaster_default",
        "forecaster_high_value",
        "summarizer",
        "researcher",
        "parser",
    )

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        # Only names the purposes for ForecastBot's checks. The models used are
        # always the metered ones from _metered_llms.
        return {purpose: None for purpose in cls._PURPOSES}

    @staticmethod
    def _metered_llms(meter: SpendMeter) -> dict[str, str | GeneralLlm | None]:
        def forecaster(model: str, purpose: str) -> MeteredLlm:
            return MeteredLlm(
                model,
                purpose=purpose,
                meter=meter,
                max_tokens=FORECASTER_MAX_TOKENS,
                timeout=300,
                allowed_tries=2,
            )

        default = forecaster(DEFAULT_FORECASTER, "forecaster")
        return {
            "default": default,
            "forecaster_default": default,
            "forecaster_high_value": forecaster(HIGH_VALUE_FORECASTER, "forecaster"),
            "summarizer": MeteredLlm(
                HELPER_MODEL,
                purpose="summarizer",
                meter=meter,
                max_tokens=HELPER_MAX_TOKENS,
                timeout=120,
                allowed_tries=2,
            ),
            "researcher": MeteredLlm(
                RESEARCH_MODEL,
                purpose="researcher",
                meter=meter,
                max_tokens=RESEARCH_MAX_TOKENS,
                max_searches=RESEARCH_MAX_SEARCHES,
                search_context_tokens=RESEARCH_SEARCH_CONTEXT_TOKENS,
                timeout=300,
                allowed_tries=2,
            ),
            "parser": MeteredLlm(
                HELPER_MODEL,
                purpose="parser",
                meter=meter,
                max_tokens=HELPER_MAX_TOKENS,
                timeout=120,
                allowed_tries=2,
            ),
        }

    def _forecaster_llm(self, route: Route | None = None) -> GeneralLlm:
        route = route if route is not None else _current_route.get()
        if route is not None and route.tier == "high_value":
            return self.get_llm("forecaster_high_value", "llm")
        return self.get_llm("forecaster_default", "llm")

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions[PredictionTypes]:
        """
        Same as ForecastBot's version, except that each of the ensemble_size
        predictions is labelled with its model (and member number, when there
        is more than one). _aggregate_predictions then takes the median: of
        the probabilities for binary questions, of each option's probability
        for multiple choice, and of the cumulative distributions, point by
        point, for numeric, discrete and date questions. The median of one
        forecast is that forecast.
        """
        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            summary_report if self.use_research_summary_to_forecast else research
        )

        tasks = [
            self._make_prediction_with(member, question, research_to_use)
            for member in range(1, self.ensemble_size + 1)
        ]
        valid_predictions, errors, exception_group = (
            await self._gather_results_and_exceptions(tasks)
        )
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if len(valid_predictions) == 0:
            assert exception_group, "Exception group should not be None"
            self._reraise_exception_with_prepended_message(
                exception_group,
                "Error while running research and predictions",
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    async def _make_prediction_with(
        self, member: int, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[PredictionTypes]:
        _ensemble_member.set(member)
        label = _member_label(self._ensemble_llm().model, member, self.ensemble_size)
        try:
            prediction = await self._make_prediction(question, research)
        except Exception as e:
            raise RuntimeError(f"{label} failed: {e.__class__.__name__}: {e}") from e
        return prediction.model_copy(
            update={"reasoning": f"{_MODEL_MARKER}{label}\n\n{prediction.reasoning}"}
        )

    def _ensemble_llm(self) -> GeneralLlm:
        return self._forecaster_llm()

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        # Same as the template's, but names the model behind each forecast.
        forecast_lines = ""
        for forecast in predicted_research.predictions:
            model, _ = _split_model_line(forecast.reasoning)
            readable_prediction = report_type.make_readable_prediction(
                forecast.prediction_value
            )
            forecast_lines += f"*{model}*: {readable_prediction}\n"

        return clean_indents(
            f"""
            ## Report {report_number} Summary
            ### Forecasts
            {forecast_lines}

            ### Research Summary
            {predicted_research.summary_report}
            """
        )

    def _format_forecaster_rationales(
        self, report_number: int, researched_predictions: ResearchWithPredictions
    ) -> str:
        # Same as the template's, but titles each rationale with its model.
        rationales = []
        for forecast in researched_predictions.predictions:
            model, reasoning = _split_model_line(forecast.reasoning)
            sections = MarkdownTree.turn_markdown_into_report_sections(reasoning)
            try:
                modified_content = MarkdownTree.report_sections_to_markdown(sections, 3)
            except Exception as e:
                logger.error(f"Error formatting research report: {e}")
                modified_content = MarkdownTree.report_sections_to_markdown(
                    sections, None
                ).replace("#", "[Hashtag]")
            new_rationale = clean_indents(
                f"""
                ## R{report_number}: {model} reasoning
                {modified_content}
                """
            )
            rationales.append(new_rationale)
        return "\n".join(rationales)

    ##################################### COMMENT #####################################

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if isinstance(question, MultipleChoiceQuestion):
            # The template averages; Glass Earth takes the median, like the
            # other question types (already floored, see the function).
            aggregated = _median_of_option_lists(predictions)  # type: ignore[arg-type]
        else:
            # Binary: the median (forecasting-tools). Numeric, discrete and
            # date: the median of the cumulative distributions, point by point.
            aggregated = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, BinaryQuestion):
            # The comment shows the final probability to two decimal places of a
            # percent. Rounding to match means the comment states exactly the
            # value that is submitted.
            aggregated = round(aggregated, 4)
        # The guardrails (guardrails.py) on the combined forecast of every
        # type, the parts of a conditional one included. What this returns is
        # what the comment states, the ledger records and Metaculus is sent.
        return apply_guardrails(aggregated)

    def _combination_rule(self, question: MetaculusQuestion) -> str:
        model = _model_label(self._forecaster_llm().model)
        binary_guardrail = f"held between {_percent(BINARY_MIN)} and {_percent(BINARY_MAX)}"
        option_guardrail = ""
        if isinstance(question, MultipleChoiceQuestion):
            floor = multiple_choice_floor(len(question.options))
            option_guardrail = f", with every option at least {_percent(floor)}"
        numeric_guardrail = ""
        if isinstance(question, (NumericQuestion, DateQuestion)):
            numeric_guardrail = _numeric_guardrail_text(
                question.open_lower_bound, question.open_upper_bound
            )
        if self.ensemble_size == 1:
            # One forecast: say so, rather than "the median of 1".
            source = f"1 forecast from {model}"
            if isinstance(question, BinaryQuestion):
                return f"{source}, {binary_guardrail}."
            if isinstance(question, MultipleChoiceQuestion):
                return f"{source}, its option probabilities rescaled to sum to 100%{option_guardrail}."
            if isinstance(question, (NumericQuestion, DateQuestion)):
                return f"{source}, submitted as its cumulative distribution{numeric_guardrail}."
            return (
                f"{source}, submitted through the forecasting-tools method for this question type, "
                f"each yes/no part {binary_guardrail}."
            )
        source = f"{self.ensemble_size} independent forecasts from {model}"
        if isinstance(question, BinaryQuestion):
            return f"the median of {source}, each {binary_guardrail}."
        if isinstance(question, MultipleChoiceQuestion):
            return (
                f"the median of each option's probability across {source}, "
                f"rescaled to sum to 100%{option_guardrail}."
            )
        if isinstance(question, (NumericQuestion, DateQuestion)):
            return (
                f"the median of {source}, taken point by point along their "
                f"cumulative distributions{numeric_guardrail}."
            )
        return (
            f"the forecasting-tools method for this question type, applied to {source}, "
            f"each yes/no part {binary_guardrail}."
        )

    @staticmethod
    def _outcome_line(question: MetaculusQuestion, outcome: str, cost: str | None = None) -> str:
        return question_line(question.id_of_question, question.page_url, question.id_of_post, outcome, cost)

    def _cost_text(self, question: MetaculusQuestion) -> str:
        cost = self.meter.question_cost(question.id_of_question)
        if cost.unknown_calls:
            return (
                f"Unknown ({cost.unknown_calls} of {cost.calls} model calls had no reported cost; "
                f"at least US${money_text(cost.known)})"
            )
        return f"US${money_text(cost.known)} over {cost.calls} model calls"

    def _create_comment(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: PredictionTypes,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        """
        The template's comment layout, plus a line saying how the model forecasts
        were combined. The comment is built from the same aggregated prediction
        that is submitted, and this refuses to return a comment that does not
        state it, so the published comment always matches the forecast.
        """
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        final_prediction = report_type.make_readable_prediction(aggregated_prediction)

        all_summaries = []
        all_core_research = []
        all_forecaster_rationales = []
        for i, collection in enumerate(research_prediction_collections):
            all_summaries.append(
                self._format_and_expand_research_summary(i + 1, report_type, collection)
            )
            all_core_research.append(self._format_main_research(i + 1, collection))
            all_forecaster_rationales.append(
                self._format_forecaster_rationales(i + 1, collection)
            )
        combined_summaries = "\n".join(all_summaries)
        combined_research_reports = "\n".join(all_core_research)
        combined_rationales = "\n".join(all_forecaster_rationales)

        disabled = "extra_metadata_in_explanation is disabled"
        show_metadata = self.extra_metadata_in_explanation
        # The bot's own meter, not litellm's estimate: OpenRouter's reported
        # cost where it gives one (see spend.py).
        cost = self._cost_text(question) if show_metadata else disabled
        time_spent = (
            f"{round(time_spent_in_minutes, 2)} minutes" if show_metadata else disabled
        )
        llms = self.make_llm_dict() if show_metadata else disabled
        bot_name = self.__class__.__name__ if show_metadata else disabled
        question_line = question.question_text
        if question.group_question_option:
            question_line += f" (sub-question: {question.group_question_option})"
        comment = clean_indents(
            f"""
            # SUMMARY
            *Question*: {question_line}
            *Final Prediction*: {final_prediction}
            *How the final prediction was made*: {self._combination_rule(question)}
            *Total Cost*: {cost}
            *Time Spent*: {time_spent}
            *LLMs*: `{llms}`
            *Bot Name*: {bot_name}

            {combined_summaries}

            # RESEARCH
            {combined_research_reports}

            # FORECASTS
            {combined_rationales}
            """
        )
        max_comment_size = 150000
        if len(comment) > max_comment_size:
            comment = (
                comment[: max_comment_size - 200]
                + "\n\n---\n\nThe comment was too long and has been cut short."
            )
        # clean_indents may change leading spaces, so compare without them.
        stated = " ".join(f"*Final Prediction*: {final_prediction}".split())
        if stated not in " ".join(comment.split()):
            raise ValueError(
                "The comment does not state the final prediction, so nothing will be published"
            )
        return comment

    ##################################### LEDGER #####################################

    def _select_questions(
        self, questions: Sequence[MetaculusQuestion]
    ) -> tuple[list[tuple[MetaculusQuestion, str]], list[tuple[MetaculusQuestion, str]]]:
        """
        The questions to forecast now, each with the reason, and the others,
        each with why it was skipped (one of run_log.SKIP_REASONS).
        - Only open questions: a group can hold closed or upcoming
          sub-questions next to open ones.
        - Market Pulse (a re-forecast policy): forecast when due.
        - Otherwise, with skip_previously_forecasted_questions: once only,
          skipping anything Metaculus says was forecast and anything already
          in the ledger (a second guard next to the template's check).
        """
        now = datetime.now(timezone.utc)
        ledger_times = self.ledger.last_recorded_at()
        # Any line at all counts for the one-forecast rule, even an unreadable time.
        in_ledger = self.ledger.question_ids()
        selected: list[tuple[MetaculusQuestion, str]] = []
        skips: list[tuple[MetaculusQuestion, str]] = []
        already = 0
        for question in questions:
            if question.state is not None and question.state != QuestionState.OPEN:
                logger.info(
                    f"Not forecasting {question.page_url} ({question.group_question_option}): it is {question.state.value}"
                )
                skips.append((question, "not open"))
                continue
            policy = policy_for(question.tournament_slugs)
            if policy is not None:
                last = latest(
                    _last_forecast_on_metaculus(question),
                    ledger_times.get(question.id_of_question or -1),
                )
                due, reason = forecast_due(
                    now=now,
                    close_time=question.close_time,
                    last_forecast_at=last,
                    policy=policy,
                )
                if not due:
                    self.skipped_not_due.append(f"{question.page_url}: {reason}")
                    logger.info(f"Not forecasting {question.page_url} yet, {reason}")
                    skips.append((question, "not due"))
                    continue
                selected.append((question, reason))
            elif self.skip_previously_forecasted_questions:
                if question.already_forecasted or question.id_of_question in in_ledger:
                    already += 1
                    skips.append((question, "already forecast"))
                    continue
                selected.append((question, "first forecast"))
            else:
                selected.append((question, "forecast on request (re-forecasting allowed here)"))
        if already:
            logger.info(
                f"Skipping {already} questions already forecast (on Metaculus or in the ledger)"
            )
        return selected, skips

    @staticmethod
    def _forecasts_per_member(question: MetaculusQuestion) -> int:
        # A conditional question forecasts up to 4 linked questions per member.
        return 4 if isinstance(question, ConditionalQuestion) else 1

    def _question_reservation(self, question: MetaculusQuestion) -> Decimal:
        """
        What an Opus question needs left of the cap to start: the reservation
        per forecast (twice the measured cost of one real question; see
        QUESTION_RESERVATION_USD in spend.py) times its forecasts, counted
        like the worst case below: one per member of the ensemble, and 4 per
        member for a conditional question.
        """
        return (
            self.question_reservation_usd
            * self.ensemble_size
            * self._forecasts_per_member(question)
        )

    def _worst_case_question_cost(
        self, question: MetaculusQuestion, forecaster_model: str
    ) -> Decimal:
        """
        The most forecasting this question could cost: every call at its
        output limit, every retry used, and one forecast (with its parsing)
        per member of the ensemble. Calculated from MODEL_PRICES. The cap
        check before a Fable question counts it. For an Opus question the
        check counts the reservation (_question_reservation) instead; this
        then sets how far past the cap the question may go once it has
        started (worst case minus reservation), and is recorded for
        information when the question is skipped.
        """
        question_tokens = estimate_tokens(
            self._get_research_prompt(question, self.get_llm("researcher"))
            + (question.background_info or "")
        )
        # Text written by one model is read by the next. estimate_tokens
        # counts 3 characters a token where models write about 4, so scale up.
        research_out = math.ceil(RESEARCH_MAX_TOKENS * 4 / 3)
        member_out = math.ceil(FORECASTER_MAX_TOKENS * 4 / 3)
        research = worst_case_call_cost(
            RESEARCH_MODEL,
            question_tokens + RESEARCH_SEARCH_CONTEXT_TOKENS,
            RESEARCH_MAX_TOKENS,
            RESEARCH_MAX_SEARCHES,
        ) * self.get_llm("researcher", "llm").allowed_tries
        summary = worst_case_call_cost(
            HELPER_MODEL, question_tokens + research_out, HELPER_MAX_TOKENS
        )
        forecasts_per_member = self._forecasts_per_member(question)
        members = (
            self.ensemble_size
            * forecasts_per_member
            * worst_case_call_cost(
                forecaster_model,
                question_tokens + research_out + FORECAST_TEMPLATE_TOKENS,
                FORECASTER_MAX_TOKENS,
            )
        )
        parser_calls = (
            self.ensemble_size
            * forecasts_per_member
            * self._structure_output_validation_samples
            * STRUCTURE_OUTPUT_TRIES
        )
        parser = parser_calls * worst_case_call_cost(
            HELPER_MODEL, member_out + FORECAST_TEMPLATE_TOKENS, HELPER_MAX_TOKENS
        )
        return research + summary + members + parser

    def _route_for(self, question: MetaculusQuestion) -> Route:
        self.meter.refresh_day()
        return choose_route(
            value_usd=prize_value_usd(question.tournament_slugs, question.question_weight),
            default_question_usd=self._question_reservation(question),
            high_value_question_usd=(
                self._worst_case_question_cost(question, HIGH_VALUE_FORECASTER)
                if self.high_value_model
                else None
            ),
            spent_today_usd=self.meter.spent_today + self.meter.in_flight,
            cap_usd=self.meter.cap,
            reserve_usd=self.meter.reserve,
            use_high_value_model=self.high_value_model,
        )

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        """
        Like ForecastBot's version, but one question at a time, each after a
        cap check, so that no question starts unless what it needs to start
        (its route's reservation) fits in what is left of today's cap. A
        question that does not fit is skipped and logged (in the run log and
        forecasts/spend.jsonl), not failed. A question that has started may
        finish past the cap by up to its calculated worst case minus its
        reservation (SpendMeter.question_running), so that its forecast is
        not refused after its research is paid for.

        ForecastBot's own version starts every question at once (its
        _max_concurrent_questions limits only research calls). Here the
        ensemble members of one question run at the same time, which the cap
        check counts, but questions never do: a second batch started while
        this one runs against the same meter is refused before it spends.
        """
        if self.publish_reports_to_metaculus and self._ledger_key is None:
            # Never a full line in the public ledger: stop before any spending.
            raise LedgerKeyError(f"Publishing needs the ledger key ({KEY_ENV}) to seal the ledger.")
        reports: list[ForecastReport | BaseException] = []
        publishing = self.publish_reports_to_metaculus
        with self.meter.one_batch_at_a_time():
            selected, skips = self._select_questions(questions)
            # Each question's outcome as a content-free line (run_log.note):
            # in the console of a live run on GitHub Actions, in the log of
            # any other run.
            skip_counts = Counter(reason for _, reason in skips)
            for question, reason in skips:
                note(self._outcome_line(question, skipped(reason)))
            posted = failures = 0
            for question, reason in selected:
                route = self._route_for(question)
                tag = _question_tag(question)
                # The calculated worst case with the route's model (Opus for
                # a skip): recorded for a skip, and it sets how far past the
                # cap a question that starts may go.
                worst_case = self._worst_case_question_cost(
                    question, route.model or DEFAULT_FORECASTER
                )
                if route.action == "skip":
                    logger.warning(
                        f"Spend cap: skipping {question.page_url}"
                        f"{' (' + question.group_question_option + ')' if question.group_question_option else ''}: {route.reason}"
                    )
                    self.meter.record_skip(tag, route.reason, route.reserved_usd, worst_case)
                    skip_counts["daily cap"] += 1
                    note(self._outcome_line(question, skipped("daily cap")))
                    continue
                logger.info(
                    f"Forecasting {question.page_url} with {_model_label(route.model or '')}: "
                    f"{route.reason}; {reason}. It needed US${money_display(route.reserved_usd)} "
                    f"left of the cap to start (calculated worst case US${money_display(worst_case)})."
                )
                tokens = (
                    _current_route.set(route),
                    _forecast_reason.set(reason),
                    current_question.set(tag),
                )
                try:
                    with self.meter.question_running(
                        reservation=route.reserved_usd, worst_case=worst_case
                    ):
                        reports.append(
                            await self._run_individual_question_with_error_propagation(question)
                        )
                    posted += 1
                    note(self._outcome_line(question, done(publishing), self._cost_text(question)))
                except Exception as e:
                    failures += 1
                    # The exception's type only: its message can hold model output.
                    note(self._outcome_line(question, failed(e), self._cost_text(question)))
                    if not return_exceptions:
                        raise
                    reports.append(e)
                finally:
                    _current_route.reset(tokens[0])
                    _forecast_reason.reset(tokens[1])
                    current_question.reset(tokens[2])
            note(
                counts_line(
                    found=len(questions),
                    selected=len(selected),
                    skips=skip_counts,
                    posted=posted,
                    failures=failures,
                    publishing=publishing,
                )
            )
        if self.folder_to_save_reports_to:
            finished = [r for r in reports if not isinstance(r, BaseException)]
            ForecastReport.save_object_list_to_file_path(
                finished, self._create_file_path_to_save_to(list(questions))
            )
        return reports

    async def _run_individual_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        try:
            report = await super()._run_individual_question(question)
            if not self.publish_reports_to_metaculus:
                self._record_dry_run_forecast(report)
            return report
        finally:
            for question_id in _question_ids_submitted_for(question):
                self._ledger_contexts.pop(question_id, None)

    def _create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: PredictionTypes,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        explanation = super()._create_unified_explanation(
            question,
            research_prediction_collections,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        # Publishing happens next, so keep what the ledger needs to know.
        context = self._ledger_context(
            question, research_prediction_collections, aggregated_prediction
        )
        for question_id in _question_ids_submitted_for(question):
            self._ledger_contexts[question_id] = context
        return explanation

    def _ledger_context(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: PredictionTypes,
    ) -> dict[str, Any]:
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        model_forecasts = []
        for collection in research_prediction_collections:
            for prediction in collection.predictions:
                model, _ = _split_model_line(prediction.reasoning)
                model_forecasts.append(
                    {
                        "model": model,
                        "forecast": forecast_as_json(prediction.prediction_value),
                        "forecast_readable": report_type.make_readable_prediction(
                            prediction.prediction_value
                        ),
                    }
                )
        models_with_forecast = {entry["model"] for entry in model_forecasts}
        forecaster = self._forecaster_llm().model
        expected_members = [
            _member_label(forecaster, member, self.ensemble_size)
            for member in range(1, self.ensemble_size + 1)
        ]
        route = _current_route.get()
        cost = self.meter.question_cost(question.id_of_question)
        return {
            "post_id": question.id_of_post,
            "url": question.page_url,
            "type": getattr(question, "question_type", type(question).__name__),
            "group_option": question.group_question_option,
            "title": question.question_text,
            # Only to give each option by its index in a sealed record.
            "option_labels": (
                list(question.options) if isinstance(question, MultipleChoiceQuestion) else None
            ),
            "tournaments": question.tournament_slugs,
            # As scheduled when forecast; Metaculus can move a close.
            "scheduled_close_time_utc": (
                utc_text(question.close_time) if question.close_time is not None else None
            ),
            "final_forecast_readable": report_type.make_readable_prediction(
                aggregated_prediction
            ),
            "combined_by": self._combination_rule(question),
            "model_forecasts": model_forecasts,
            "models_without_forecast": [
                label for label in expected_members if label not in models_with_forecast
            ],
            "forecaster_model": _model_label(forecaster),
            "route": route.reason if route is not None else None,
            "forecast_reason": _forecast_reason.get(),
            "cost_usd": cost.as_text(),
            "cost_usd_cap_basis": money_text(cost.cap_basis),
            "cost_by_model_usd": {
                model: money_text(amount) for model, amount in sorted(cost.by_model.items())
            },
            "model_calls": cost.calls,
            "prices_retrieved_at_utc": PRICES_RETRIEVED_AT_UTC,
        }

    def _record_submitted_forecast(
        self, question_id: int, final_forecast: dict[str, Any]
    ) -> None:
        """
        Seal the forecast Metaculus has just accepted (sealed_ledger.py): its
        box goes in forecasts/sealed_boxes.jsonl first, so that a sealed line
        never exists without its box, then the sealed line in the ledger.
        """
        if self._ledger_key is None:
            # Never a full line in the public ledger.
            raise LedgerKeyError(f"Publishing needs the ledger key ({KEY_ENV}) to seal the ledger.")
        context = self._ledger_contexts.get(question_id, {})
        sealed = seal_entry(
            ledger_entry(
                question_id=question_id,
                final_forecast=final_forecast,
                context=context,
                run_mode=self.run_mode,
            ),
            self._ledger_key,
        )
        append_json_line(self.sealed_boxes_path, sealed.box_line)
        self.ledger.append(sealed.line)
        logger.info(
            f"Ledger: sealed the forecast submitted on question {question_id} "
            f"(seal {sealed.line['seal_sha256'][:12]})"
        )

    def _record_dry_run_forecast(self, report: ForecastReport) -> None:
        question_id = report.question.id_of_question
        final_forecast = payload_for(report)
        if question_id is None or final_forecast is None:
            return
        entry = ledger_entry(
            question_id=question_id,
            final_forecast=final_forecast,
            context=self._ledger_contexts.get(question_id, {}),
            run_mode=self.run_mode,
        )
        entry["dry_run"] = True
        self.dry_run_ledger.append(entry)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {_question_for_prompt(question)}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your reasoning will be published next to your forecast.

            Your question is:
            {_question_for_prompt(question)}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write your reasoning in these steps, in this order:
            1. Reference class and base rate. Choose a reference class of comparable past cases. Say what the class is, roughly how many cases it covers, and how often such cases ended in Yes. Write this as "Base rate: XX%" before you consider anything specific to this question. If no close class exists, use the nearest one and say how well it fits.
            2. Current evidence. Go through the specific facts in the background and research. For each one, say whether it moves the probability up or down from the base rate, and roughly how far.
            3. Time and status quo. Say how long until the outcome is known, and what the outcome would be if nothing changed.
            4. Scenarios. Briefly describe one way this ends in No and one way it ends in Yes.
            5. Adjusted forecast. Start from the base rate, apply your adjustments, and say how far you moved and why.

            Good forecasters anchor on the base rate and move away from it only as far as the evidence justifies. They also put extra weight on the status quo, since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._ensemble_llm().invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            - The text states a base rate before its final answer. Return the final answer, not the base rate.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        # Each forecast gets the yes/no guardrail too (the template holds it
        # between 1% and 99%), so the comment shows every forecast as used.
        decimal_pred = guard_binary(binary_prediction.prediction_in_decimal)

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your reasoning will be published next to your forecast.

            Your question is:
            {_question_for_prompt(question)}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write your reasoning in these steps, in this order:
            1. Reference class and base rate. Choose a reference class of comparable past cases. Say what the class is, roughly how many cases it covers, and how often outcomes like each option happened. Give these base rates in a sentence, before you consider anything specific to this question.
            2. Current evidence. Go through the specific facts in the background and research. For each one, say which options it moves up or down from their base rates, and roughly how far.
            3. Time and status quo. Say how long until the outcome is known, and what the outcome would be if nothing changed.
            4. Surprise. Briefly describe a scenario that leads to an unexpected outcome.
            5. Adjusted forecast. Start from the base rates, apply your adjustments, and say how far you moved and why.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Good forecasters anchor on the base rates and move away from them only as far as the evidence justifies. They put extra weight on the status quo, since the world changes slowly most of the time, and they leave some moderate probability on most options to allow for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            The text states base rates before its final answer. Return the final probabilities, not the base rates.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._ensemble_llm().invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your reasoning will be published next to your forecast.

            Your question is:
            {_question_for_prompt(question)}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Write your reasoning in these steps, in this order:
            1. Reference class and base rate. Choose a reference class, such as past values of this quantity or of similar quantities. Say what the class is and what values were typical in it: a middle value and a range, in words, before you consider anything specific to now. Do not use the "Percentile" format here; keep it for your final answer.
            2. Current evidence. Cover the outcome if nothing changed, the outcome if the current trend continued, and what experts and markets expect. For each, say whether it moves you up or down from the base rate, and roughly how far. If the quantity is a figure a company will report for a period (such as revenue or sales), anchor first on the company's own guidance for that period, then on how that company has usually landed against its guidance, and say both before going further.
            3. Time left. Say how long until the outcome is known.
            4. Tails. Briefly describe an unexpected scenario that gives a low outcome and one that gives a high outcome.
            5. Adjusted forecast. Start from the base rate, apply your adjustments, and say how far you moved and why.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters anchor on the base rate, move only as far as the evidence justifies, and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._ensemble_llm().invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your reasoning will be published next to your forecast.

            Your question is:
            {_question_for_prompt(question)}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Write your reasoning in these steps, in this order:
            1. Reference class and base rate. Choose a reference class, such as how long comparable events took in the past. Say what the class is and when such events typically happened: a middle date and a range, in words, before you consider anything specific to now. Do not use the "Percentile" format here; keep it for your final answer.
            2. Current evidence. Cover the outcome if nothing changed, the outcome if the current trend continued, and what experts and markets expect. For each, say whether it moves you earlier or later than the base rate, and roughly how far.
            3. Time left. Say how long until the outcome is known.
            4. Tails. Briefly describe an unexpected scenario that gives an early outcome and one that gives a late outcome.
            5. Adjusted forecast. Start from the base rate, apply your adjustments, and say how far you moved and why.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters anchor on the base rate, move only as far as the evidence justifies, and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._ensemble_llm().invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text.\n"
            "- The text describes a base rate before its final answer. Return only the final percentile distribution, never the base rate."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


MARKET_PULSE_ENV = "GLASS_EARTH_MARKET_PULSE"

# Per-mode tournament URL shown in the summary banner footer. These piggyback
# on the forecasting_tools SDK constants and need updating whenever those
# rotate seasons.
TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "market_pulse": "https://www.metaculus.com/tournament/market-pulse-26q4/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Glass Earth forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "market_pulse", "metaculus_cup", "test_questions"],
        default="tournament",
        help=(
            "What to forecast on (default: tournament, which is the Fall 2026 "
            f"FutureEval tournament and MiniBench, plus Market Pulse when {MARKET_PULSE_ENV}=true)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forecast without posting anything to Metaculus (publish_reports_to_metaculus=False)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Forecast at most this many questions from each target (for a short paid test)",
    )
    parser.add_argument(
        "--question-url",
        type=str,
        default=None,
        help="Forecast only this question (a dry run only). A group forecasts its open sub-questions.",
    )
    parser.add_argument(
        "--daily-cap-usd",
        type=str,
        default=None,
        help=(
            "The daily spend cap in US dollars for this run, with no end date (default: "
            "GLASS_EARTH_DAILY_CAP_USD until GLASS_EARTH_DAILY_CAP_UNTIL_UTC, else 2.00)"
        ),
    )
    parser.add_argument(
        "--high-value-reserve-usd",
        type=str,
        default=None,
        help=(
            "The part of the cap low-value questions may not use (default: "
            "GLASS_EARTH_HIGH_VALUE_RESERVE_USD, else US$2.00 or 40%% of the cap, whichever is less)"
        ),
    )
    return parser


def _reveal_closed_questions(
    key: bytes,
    post_statuses: PostStatuses,
    *,
    github_actions: bool,
    step_summary: str | None,
) -> None:
    """
    Reveal the sealed forecasts whose questions Metaculus now reports closed
    or resolved (sealed_ledger.reveal_closed_questions). Whatever happens
    here, forecasting goes ahead. A sealed line that cannot be revealed is
    said in the log, as a GitHub Actions warning and on the run's summary page.
    """
    try:
        report = reveal_closed_questions(
            key, post_statuses, ledger_path=LEDGER_PATH, boxes_path=SEALED_BOXES_PATH
        )
    except Exception as error:  # revealing must never stop forecasting
        logger.exception("Ledger reveals failed")
        show(
            f"Ledger reveals failed ({error_kind(error)}); nothing more is revealed "
            "this run, and forecasting carries on."
        )
        return
    show(report.line())
    for problem in report.problems():
        logger.warning(f"Ledger reveals: {problem}")
        if github_actions:
            show(f"::warning title=Sealed ledger::{credit._workflow_command_text(problem)}")
        credit._add_to_summary(step_summary, f"**Sealed ledger.** {problem}")


async def _forecast_target(
    bot: GlassEarthBot, target: int | str, max_questions: int | None
) -> list[ForecastReport | BaseException]:
    note(f"Forecasting on {target}.")
    if max_questions is None:
        return await bot.forecast_on_tournament(target, return_exceptions=True)
    questions = bot.metaculus_client.get_all_open_questions_from_tournament(target)
    return await bot.forecast_questions(questions[:max_questions], return_exceptions=True)


def run(
    argv: Sequence[str] | None = None,
    *,
    fetch_key_info: Callable[[str], dict[str, Any]] | None = None,
    post_statuses: PostStatuses | None = None,
) -> list[ForecastReport | BaseException]:
    """
    One run of the bot. fetch_key_info reads OpenRouter's report on the key
    (default credit.fetch_key_info, a free call), and post_statuses asks
    Metaculus for the status of the questions in a post before a live run
    reveals any (default sealed_ledger.metaculus_post_statuses, a read-only
    GET); the tests pass fakes.
    """
    args = build_arg_parser().parse_args(argv)
    run_mode: Literal["tournament", "market_pulse", "metaculus_cup", "test_questions"] = args.mode
    if args.question_url and not args.dry_run:
        raise SystemExit("--question-url is for dry runs only; add --dry-run")
    if args.max_questions is not None and args.max_questions < 1:
        raise SystemExit("--max-questions must be at least 1")
    publish_to_metaculus = not args.dry_run
    # Settings from the repository variables. A value the bot cannot use stops
    # the run here, before anything is fetched or spent; it never falls back.
    try:
        ensemble_size = ensemble_size_from_env()
        high_value_model = high_value_model_from_env()
        cap_setting = daily_cap_setting_from_env(args.daily_cap_usd, args.high_value_reserve_usd)
        question_reservation = question_reservation_from_env()
        # A live run seals each forecast in the public ledger with this key
        # (sealed_ledger.py), so without a usable one it stops here. A dry
        # run needs none.
        ledger_key = ledger_key_from_env() if publish_to_metaculus else None
    except ValueError as error:
        raise SystemExit(f"Stopped before forecasting: {error}") from error

    # The lines through show() carry no question content, so a live run on
    # GitHub Actions shows them (run_log.py); everything else it writes goes
    # only to its log file on the runner.
    check_environment(strict=True, out=show)
    github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    print_startup_banner(run_mode, will_publish=publish_to_metaculus, out=show)
    show(setup_text(ensemble_size, high_value_model))

    # What is left on the OpenRouter key, from OpenRouter's free key check. A
    # check that fails gives Unknown and the run carries on; the key's own
    # limit is the hard stop on spending.
    key_credit = credit.read_credit(
        os.getenv("OPENROUTER_API_KEY"),
        fetch_key_info if fetch_key_info is not None else credit.fetch_key_info,
    )
    show(key_credit.line())
    if key_credit.low:
        credit.warn(key_credit, github_actions=github_actions, step_summary=step_summary, out=show)

    log_path, read_paths = spend_log_paths()
    # The meter takes each UTC day's cap from the setting, so a run still
    # going at midnight after the cap's end date gets the default cap.
    meter = SpendMeter(
        cap_setting=cap_setting,
        log_path=log_path,
        read_paths=read_paths,
        run_mode=run_mode,
        dry_run=not publish_to_metaculus,
    )
    show(cap_setting.describe(meter.day))
    show(reservation_text(question_reservation, ensemble_size))
    show(meter.summary_line().replace(" in this run", " so far, before this run"))

    # Configure the bot. The models are set at the top of this file and in
    # routing.py (DEFAULT_FORECASTER, HIGH_VALUE_FORECASTER).
    bot = GlassEarthBot(
        run_mode=run_mode,
        spend_meter=meter,
        ensemble_size=ensemble_size,
        high_value_model=high_value_model,
        question_reservation_usd=question_reservation,
        ledger_key=ledger_key,
        research_reports_per_question=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        # A dry run keeps its full reports for reading; git ignores this folder.
        folder_to_save_reports_to=(
            DRY_RUN_LEDGER_PATH.parent.as_posix() if args.dry_run else None
        ),
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    # Before forecasting, a live run reveals the sealed forecasts whose
    # questions Metaculus now reports closed or resolved.
    if publish_to_metaculus:
        assert ledger_key is not None
        _reveal_closed_questions(
            ledger_key,
            post_statuses if post_statuses is not None else metaculus_post_statuses,
            github_actions=github_actions,
            step_summary=step_summary,
        )

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below. Targets run in order of prize value, so the cap
    # reaches the FutureEval tournament first.
    client = MetaculusClient()
    if args.question_url:
        bot.skip_previously_forecasted_questions = False
        found = bot.metaculus_client.get_question_by_url(
            args.question_url, group_question_mode="unpack_subquestions"
        )
        questions = found if isinstance(found, list) else [found]
        questions = [q for q in questions if q.state in (None, QuestionState.OPEN)]
        if args.max_questions is not None:
            questions = questions[: args.max_questions]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )
    elif run_mode == "tournament":
        forecast_reports = asyncio.run(
            _forecast_target(bot, client.CURRENT_AI_COMPETITION_ID, args.max_questions)
        )
        forecast_reports += asyncio.run(
            _forecast_target(bot, client.CURRENT_MINIBENCH_ID, args.max_questions)
        )
        if os.getenv(MARKET_PULSE_ENV, "").strip().lower() == "true":
            forecast_reports += asyncio.run(
                _forecast_target(bot, client.CURRENT_MARKET_PULSE_ID, args.max_questions)
            )
    elif run_mode == "market_pulse":
        # Market Pulse questions are forecast again on a schedule (reforecast.py).
        forecast_reports = asyncio.run(
            _forecast_target(bot, client.CURRENT_MARKET_PULSE_ID, args.max_questions)
        )
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            _forecast_target(bot, client.CURRENT_METACULUS_CUP_ID, args.max_questions)
        )
    else:
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            _forecast_target(bot, "bot-testing-area", args.max_questions)
        )

    # The template's report summary and banner name each question and can
    # quote errors, so in a quiet run they go only to the log file (as does
    # any print), and each question's outcome has already been noted.
    bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    show(meter.summary_line())
    if bot.skipped_not_due:
        show(f"{len(bot.skipped_not_due)} Market Pulse question(s) were not due for a new forecast.")
    # With all forecasting done: on GitHub, a publishing run with low credit
    # ends with a failing exit code once per UTC day, so that GitHub emails
    # the owner. The day goes in forecasts/credit_warnings.jsonl first.
    failure = credit.end_of_run_failure(
        key_credit,
        github_actions=github_actions,
        dry_run=not publish_to_metaculus,
        run_mode=run_mode,
        step_summary=step_summary,
    )
    if failure is not None:
        raise SystemExit(failure)
    return forecast_reports


def main(argv: Sequence[str] | None = None) -> int:
    """
    The command line (python main.py). A live run on GitHub Actions shows
    only content-free lines and keeps everything else in a log file on the
    runner (run_log.py); an exception that ends it shows as its type only.
    Any other run logs to the console as before.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    if not quiet_wanted(publishing=not args.dry_run):
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        run(argv)
        return 0
    path = start_quiet()
    show(
        "Live run on GitHub Actions: this log shows only lines with no question content. "
        f"The full log stays on the runner as {path.name} and is never uploaded or committed."
    )
    try:
        run(argv)
    except SystemExit as stop:
        return public_exit(stop)
    except BaseException as error:  # never the message: it can hold model output
        logger.error("The run stopped on an exception", exc_info=True)
        show(f"Stopped: {error_kind(error)}. {DETAILS}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
