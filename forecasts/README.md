# Forecast records

This folder is the public record of every forecast glass-earth-bot submits to
Metaculus.

## ledger.jsonl

The ledger has one line per submitted forecast. The bot adds the line at the
moment Metaculus accepts the forecast, and the workflow that ran the bot
commits the file at the end of the run. Lines are only ever added, never
changed or removed.

Each line is a JSON object with these fields:

| Field | Meaning |
| --- | --- |
| `ledger_version` | Format version of the line (currently 2; version 1 lines lack the fields from `forecaster_model` on). |
| `recorded_at_utc` | When Metaculus accepted the forecast, in UTC. |
| `question_id` | Metaculus question ID. A group post has one per sub-question. |
| `post_id` | Metaculus post ID, the number in the question's web address. |
| `url` | The question page. |
| `type` | `binary` (yes or no), `multiple_choice`, `numeric`, `discrete` or `date`. |
| `group_option` | For a sub-question of a group, the option it is about. |
| `title` | The question as written. |
| `tournaments` | The tournaments the question belongs to. |
| `run_mode` | `tournament`, `market_pulse`, `metaculus_cup`, or `test_questions` for the unscored bot testing area. |
| `final_forecast` | Exactly what Metaculus was sent: `probability_yes`, `probability_yes_per_category`, or the `continuous_cdf` (201 points for a numeric question, one more than the number of outcomes for a discrete one). |
| `final_forecast_readable` | The same forecast as the comment on Metaculus states it. |
| `combined_by` | How the final forecast was made: "1 forecast from ..." (the default), or how several were combined. |
| `model_forecasts` | Each forecast before combining (one by default, `GLASS_EARTH_ENSEMBLE_SIZE` in all), with its model, and its member number when there is more than one. |
| `models_without_forecast` | Forecasts (model, and member number when there is more than one) that failed on this question. |
| `forecaster_model` | The model that made the forecasts (see `routing.py`). |
| `route` | Why that model was chosen, from the question's prize value and the daily cap. |
| `forecast_reason` | Why the question was forecast now: a first forecast, or for Market Pulse a scheduled re-forecast or the final refresh before close. |
| `cost_usd` | What forecasting this question cost, in US dollars, over all its model calls; `Unknown` if any call's cost could not be known. |
| `cost_usd_cap_basis` | What the daily cap counted for it: known costs, plus the worst case of each unknown one. |
| `cost_by_model_usd` | The known part of the cost, by model. |
| `model_calls` | How many model calls the question took. |
| `prices_retrieved_at_utc` | When the prices used for calculated costs were read from OpenRouter's model list. |
| `code_commit` | The commit of this repository that made the forecast. |
| `run_url` | The GitHub Actions run. Its log shows the full reasoning for 90 days. |

The bot also posts its full reasoning as a private comment on each question,
and Metaculus makes those comments public after the question closes.

A dry run submits nothing. It writes the forecast it would have submitted to
`forecasts/dry_run/`, which git ignores.

A Market Pulse question can have several lines, one per forecast, because
Market Pulse scores the forecast standing at each question's close and
expects bots to update. FutureEval and MiniBench questions have one line each.

## spend.jsonl

One line per model call, per refused call and per question skipped by the
daily cap, written by every run on GitHub, dry runs included, because a dry
run spends too. The bot reads today's lines (by UTC day) at the start of each
run, so the cap holds across runs. Runs on your own computer write to
`forecasts/dry_run/spend.jsonl` instead and count both files.

| Field | Meaning |
| --- | --- |
| `event` | `model_call`, `call_refused` (the cap had no room) or `question_skipped`. |
| `utc_day` | The day the cap counts it against. |
| `question_id`, `post_id` | The question the call was for. |
| `purpose`, `model` | `researcher`, `summarizer`, `forecaster` or `parser`, and the model. |
| `outcome` | `ok`, or `failed (...)` with the error type. |
| `prompt_tokens`, `completion_tokens`, `cached_prompt_tokens` | As the response reported them. |
| `cost_usd` | The call's cost in US dollars, or `Unknown`. |
| `cost_source` | `openrouter_reported`, `calculated_from_tokens` or `unknown`. |
| `worst_case_usd` | For a call: the most it could have cost, which the cap checked before it. For a skipped question: its calculated worst case, for information. |
| `reservation_usd` | Only for a skipped question: what it needed left of the cap to start (its reservation), which did not fit. |
| `over_cap_allowance_usd` | Only on a call that took the day past the daily cap under the allowance of a question already started: that question's allowance, its calculated worst case minus its reservation. |
| `cap_basis_usd` | What the cap counts: the cost, or the worst case when the cost is unknown. |
| `daily_cap_usd` | The cap in force when the line was written (the default `2.00` again after the last day set by `GLASS_EARTH_DAILY_CAP_UNTIL_UTC`). |

## credit_warnings.jsonl

One line for each UTC day on which a run on GitHub ended with a failure on
purpose because the bot's OpenRouter key had less than US$5.00 left, so that
GitHub emailed the owner. Each run reads it so that this happens at most once
a UTC day. The file appears after the first such day.

| Field | Meaning |
| --- | --- |
| `credit_warning_version` | Format version of the line (currently 1). |
| `recorded_at_utc`, `utc_day` | When the run failed on purpose, and the UTC day it counts for. |
| `event` | `low_credit_run_failed`. |
| `limit_remaining_usd` | What OpenRouter reported left on the key at the start of that run. |
| `limit_usd` | The key's limit as OpenRouter reported it, or `Unknown`. |
| `warn_below_usd` | The warning level (`5.00`). |
| `run_mode`, `code_commit`, `run_url` | The run that failed on purpose. |

## proofs/

Once a day, a workflow stamps the ledger with
[OpenTimestamps](https://opentimestamps.org), if it has changed since the last
stamp. Each receipt (`.ots` file) proves that the ledger, exactly as it was at
one commit, existed no later than a certain Bitcoin block. Anyone can use a
receipt to check that the forecasts in that version of the ledger were written
by then. A forecast added later with hindsight, or edited after the event,
would not match the earlier receipts.

A new receipt is pending until a Bitcoin block confirms it, usually within a
day. The next daily run upgrades it to a complete proof.

`proofs/index.jsonl` has one line per receipt: when it was made, the ledger's
SHA-256 hash, size in bytes and line count, the commit whose ledger it covers
(`ledger_commit`), and the receipt's path.

To check a receipt yourself, use a Unix-style shell such as Git Bash (Windows
PowerShell's `>` changes the bytes of what it saves):

1. Take a line from `proofs/index.jsonl`.
2. Save that version of the ledger:
   `git show <ledger_commit>:forecasts/ledger.jsonl > ledger-snapshot.jsonl`
3. Check that `sha256sum ledger-snapshot.jsonl` matches `ledger_sha256`.
4. Verify the receipt: drop `ledger-snapshot.jsonl` and the `.ots` file onto
   https://opentimestamps.org, or run
   `ots verify -f ledger-snapshot.jsonl <receipt>` (this needs a Bitcoin node).

Lines are only ever added to the ledger, so the snapshot is also the first
`ledger_bytes` bytes of today's ledger.
