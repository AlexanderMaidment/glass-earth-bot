# Forecast records

This folder is the public record of every forecast glass-earth-bot submits to
Metaculus.

## ledger.jsonl

The ledger is public, and other bots may use any forecast they can see while
its question is open. The Metaculus Terms of Use also bar republishing site
material without permission. So every forecast here is sealed until its
question closes, and the ledger never holds a question's title, a
sub-question's label or any other text copied from Metaculus.

The bot adds lines as it works, and the workflow that ran the bot commits the
file at the end of the run. Lines are only ever added, never changed or
removed. There are two kinds of line: a sealed line for each forecast, and
later a reveal line for it.

### Sealed lines

At the moment Metaculus accepts a forecast, the bot adds a sealed line. It
shows that a forecast was made, on which question and when, and fixes the
forecast without showing it:

| Field | Meaning |
| --- | --- |
| `ledger_version` | Format version (currently 3). |
| `sealed` | Always `true`. |
| `question_id` | Metaculus question ID. A group post has one per sub-question. |
| `url` | The question page. |
| `recorded_at_utc` | When Metaculus accepted the forecast, in UTC. |
| `seal_sha256` | The SHA-256 hash of the forecast's record (below), in hex. |

The record itself waits, encrypted, in `sealed_boxes.jsonl`.

### Reveal lines

Each live run first asks Metaculus about the questions of the sealed lines
not yet revealed. Once Metaculus reports a question closed or resolved, the
bot adds a reveal line for each sealed line on it:

| Field | Meaning |
| --- | --- |
| `revealed` | Always `true`. |
| `seal_sha256` | The sealed line this reveals. |
| `revealed_at_utc` | When the bot revealed it, in UTC. |
| `salt` | The record's salt, shown on its own as well. |
| `record` | The forecast's record. |

The scheduled close time is never enough, because Metaculus can move a
close. If Metaculus gives an error or answers 429 (too many requests), the
run reveals nothing more and forecasts as usual, and a later run tries again.
A run asks about at most 20 posts, those due to close soonest first, each
with one read-only request.

### The record

The record is the full line the bot would have written without sealing,
minus the text copied from Metaculus (the title, the sub-question's label and
the forecast written out in words), plus a salt. The salt is 32 random bytes
in hex, new for each forecast. Without it a guessed forecast could be hashed
and checked against `seal_sha256`, so it stays sealed with the rest until the
reveal.

| Field | Meaning |
| --- | --- |
| `ledger_version`, `question_id`, `url`, `recorded_at_utc` | As in the sealed line. Version 3 added `scheduled_close_time_utc` and sealing; earlier versions were never published. |
| `post_id` | Metaculus post ID, the number in the question's web address. |
| `type` | `binary` (yes or no), `multiple_choice`, `numeric`, `discrete` or `date`. |
| `tournaments` | The tournaments the question belongs to. |
| `scheduled_close_time_utc` | When the question was due to close, as Metaculus said when it was forecast. |
| `run_mode` | `tournament`, `market_pulse`, `metaculus_cup`, or `test_questions` for the unscored bot testing area. |
| `final_forecast` | Exactly what Metaculus was sent: `probability_yes`, `probability_yes_per_category` (keyed by the options' names, as Metaculus requires), or the `continuous_cdf` (201 points for a numeric question, one more than the number of outcomes for a discrete one). |
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
| `salt` | 32 random bytes in hex. |

The bot also posts its full reasoning as a private comment on each question,
and Metaculus makes those comments public after the question closes.

### Checking a reveal

Anyone can check a reveal without the key:

1. Find the sealed line with the reveal's `seal_sha256`. It must come earlier
   in the file.
2. Write the reveal's `record` as canonical JSON: keys sorted, no spaces
   (separators `,` and `:`), characters as themselves, encoded as UTF-8.
3. The SHA-256 of those bytes must be the sealed line's `seal_sha256`, and
   the record's `ledger_version`, `question_id`, `url` and `recorded_at_utc`
   must be the sealed line's. The `salt` shown on its own must be the
   record's.
4. A proof receipt (see proofs/ below) for a version of the ledger that holds
   the sealed line shows that the forecast was fixed no later than its
   Bitcoin block. Compare that with when the question closed.

In Python, for every reveal in the ledger:

```python
import hashlib
import json

lines = [json.loads(l) for l in open("forecasts/ledger.jsonl", encoding="utf-8") if l.strip()]
sealed = {l["seal_sha256"]: (n, l) for n, l in enumerate(lines) if l.get("sealed")}
for n, reveal in enumerate(lines):
    if not reveal.get("revealed"):
        continue
    where, line = sealed[reveal["seal_sha256"]]
    record = reveal["record"]
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert where < n
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == line["seal_sha256"]
    assert reveal["salt"] == record["salt"]
    for name in ("ledger_version", "question_id", "url", "recorded_at_utc"):
        assert record[name] == line[name]
print("Every reveal matches its sealed line.")
```

A dry run submits nothing and seals nothing. It writes the full line it would
have written, title and forecast in words included, to `forecasts/dry_run/`,
which git ignores.

A Market Pulse question can have several sealed lines, one per forecast,
because Market Pulse scores the forecast standing at each question's close
and expects bots to update. FutureEval and MiniBench questions have one each.

## sealed_boxes.jsonl

One line per sealed forecast, `{"seal_sha256": ..., "box": ...}`, written
just before its sealed line and committed with the ledger. `box` is base64 of
a random 12-byte nonce followed by the AES-GCM ciphertext, with its 16-byte
tag, of the record as canonical JSON. The 256-bit key is held only by the
bot's owner and in the repository's GitHub Actions secret
`GLASS_EARTH_LEDGER_KEY`, and `seal_sha256` (its hex text, as ASCII bytes) is
the associated data, so a box cannot be moved to another seal. Without the
key a box shows nothing but its length.

A reveal opens the box and checks the record's hash before writing it.
`scripts/backup_salts.py` copies the salts and records out of the boxes to a
folder away from GitHub, so that they survive if the key or this file is lost.
A sealed line whose box is missing can never be revealed: each run says so in
its log and on its summary page, and the backup reports it and fails.

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
one commit, existed no later than a certain Bitcoin block. Each sealed line in
it fixes its forecast through `seal_sha256`, so with the later reveal anyone
can use a receipt to check that the forecasts in that version of the ledger
were made by then. A forecast added later with hindsight, or edited after the
event, would not match the earlier receipts.

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
