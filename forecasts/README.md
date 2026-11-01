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
| `ledger_version` | Format version of the line (currently 1). |
| `recorded_at_utc` | When Metaculus accepted the forecast, in UTC. |
| `question_id` | Metaculus question ID. A group post has one per sub-question. |
| `post_id` | Metaculus post ID, the number in the question's web address. |
| `url` | The question page. |
| `type` | `binary` (yes or no), `multiple_choice`, `numeric`, `discrete` or `date`. |
| `group_option` | For a sub-question of a group, the option it is about. |
| `title` | The question as written. |
| `tournaments` | The tournaments the question belongs to. |
| `run_mode` | `tournament`, `metaculus_cup`, or `test_questions` for the unscored bot testing area. |
| `final_forecast` | Exactly what Metaculus was sent: `probability_yes`, `probability_yes_per_category`, or the 201-point `continuous_cdf`. |
| `final_forecast_readable` | The same forecast as the comment on Metaculus states it. |
| `combined_by` | How the model forecasts were combined. |
| `model_forecasts` | Each model's own forecast, before combining. |
| `models_without_forecast` | Ensemble models that failed on this question. |
| `code_commit` | The commit of this repository that made the forecast. |
| `run_url` | The GitHub Actions run. Its log shows the full reasoning for 90 days. |

The bot also posts its full reasoning as a private comment on each question,
and Metaculus makes those comments public after the question closes.

A dry run submits nothing. It writes the forecast it would have submitted to
`forecasts/dry_run/`, which git ignores.

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
