# Glass Earth v0: glass-earth-bot

Glass Earth is a public AI forecasting desk. Its forecasts are timestamped and
scored. This repository is version 0: a bot for the Metaculus FutureEval bot
tournament, built from Metaculus's official template
([metac-bot-template](https://github.com/Metaculus/metac-bot-template)). The
tournament is the scoreboard.

## What it forecasts

| Target | ID | Where it comes from |
| --- | --- | --- |
| Fall 2026 FutureEval Bot Tournament | `33121` (slug `fall-futureeval-2026`) | `MetaculusClient.CURRENT_AI_COMPETITION_ID` in forecasting-tools 0.3.5 |
| MiniBench | `minibench` (always the current two-week round) | `MetaculusClient.CURRENT_MINIBENCH_ID` in forecasting-tools 0.3.5 |
| Bot testing area, for tests only | `bot-testing-area` (ID `32977`), not scored | `main.py --mode test_questions` |

These IDs were also checked on the live Metaculus pages on 1 November 2026.
The Fall 2026 tournament started on 28 September 2026 and takes forecasts until
6 January 2027. On 1 November, the next MiniBench round was due to run from 2
to 20 November 2026.

## How it works

1. **Research.** One web search summary per question, from
   `openai/gpt-6.1-terra` with OpenRouter's `:online` web search.
2. **Three forecasters**, one from each model family:
   `openai/gpt-6.1-sol`, `anthropic/claude-sonnet-5.5` and
   `google/gemini-4-argon`. Each one first names a reference class and states
   its base rate, then adjusts for the current evidence.
3. **Combining.** The median of the three for yes/no questions. For other
   types, the template's own method: the average of each option's probability
   for multiple choice, and the median of the three distributions for numeric
   and date questions.
4. **Publishing.** The forecast goes to Metaculus with a private comment
   holding all the reasoning. The comment states the submitted forecast and how
   it was combined, and the bot will not publish a comment that leaves it out.
   Metaculus makes these comments public after each question closes.
5. **Ledger.** Each forecast Metaculus accepts becomes one line in
   `forecasts/ledger.jsonl`, which the workflow commits.
6. **Proof.** A daily workflow stamps the ledger with OpenTimestamps.
   `forecasts/README.md` explains the ledger fields and how to check a receipt.

The bot never forecasts the same tournament question twice. It skips questions
Metaculus says it has already forecast, and anything already in the ledger.

The models are set at the top of `main.py` (`ENSEMBLE_MODELS`,
`RESEARCH_MODEL` and `HELPER_MODEL`).

## Setup

### 1. Add the two secrets

On GitHub, open this repository, then **Settings > Secrets and variables >
Actions > New repository secret**. Add these two names, each with its value:

- `METACULUS_TOKEN`: the token of your Metaculus bot account.
- `OPENROUTER_API_KEY`: your OpenRouter key.

The workflow files also list other optional names, such as `OPENAI_API_KEY`.
Leave those unset. Every model call goes through OpenRouter.

### 2. Turn on Actions

GitHub turns off Actions in a new fork. In the **Actions** tab, click **I
understand my workflows, go ahead and enable them**. This does not start
forecasting: the tournament workflow stays off until you switch it on (step 5).

### 3. Run the Test Bot

**Actions > Test Bot > Run workflow**. Leave **Post the forecasts to the bot
testing area** unticked for a dry run, then click the green **Run workflow**
button. The run uses OpenRouter credit. It researches one question at a time,
so a run over every test question can take a while.

A successful dry run:

- finishes with a green tick;
- shows `Running mode=test_questions, publish=no (dry run)` near the top of the
  **Run bot** step;
- shows each model's reasoning in that log, starting from its base rate;
- ends that step with `Bot produced (dry run) N forecast(s).` and one line per
  question with a tick;
- says `Nothing new to commit.` in the last step, because a dry run submits
  nothing.

If any question fails, the run shows a red cross and the log says which
question and why.

### 4. Test posting, if you want to

Run the Test Bot again with the box ticked. It posts forecasts and comments to
the bot testing area, where nothing is scored. The last step then commits the
new ledger lines (marked `test_questions`). To see the forecasts on Metaculus,
switch to your bot account (**Settings > My Forecasting Bots > Switch to bot
account**).

### 5. Go live

When you have approved a dry run: **Settings > Secrets and variables >
Actions > Variables > New repository variable**, name `GLASS_EARTH_LIVE`,
value `true`. The tournament workflow then runs every 20 minutes (at 7, 27 and
47 minutes past each hour) on the Fall 2026 tournament and MiniBench.

To pause, change the variable to `false`. The Metaculus Cup workflow has its
own switch, `GLASS_EARTH_CUP_LIVE`, and stays off without it. Bots win no
prizes there, and it forecasts every open question again on each run.

Metaculus asks every participant to fill in the first part of its
[participation form](https://forms.gle/aQdYMq9Pisrf1v7d8) before submitting
forecasts.

## Running it on your own computer

You need [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env` and
fill in the same two values. Git ignores `.env`.

```bash
uv sync --locked
uv run python main.py --mode test_questions --dry-run
```

`--dry-run` posts nothing. It saves the full reports and the ledger lines it
would have written in `forecasts/dry_run/`, which git also ignores. To keep
the Python environment out of a synced folder such as OneDrive, set
`UV_PROJECT_ENVIRONMENT` to a folder elsewhere first.

## Cost

Every model call is paid from your OpenRouter credit. Prices from
OpenRouter's model list on 1 November 2026, per million tokens:

| Model | Role | Input | Output |
| --- | --- | --- | --- |
| `openai/gpt-6.1-sol` | forecaster | US$2 | US$10 |
| `anthropic/claude-sonnet-5.5` | forecaster | US$2 | US$10 |
| `google/gemini-4-argon` | forecaster | US$2 | US$10 |
| `openai/gpt-6.1-terra` | research | US$1 | US$5, plus US$0.01 per web search |
| `openai/gpt-6-luna` | summary and parsing | US$0.10 | US$0.50 |

Estimate (an assumption, not a measurement): about US$0.10 to US$0.30 per
question. That is one research call, three forecasts, one summary and six small
parsing calls, at about 5,000 input and 2,000 to 6,000 output tokens per
forecast. A test run covers every open question in the bot testing area. Check
the real cost on OpenRouter's Activity page after the first run.

## Rules to keep in mind

From the FutureEval resources page:

- Submit only one forecast per question in the bot tournaments. Test in the
  bot testing area, where resubmitting is fine.
- Do not look at how the bot forecasts open or upcoming tournament questions
  and then change it for them, and do not rerun it on a question because you
  dislike its forecast.
- No human in the loop: the bot must forecast and comment through the API on
  its own.
- Bots must leave a comment with every forecast. The template posts private
  comments, as Metaculus asks.
