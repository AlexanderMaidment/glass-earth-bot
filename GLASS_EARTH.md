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
| Market Pulse 26Q4, only when switched on | `market-pulse-26q4` | `MetaculusClient.CURRENT_MARKET_PULSE_ID` in forecasting-tools 0.3.5 |
| Bot testing area, for tests only | `bot-testing-area` (ID `32977`), not scored | `main.py --mode test_questions` |

These IDs were also checked on the live Metaculus pages on 1 and 3 November
2026. The Fall 2026 tournament started on 28 September 2026; questions stop
opening a few weeks before 1 January 2027. MiniBench rounds last two weeks,
and most of a round's questions open in its first days. Market Pulse 26Q4
runs until 31 December 2026, with 21 December the last day for prize
participation, and allows one entry per person, as a bot or as a human.

## How it works

1. **Research.** One web search summary per question, from
   `openai/gpt-6.1-terra` with OpenRouter's `:online` web search.
2. **Model.** `routing.py` picks one model per question:
   `anthropic/claude-opus-5.5` by default, and `anthropic/claude-fable-5.1`
   for high-value questions while the daily cap has room for it (see
   Spending). High value means a prize value of at least US$100 per question:
   the prize pool divided by the planned number of questions, times the
   question's weight. That is the FutureEval tournament (US$50,000 over 300 to
   400 questions) and Market Pulse (US$7,500 over 65 questions), but not
   MiniBench (US$1,000 over about 60) or the unscored areas.
3. **Three forecasts.** The model forecasts the question three times,
   independently. Each forecast first names a reference class and states its
   base rate, then adjusts for the current evidence.
4. **Combining.** The median of the three: of the probabilities for yes/no
   questions, of each option's probability for multiple choice (rescaled to
   sum to 100%), and of the cumulative distributions, point by point, for
   numeric, discrete and date questions.
5. **Publishing.** The forecast goes to Metaculus with a private comment
   holding all the reasoning. The comment states the submitted forecast, how
   it was combined and what it cost, and the bot will not publish a comment
   that leaves the forecast out. Metaculus makes these comments public after
   each question closes.
6. **Ledger.** Each forecast Metaculus accepts becomes one line in
   `forecasts/ledger.jsonl`, with its cost, which the workflow commits.
7. **Proof.** A daily workflow stamps the ledger with OpenTimestamps.
   `forecasts/README.md` explains the ledger fields and how to check a receipt.

### Group questions

Market Pulse and some tournament questions are groups: one post with several
sub-questions, each with its own label (a company, a date range, a measure).
The bot forecasts each open sub-question on its own, states its label in
every prompt and in the comment, posts a distribution with the sub-question's
own number of points (201 for numeric, fewer for discrete), and skips
sub-questions that are closed or not yet open.

### Forecasting again

The FutureEval tournament and MiniBench allow one forecast per question, so
the bot never forecasts their questions twice: it skips questions Metaculus
says it has already forecast, and anything already in the ledger.

Market Pulse scores the forecast standing when each question closes, and
bots there must keep their forecasts up to date. `reforecast.py` forecasts a
Market Pulse question again when its last forecast is 7 days old, and once
more in the 48 hours before it closes (if the last forecast is at least 20
hours old). Each forecast gets its own ledger line with the reason.

### Spending

`spend.py` meters every model call. Before each call the bot works out the
most it could cost (every output token used, at OpenRouter's list price), and
it never starts a call that could take the day's spending over the **daily
cap** (default **US$5.00** per UTC day). Before each question it checks that
the question's worst case fits; if not, it skips the question and logs why,
instead of failing halfway. Low-value questions do not spend into the last
US$2.00 of the cap (or 40% of a cap under US$5), which is kept for high-value
questions, and Fable is used only while the cap still has room for a Fable
question plus one more default question.

The cost of each call comes from OpenRouter's own report on the response
where it gives one, otherwise from the token counts times the list price. A
cost that cannot be known is recorded as `Unknown`, never as zero, and the
cap counts it at its worst case. Every call is a line in
`forecasts/spend.jsonl`, which the workflows commit after each run so that
the next run counts it.

The models, token limits and worst-case assumptions are set at the top of
`main.py`; the prices are in `spend.py`.

## Setup

### 1. Add the two secrets

On GitHub, open this repository, then **Settings > Secrets and variables >
Actions > New repository secret**. Add these two names, each with its value:

- `METACULUS_TOKEN`: the token of your Metaculus bot account.
- `OPENROUTER_API_KEY`: your OpenRouter key.

The workflow files also list other optional names, such as `OPENAI_API_KEY`.
Leave those unset. Every model call goes through OpenRouter.

`scripts/go_live.py` can do this step from your own computer, reading both
values from `.env` without showing them (see the top of that file).

### 2. Turn on Actions

GitHub turns off Actions in a new fork. In the **Actions** tab, click **I
understand my workflows, go ahead and enable them**. This does not start
forecasting: the tournament workflow stays off until you switch it on (step 5).

### 3. Run the Test Bot

**Actions > Test Bot > Run workflow**. Leave **Post the forecasts to the bot
testing area** unticked for a dry run, then click the green **Run workflow**
button. The run uses OpenRouter credit. It forecasts one question at a time,
so a run over every test question can take a while.

A successful dry run:

- finishes with a green tick;
- passes the **Run the offline tests** step first;
- shows `Running mode=test_questions, publish=no (dry run)` near the top of the
  **Run bot** step;
- shows each forecast's reasoning in that log, starting from its base rate;
- ends that step with `Bot produced (dry run) N forecast(s).`, one line per
  question with a tick, and a `Spend (UTC day ...)` line;
- commits only `forecasts/spend.jsonl` in the last step, because a dry run
  submits nothing but still spends.

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

Optional repository variables:

- `GLASS_EARTH_DAILY_CAP_USD`: the daily cap in US dollars (default `5.00`).
- `GLASS_EARTH_HIGH_VALUE_RESERVE_USD`: the part of the cap kept for
  high-value questions (default `2.00`, or 40% of a cap under US$5).
- `GLASS_EARTH_MARKET_PULSE`: `true` adds Market Pulse to each tournament run.
  Market Pulse allows one entry per person, so decide first whether to enter
  as the bot.

To pause, change `GLASS_EARTH_LIVE` to `false`. The Metaculus Cup workflow has
its own switch, `GLASS_EARTH_CUP_LIVE`, and stays off without it. Bots win no
prizes there, and it forecasts every open question again on each run.

All workflows that can spend on models share one queue, so their runs never
overlap and each sees what the runs before it spent. GitHub cancels an older
waiting run when a newer one joins the queue, so a manual Test Bot run started
during a long tournament run may need starting again.

Metaculus asks every participant to fill in the first part of its
[participation form](https://forms.gle/aQdYMq9Pisrf1v7d8) before submitting
forecasts.

## Running it on your own computer

You need [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env` and
fill in the same two values. Git ignores `.env`.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests    # offline, free
uv run --locked python main.py --mode test_questions --dry-run --max-questions 1
```

`--dry-run` posts nothing. It saves the full reports, the ledger lines it
would have written and its spending in `forecasts/dry_run/`, which git also
ignores. `--max-questions N` limits each target to N questions,
`--question-url URL` forecasts one question (dry runs only), and
`--daily-cap-usd X` and `--high-value-reserve-usd Y` set the cap and reserve
for that run. To keep the Python environment
out of a synced folder such as OneDrive, set `UV_PROJECT_ENVIRONMENT` to a
folder elsewhere first.

The offline tests use made-up questions and model answers (marked SYNTHETIC)
and refuse every network connection.

## Cost

Every model call is paid from your OpenRouter credit. Prices observed on
OpenRouter's public model list (`GET https://openrouter.ai/api/v1/models`) at
2026-11-03T04:29:51Z, per million tokens:

| Model | Role | Input | Output |
| --- | --- | --- | --- |
| `anthropic/claude-opus-5.5` | forecaster (default) | US$4 | US$20 |
| `anthropic/claude-fable-5.1` | forecaster (high value) | US$10 | US$50 |
| `openai/gpt-6.1-terra` | research | US$1 | US$5, plus US$0.01 per web search |
| `openai/gpt-6-luna` | summary and parsing | US$0.10 | US$0.50 |

Worst case per question (calculated by the cap check from these prices and
the token limits in `main.py`): about US$1 with Opus and US$2 with Fable for
a short question. Typical (an assumption, not a measurement: about 4,000
input and 3,000 output tokens per forecast, one research call with two
searches): about US$0.27 per question with Opus and US$0.61 with Fable. Each
line of `forecasts/spend.jsonl` shows the real cost; OpenRouter's Activity
page shows the same.

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
- Market Pulse needs numeric group questions handled and forecasts kept up
  to date during each question's life.
