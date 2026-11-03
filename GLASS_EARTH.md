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
   `anthropic/claude-opus-5.5` by default, for every question. If the
   repository variable `GLASS_EARTH_HIGH_VALUE_MODEL` is `true`, high-value
   questions go to `anthropic/claude-fable-5.1` instead while the daily cap
   has room for it (see Spending). High value means a prize value of at least
   US$100 per question: the prize pool divided by the planned number of
   questions, times the question's weight. That is the FutureEval tournament
   (US$50,000 over 300 to 400 questions) and Market Pulse (US$7,500 over 65
   questions), but not MiniBench (US$1,000 over about 60) or the unscored
   areas.
3. **Forecasts.** One forecast per question by default. The repository
   variable `GLASS_EARTH_ENSEMBLE_SIZE` (1 to 5) asks the model for that many
   independent forecasts instead. Each forecast first names a reference class
   and states its base rate, then adjusts for the current evidence.
4. **Combining.** With one forecast, that forecast is submitted. With more,
   the median: of the probabilities for yes/no questions, of each option's
   probability for multiple choice (rescaled to sum to 100%), and of the
   cumulative distributions, point by point, for numeric, discrete and date
   questions. Then the guardrails (below) apply.
5. **Publishing.** The forecast goes to Metaculus with a private comment
   holding all the reasoning. The comment states the submitted forecast, how
   it was made ("1 forecast from anthropic/claude-opus-5.5", or "the median of
   3 independent forecasts from ...") and what it cost, and the bot will not
   publish a comment that leaves the forecast out. Metaculus makes these
   comments public after each question closes.
6. **Ledger.** Each forecast Metaculus accepts gets a sealed line in
   `forecasts/ledger.jsonl`, revealed once its question has closed (see
   Sealed until close), and the workflow commits it.
7. **Proof.** A daily workflow stamps the ledger with OpenTimestamps.
   `forecasts/README.md` explains the ledger fields and how to check a receipt.

### Guardrails

A log score has no floor: a confident forecast that turns out wrong can cost
a great deal. So `guardrails.py` limits every forecast the bot submits, after
its forecasts are combined:

- **Yes/no:** the probability stays between 2% and 98%.
- **Multiple choice:** every option gets at least 1%, taken from the options
  above 1% in proportion to their probabilities. With more than 50 options,
  the floor is 0.5 divided by the number of options.
- **Numeric, discrete and date:** 5% of the probability is spread evenly over
  the range, and at least 1% stays beyond each open bound. The forecast keeps
  the rest: 93% with both bounds open, 94% with one, 95% with none.
  forecasting-tools makes the same kind of mix with smaller weights (1% and
  0.1%) before submitting; the guardrail takes the place of that last step.
- **Conditional questions:** each part gets the guardrail for its type.

The comment states the guardrail next to the forecast, and the ledger records
the guarded forecast, which is exactly what Metaculus receives.

The trade-off, calculated in natural logs: when a numeric forecast is right,
its log score is about 0.06 lower than with the template's weights
(ln(0.93 / 0.988), both bounds open), and a yes/no forecast held at 98% that
is right loses 0.01. In return the worst cases are milder than before the
guardrails. A numeric surprise inside the range costs at most 2.8 (the log of
the forecast's density there against a flat forecast) instead of 3.9, and one
beyond an open bound at most 4.5 instead of 6.2. A yes/no surprise costs at
most 3.9 instead of 4.6, and a multiple choice one 4.6 instead of 6.9.

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
hours old). Each forecast gets its own ledger line, with the reason in its
record.

### Sealed until close

The ledger is public, and other bots may use any forecast they can see while
its question is open; the Metaculus Terms of Use also bar republishing site
material. So the ledger never shows a forecast before its question closes,
and never holds a question's title or other text copied from Metaculus.

For each forecast the ledger gets a sealed line: the question's ID and
address, when the forecast was made, and the SHA-256 hash of the forecast's
record. The record holds the forecast, how it was made, what it cost and 32
random bytes of salt, so that nobody can find the forecast by hashing
guesses. It goes in `forecasts/sealed_boxes.jsonl`, encrypted with AES-GCM
under the key in the repository secret `GLASS_EARTH_LEDGER_KEY`.

Each live run first asks Metaculus about the questions still sealed, and
once Metaculus reports a question closed or resolved, adds its records to
the ledger. Anyone can check a record against its sealed line and an
OpenTimestamps receipt; `forecasts/README.md` shows how. A live run without
a usable key stops before it fetches or spends anything, and dry runs seal
nothing. `scripts/backup_salts.py` copies the salts and records to a folder
away from GitHub.

### Run logs

Anyone signed in to GitHub can read this repository's workflow logs, while
the questions a run forecast are still open. So a live run on GitHub Actions
shows only lines with no question content (`run_log.py`): the run's start
and setup, the credit and cap lines, the ledger reveal counts, how many
questions were found, selected, skipped (and why) and posted, and for each
question its ID, address, outcome and cost. An error shows as its exception's
type only, never its message. Everything else (research, reasoning,
forecasts, the template's report summary, warnings, tracebacks) goes to a log
file on the runner, which is never uploaded or committed and goes when the
runner does; each forecast's reasoning is still in its comment on Metaculus.
The offline tests' output stays in a file on the runner too, and the log
shows only their result. Dry runs, and runs on your own computer, log
everything as before.

### Spending

`spend.py` meters every model call. Before each call the bot works out the
most it could cost (every output token used, at OpenRouter's list price), and
it never starts a call that could take the day's spending over the **daily
cap** (default **US$2.00** per UTC day), except to let a question that has
already started finish (below). The repository variable
`GLASS_EARTH_DAILY_CAP_USD` changes the cap, and
`GLASS_EARTH_DAILY_CAP_UNTIL_UTC` (a date, `YYYY-MM-DD`) gives that cap a
last UTC day: from the next day the default US$2.00 applies again, and the
default reserve below follows it. A run still going at midnight UTC takes
the new day's cap.

Before each question the bot checks that the question's **reservation** fits
in what is left of the cap; if not, it skips the question and logs why,
instead of failing halfway. The reservation is twice the measured cost of one
real question, rounded up to the cent: **US$0.27 per Opus forecast**
(calculated from US$0.13286552, observed for one test-area question on 3
November 2026; see Cost), times the number of forecasts, and 4 times that
for a conditional question, which forecasts four linked questions. The
repository variable `GLASS_EARTH_QUESTION_RESERVATION_USD` changes the amount
per forecast. A Fable question must fit at its calculated worst case instead.

A question can cost more than its reservation. Once it has started, its calls
may take the day's spending past the cap by up to the question's calculated
worst case minus its reservation, so that a question started near the cap is
not stopped after its research is paid for. With one Opus forecast that is
about US$0.19 for a short question and about US$0.20 for the measured one
(calculated); it grows by about US$0.002 for every 1,000 characters of
question text. A call that needed this allowance is marked in
`forecasts/spend.jsonl`. No new question can start once the cap is passed,
so a day ends at most one question's allowance over the cap: about US$0.20
for a question of typical length (calculated). The OpenRouter key's own
limit is the hard stop.

Low-value questions do not start in the part of the cap kept for high-value
questions: 40% of the cap, at most US$2.00 (US$0.80 at the default cap). One
that then costs more than its reservation can use a little of that part.
When Fable is switched on, it is used only while the cap still has room for
a Fable question at its worst case plus one more Opus question's
reservation.

At the measured cost, the default cap lets about 14 high-value questions
start in a UTC day, or about 7 low-value ones, which stay out of the US$0.80
reserve. A US$4.00 cap lets about 29 high-value or 17 low-value questions
start (its reserve is US$1.60). All four figures are calculated from the
measured cost of one question.

Questions run one at a time, so each cap check sees everything spent before
it. The forecasts of one question run together, and the check counts all of
them. A second batch started against the same cap while one is running is
refused before it spends anything.

The cost of each call comes from OpenRouter's own report on the response
where it gives one, otherwise from the token counts times the list price. A
cost that cannot be known is recorded as `Unknown`, never as zero, and the
cap counts it at its worst case. Every call is a line in
`forecasts/spend.jsonl`, which the workflows commit after each run so that
the next run counts it.

At the start of each run the bot asks OpenRouter how much is left on its key
(a free call to `https://openrouter.ai/api/v1/key`). With less than US$5.00
left, the run says so in its log, as a warning on the GitHub Actions run and
on the run's summary page. Once a UTC day, a run on GitHub that can publish
then ends with a red cross after all its forecasting is done, so that GitHub
emails you as it does for any failed run. `forecasts/credit_warnings.jsonl`
records each day it did this. If OpenRouter does not answer, the run logs the
credit as `Unknown` and carries on.

The models, token limits and worst-case assumptions are set at the top of
`main.py`; the prices and the reservation are in `spend.py`, and the credit
check is in `credit.py`.

## Setup

### 1. Add the three secrets

On GitHub, open this repository, then **Settings > Secrets and variables >
Actions > New repository secret**. Add these three names, each with its value:

- `METACULUS_TOKEN`: the token of your Metaculus bot account.
- `OPENROUTER_API_KEY`: your OpenRouter key.
- `GLASS_EARTH_LEDGER_KEY`: the key that seals the ledger, 32 random bytes
  in URL-safe base64 (`.env.example` shows how to make one). Keep a copy
  somewhere safe: GitHub never shows a secret again, and without the key no
  sealed forecast can be revealed.

The workflow files also list other optional names, such as `OPENAI_API_KEY`.
Leave those unset. Every model call goes through OpenRouter.

`scripts/go_live.py` can do this step from your own computer, reading all
three values from `.env` without showing them (see the top of that file).

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
  **Run bot** step, then a `Setup:` line naming the forecasts per question
  and whether Fable is on, the credit left on the OpenRouter key, and the
  daily cap;
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
new sealed ledger lines and their boxes. To see the forecasts on Metaculus,
switch to your bot account (**Settings > My Forecasting Bots > Switch to bot
account**).

### 5. Go live

When you have approved a dry run: **Settings > Secrets and variables >
Actions > Variables > New repository variable**, name `GLASS_EARTH_LIVE`,
value `true`. The tournament workflow then runs every 20 minutes (at 7, 27 and
47 minutes past each hour) on the Fall 2026 tournament and MiniBench.

Optional repository variables (the same page, **New repository variable**):

- `GLASS_EARTH_DAILY_CAP_USD`: the daily cap in US dollars (default `2.00`).
- `GLASS_EARTH_DAILY_CAP_UNTIL_UTC`: the last UTC day that cap applies,
  written `YYYY-MM-DD` (for example `2026-11-08`). From the next UTC day the
  cap is `2.00` again. Unset, the cap has no end date.
- `GLASS_EARTH_HIGH_VALUE_RESERVE_USD`: the part of the cap kept for
  high-value questions (default 40% of the cap, at most `2.00`: `0.80` at the
  default cap, `1.60` at a cap of `4.00`).
- `GLASS_EARTH_QUESTION_RESERVATION_USD`: what a question needs left of the
  cap to start, per Opus forecast (default `0.27`, twice the measured cost of
  one question).
- `GLASS_EARTH_ENSEMBLE_SIZE`: forecasts per question, a whole number from
  `1` to `5` (default `1`). With more than one, the median is submitted.
- `GLASS_EARTH_HIGH_VALUE_MODEL`: `true` sends high-value questions to Fable
  while the cap has room (default off). Fable needs room for its own worst
  case plus an Opus question's reservation: about US$1.09 with one forecast
  and US$2.87 with three (calculated), so with three raise the cap above
  that, or every question gets Opus.
- `GLASS_EARTH_MARKET_PULSE`: `true` adds Market Pulse to each tournament run.
  Market Pulse allows one entry per person, so decide first whether to enter
  as the bot.

A value the bot cannot use for the cap, its end date, the reserve, the
reservation, the ensemble size or the Fable switch stops the run with a
message naming the variable, before anything is fetched or spent; it never
falls back to a default. Cost shows what each setup costs per question.

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
fill in the same values (a dry run needs only the first two). Git ignores
`.env`.

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
for that run (a cap set this way has no end date). To try another setup, set
`GLASS_EARTH_ENSEMBLE_SIZE` or `GLASS_EARTH_HIGH_VALUE_MODEL` in the shell
first. To keep the Python environment out of a synced folder such as
OneDrive, set `UV_PROJECT_ENVIRONMENT` to a folder elsewhere first.

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

Cost per question for each setup. Fable applies only to high-value
questions; the others always get Opus.

| Setup | Repository variables | Typical (calculated from an assumption) | Needed to start (calculated) | Worst case, short question (calculated) |
| --- | --- | --- | --- | --- |
| Opus 5.5, 1 forecast (default) | none | about US$0.12 (US$0.13 measured) | US$0.27 | about US$0.46 |
| Opus 5.5, 3 forecasts | `GLASS_EARTH_ENSEMBLE_SIZE=3` | about US$0.27 | US$0.81 | about US$0.98 |
| Fable 5.1, 1 forecast | `GLASS_EARTH_HIGH_VALUE_MODEL=true` | about US$0.23 | its worst case | about US$0.82 |
| Fable 5.1, 3 forecasts | both of the above | about US$0.61 | its worst case | about US$2.06 |

One real question has been measured. The 3 November 2026 dry run of one
test-area question (numeric, one Opus forecast) cost US$0.13286552: OpenRouter
reported that much used on the key, and the bot's own record of the run's five
calls adds up to the same (observed). Research cost US$0.0875395, the research
summary US$0.0002966, the forecast US$0.044644 and two parsing calls
US$0.00038542, so research cost more than the assumption below and the
forecast less. What an Opus question needs to start, its reservation, is
twice that cost rounded up to the cent, per forecast.

The worst case comes from these prices and the token limits in `main.py`. A
Fable question needs it to start; an Opus question that has started may go
up to it, past its reservation (see Spending). The typical figures are
calculated from an assumption, not a measurement:

- Research: one call of about 4,000 tokens in and 3,000 out, with two web
  searches: US$0.004 + US$0.015 + US$0.02 = US$0.039.
- Research summary: about 3,000 tokens in and 300 out on `gpt-6-luna`:
  US$0.0003 + US$0.00015 = US$0.00045.
- Each forecast: about 4,000 tokens in and 3,000 out. Opus: US$0.016 +
  US$0.06 = US$0.076. Fable: US$0.04 + US$0.15 = US$0.19. Plus its two
  parsing calls on `gpt-6-luna`, about 3,000 tokens in and 100 out each:
  2 x (US$0.0003 + US$0.00005) = US$0.0007.
- Opus, 1 forecast: 0.039 + 0.00045 + 0.0767 = US$0.11615.
- Opus, 3 forecasts: 0.039 + 0.00045 + 3 x 0.0767 = US$0.26955.
- Fable, 1 forecast: 0.039 + 0.00045 + 0.1907 = US$0.23015.
- Fable, 3 forecasts: 0.039 + 0.00045 + 3 x 0.1907 = US$0.61155.

In this assumption research is about a third of the default's cost; in the
measured question it was about two thirds. Each line of
`forecasts/spend.jsonl` shows the real cost; OpenRouter's Activity page
shows the same.

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
