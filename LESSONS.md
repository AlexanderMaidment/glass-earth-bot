# Lessons

What glass-earth-bot learns from the questions it has forecast, once they resolve. Every morning at 07:00 in Brisbane (21:00 UTC), `scripts/learning_loop.py` asks Metaculus about the questions in the bot's ledger and reports here the ones that have newly resolved. Each run rewrites this file from `forecasts/learning_state.json`.

## Rules

- Resolved questions only. The loop never learns from the bot's forecasts on open questions, and nothing about an open question is written here or in the state file.
- Annulled and ambiguous questions are left out, and so are the questions of the unscored bot testing area.
- The bot changes at most once a week, and only after the change beats the old version on closed questions, with research search capped at each question's close date.
- Until a date-capped search exists (AskNews is the planned one), no lesson-driven change is made. These lessons are notes; they change nothing by themselves.
- Questions are named by their Metaculus ID and link only. Metaculus's terms of use do not allow republishing its site material, so no question title, question text or option label is copied here or into the state file, and each cause describes the miss in its own words.

## How the numbers are worked out

- Observed: each outcome, and Metaculus's own baseline and peer scores for our forecast, from the Metaculus API. Unknown when the API does not give them. A multiple choice outcome is shown by its place in the question's list of options.
- Calculated: every other number, here, from the bot's last forecast before the question closed: as the ledger's reveal line gives it after the close, or as Metaculus reports our forecast when the ledger has not revealed it yet.
- Brier score (yes/no questions): the forecast probability of Yes minus the outcome (1 for Yes, 0 for No), squared. 0 is perfect and 1 the worst.
- Log score: the natural log of the probability the forecast gave to what happened. 0 is perfect; lower is worse. For numeric, discrete and date questions, that probability is the mass the forecast's distribution put in the outcome's bin, or beyond the bound when the outcome fell outside the range.
- Above uniform: the log score minus that of a uniform forecast, which gives 50% to each side of a yes/no question, an equal share to each option, and for numeric, discrete and date questions 5% beyond each open bound (an assumption) with the rest spread evenly over the bins. Below 0 means the forecast did worse than spreading its probability evenly.
- Worst misses: each morning, the three forecasts furthest below 0 above uniform, so that question types compare fairly. Each gets a one-line cause from the bot's cheap helper model, which sees only the resolved question, its outcome, our forecast and the bot's own reasoning when Metaculus has it. A cause that repeats the question's own words is withheld. "Unknown (not reviewed)" means no model reviewed it, or its cause was withheld.
- Assumption: a probability below 0.000001 counts as 0.000001, so that one outcome a forecast all but ruled out cannot make a mean infinite.
- Unknown means a number could not be worked out. It is left out of the means and never counted as zero.

## Calibration so far

No resolved questions reported yet.

Yes/no questions, by our forecast of Yes. Each bin holds forecasts from its lower edge up to, not including, its upper edge; the last one includes 100%.

| Forecast of Yes | Count | Mean forecast | Observed frequency of Yes |
| --- | --- | --- | --- |
| 0% to 10% | 0 | n/a | n/a |
| 10% to 20% | 0 | n/a | n/a |
| 20% to 30% | 0 | n/a | n/a |
| 30% to 40% | 0 | n/a | n/a |
| 40% to 50% | 0 | n/a | n/a |
| 50% to 60% | 0 | n/a | n/a |
| 60% to 70% | 0 | n/a | n/a |
| 70% to 80% | 0 | n/a | n/a |
| 80% to 90% | 0 | n/a | n/a |
| 90% to 100% | 0 | n/a | n/a |

By question type:

| Type | Count | Mean Brier | Mean log score | Mean above uniform |
| --- | --- | --- | --- | --- |
| Yes/no | 0 | n/a | n/a | n/a |
| Multiple choice | 0 | n/a | n/a | n/a |
| Numeric | 0 | n/a | n/a | n/a |
| Discrete | 0 | n/a | n/a | n/a |
| Date | 0 | n/a | n/a | n/a |

## Mornings

Newest first.

### 12 November 2026

Nothing new resolved.

### 11 November 2026

Nothing new resolved.

### 10 November 2026

Nothing new resolved.

### 9 November 2026

Nothing new resolved.

### 8 November 2026

Nothing new resolved.

### 7 November 2026

Nothing new resolved.

### 6 November 2026

Nothing new resolved.

### 5 November 2026

Nothing new resolved.
