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

From the 19 resolved questions reported so far.

Yes/no questions, by our forecast of Yes. Each bin holds forecasts from its lower edge up to, not including, its upper edge; the last one includes 100%.

| Forecast of Yes | Count | Mean forecast | Observed frequency of Yes |
| --- | --- | --- | --- |
| 0% to 10% | 0 | n/a | n/a |
| 10% to 20% | 1 | 17% | 0% |
| 20% to 30% | 1 | 24% | 0% |
| 30% to 40% | 1 | 31% | 0% |
| 40% to 50% | 1 | 40% | 0% |
| 50% to 60% | 2 | 58.5% | 100% |
| 60% to 70% | 0 | n/a | n/a |
| 70% to 80% | 1 | 71% | 100% |
| 80% to 90% | 0 | n/a | n/a |
| 90% to 100% | 1 | 90% | 100% |

By question type:

| Type | Count | Mean Brier | Mean log score | Mean above uniform |
| --- | --- | --- | --- | --- |
| Yes/no | 8 | 0.098 | -0.358 | 0.335 |
| Multiple choice | 1 | n/a | -0.821 | 0.788 |
| Numeric | 4 | n/a | -5.162 | 0.242 |
| Discrete | 6 | n/a | -4.214 | -0.308 |
| Date | 0 | n/a | n/a | n/a |

## Mornings

Newest first.

### 24 November 2026

4 newly resolved questions.

| Question | Type | Outcome | Our forecast | Brier | Log score | Above uniform | Metaculus baseline | Metaculus peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Question 46208](https://www.metaculus.com/questions/46008/) | Numeric | -7.421 | 0.139% in the outcome's bin | n/a | -6.578 | -1.175 | -44.2 | -8.5 |
| [Question 46202](https://www.metaculus.com/questions/46007/) | Numeric | -1.23 | 1.1% in the outcome's bin | n/a | -4.556 | 0.848 | 33.2 | 6.0 |
| [Question 46568](https://www.metaculus.com/questions/46335/) | Yes/no | Yes | 59% Yes | 0.168 | -0.528 | 0.166 | 22.2 | -2.8 |
| [Question 46585](https://www.metaculus.com/questions/46352/) | Yes/no | Yes | 90% Yes | 0.010 | -0.105 | 0.588 | 79.4 | 26.7 |

Worst misses:

1. [Question 46208](https://www.metaculus.com/questions/46008/): 0.139% in the outcome's bin, outcome -7.421, log score -6.578. Cause: The forecast underestimated the chance and magnitude of a sharp relative decline in gold versus equities, leaving the realized loss beyond its lower-tail range.

### 23 November 2026

Nothing new resolved.

### 22 November 2026

Nothing new resolved.

### 21 November 2026

Nothing new resolved.

### 20 November 2026

3 newly resolved questions.

| Question | Type | Outcome | Our forecast | Brier | Log score | Above uniform | Metaculus baseline | Metaculus peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Question 46661](https://www.metaculus.com/questions/46428/) | Yes/no | Yes | 58% Yes | 0.176 | -0.545 | 0.148 | 11.8 | -6.8 |
| [Question 46707](https://www.metaculus.com/questions/46450/) | Multiple choice | option 2 of 5 | 44% on the outcome | n/a | -0.821 | 0.788 | 45.9 | 13.1 |
| [Question 46726](https://www.metaculus.com/questions/46470/) | Yes/no | No | 31% Yes | 0.096 | -0.371 | 0.322 | 12.6 | 12.7 |

No misses: every forecast did at least as well as a uniform one.

### 19 November 2026

Nothing new resolved.

### 18 November 2026

5 newly resolved questions.

| Question | Type | Outcome | Our forecast | Brier | Log score | Above uniform | Metaculus baseline | Metaculus peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Question 46248](https://www.metaculus.com/questions/46016/) | Discrete | 130.0 | 4.9% in the outcome's bin | n/a | -3.011 | -0.015 | 1.7 | 1.2 |
| [Question 46249](https://www.metaculus.com/questions/46016/) | Discrete | 71.5 | 9.3% in the outcome's bin | n/a | -2.372 | 1.447 | 9.8 | 5.9 |
| [Question 46250](https://www.metaculus.com/questions/46016/) | Discrete | 11.0 | 0.942% in the outcome's bin | n/a | -4.665 | -1.669 | -25.1 | 2.0 |
| [Question 46279](https://www.metaculus.com/questions/46006/) | Numeric | 115844000000.0 | 0.451% in the outcome's bin | n/a | -5.402 | 0.002 | -7.4 | 7.3 |
| [Question 46199](https://www.metaculus.com/questions/45995/) | Discrete | 2.7 | 0.736% in the outcome's bin | n/a | -4.911 | 0.493 | 16.7 | 3.4 |

Worst misses:

1. [Question 46250](https://www.metaculus.com/questions/46016/): 0.942% in the outcome's bin, outcome 11.0, log score -4.665. Cause: The forecast underestimated continued spending growth, placing its upper range below the higher-than-expected guidance.
2. [Question 46248](https://www.metaculus.com/questions/46016/): 4.9% in the outcome's bin, outcome 130.0, log score -3.011. Cause: Unknown (not reviewed)

1 of the causes were withheld because they repeated the question's own words.

### 17 November 2026

Nothing new resolved.

### 16 November 2026

Nothing new resolved.

### 15 November 2026

7 newly resolved questions.

| Question | Type | Outcome | Our forecast | Brier | Log score | Above uniform | Metaculus baseline | Metaculus peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Question 46654](https://www.metaculus.com/questions/46421/) | Discrete | 1.14 | 5% in the outcome's bin | n/a | -3.001 | 0.149 | 2.4 | 5.6 |
| [Question 46655](https://www.metaculus.com/questions/46422/) | Numeric | -9.21 | 1.6% in the outcome's bin | n/a | -4.112 | 1.291 | 62.6 | 45.1 |
| [Question 46656](https://www.metaculus.com/questions/46423/) | Yes/no | No | 40% Yes | 0.160 | -0.511 | 0.182 | 24.3 | -1.0 |
| [Question 46657](https://www.metaculus.com/questions/46424/) | Discrete | 73.0 | 0.066% in the outcome's bin | n/a | -7.323 | -2.254 | -108.1 | -50.6 |
| [Question 46658](https://www.metaculus.com/questions/46425/) | Yes/no | No | 24% Yes | 0.058 | -0.274 | 0.419 | 53.9 | -3.9 |
| [Question 46659](https://www.metaculus.com/questions/46426/) | Yes/no | Yes | 71% Yes | 0.084 | -0.342 | 0.351 | 50.0 | 7.9 |
| [Question 46660](https://www.metaculus.com/questions/46427/) | Yes/no | No | 17% Yes | 0.029 | -0.186 | 0.507 | 69.4 | 3.9 |

Worst misses:

1. [Question 46657](https://www.metaculus.com/questions/46424/): 0.066% in the outcome's bin, outcome 73.0, log score -7.323. Cause: It treated a sharp price squeeze as a remote tail, underestimating how quickly disease driven supply losses and holiday demand could lift prices.

### 14 November 2026

Nothing new resolved.

### 13 November 2026

Nothing new resolved.

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
