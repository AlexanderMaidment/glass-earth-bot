# Coverage alert: a daily check, on GitHub, that no question goes without our forecast

Built Wed 25 Nov 2026 (Brisbane, UTC+10) on the branch `coverage-alert`. **Not merged, not pushed, not switched
on.** Labels: (m) measured, (c) calculated, (e) estimated, (rule) Alex's words.

## What it does

Once a day `.github/workflows/coverage_alert.yaml` runs `scripts/coverage_alert.py` on GitHub Actions (not on the
PC). It reads the bot's tournaments, the same ones as `main.py`'s tournament mode (FutureEval 33121, MiniBench slug
`minibench`, Market Pulse 26Q4 while `GLASS_EARTH_MARKET_PULSE` is true), every page of each listing, and names:

1. every open question that still has no forecast from us 24 hours after it opened (rule: Alex's brief);
2. every question that closed in the last 36 hours with no forecast from us.

Rule 2 is there because FutureEval and MiniBench questions are open about 3 hours (m), so rule 1 alone never fires
on them. Replayed on real Metaculus data (m, 25 Nov): at 15 Nov 10:00 UTC it names FutureEval question 46730, and at
17 Nov 21:30 UTC it names 26 MiniBench round 2 questions; rule 1 names none in either. The 36 hours are a day plus
room for GitHub starting a daily schedule hours late (4 h 44 m to 8 h 31 m in this repository, m, 2 to 5 Nov, `LEARNING_LOOP_DIAGNOSIS.md`), so
a miss can show on two mornings in a row.

"Ours" means Metaculus lists a forecast of ours on the question (the bot's own already-forecast test) or
`forecasts/ledger.jsonl` has a line for it. Questions are named by ID and link only, never by title or text.

**How it alerts: the bot's existing path** (`credit.py`): one GitHub Actions error line per question, the same lines
on the run's summary page, and a failing exit code, so GitHub emails the owner. A check that cannot read Metaculus
(no token, 403, 429, 5xx, a page that is not JSON) fails the same way, so a broken alert is seen too.

It only GETs from Metaculus with `METACULUS_TOKEN`, 4 s apart, never retried. No model call, no spend, no commit
(`contents: read`), not in the `glass-earth-model-spend` group (a run queued there cancels an older waiting run, which
could be a forecasting run). Its tests are `tests/coverage_alert_checks.py`, which the forecasting workflows never run
(they take only `test*.py`), so nothing here can stop a forecast.

## Tests

```powershell
uv run --locked python -m unittest discover -s tests -p coverage_alert_checks.py
```

18 checks (m, 25 Nov). The default suite (`discover -s tests`) still runs 265, the same as `main` (m).

## Steps to switch it on, and the words each needs

1. **"merge and push the coverage alert"**: merge `coverage-alert` into the bot's `main` and push. Nothing runs yet:
   the job is skipped until the variable in step 2 says `true` (GitHub shows a skipped run each day).
2. **"switch on the coverage alert"**: set the variable, then start one run by hand and read it:

   ```powershell
   gh variable set GLASS_EARTH_COVERAGE_ALERT --body true --repo AlexanderMaidment/glass-earth-bot
   ```

   ```powershell
   gh workflow run coverage_alert.yaml --repo AlexanderMaidment/glass-earth-bot --ref main
   ```

   A clean run ends "Coverage OK" with exit 0; a gap ends red with the questions named.
3. Email: GitHub sends a failed scheduled run to the account that last changed the workflow's `cron` line, so the
   push in step 1 should be from Alex's account. Check GitHub, Settings, Notifications, Actions: email on, "Only
   notify for failed workflows" ticked.

To stop it: `gh variable set GLASS_EARTH_COVERAGE_ALERT --body false --repo AlexanderMaidment/glass-earth-bot`.

## Cost

US$0 (c): no model calls; about 6 Metaculus GETs a day (two listings per tournament, more past 100 posts); GitHub
Actions minutes are free on a public repository (e).
