# RegimeCast

**Regime-Aware Volatility Forecasting**

Does recognizing market regimes improve next-five-session volatility forecasts over a strong pooled model? RegimeCast investigates this question for SPY, QQQ, TLT and GLD with time-ordered evaluation.

## Status

Phase 1 complete: 17 tests pass and all four real asset histories validate (20,980 observations). No forecasting or performance claims yet. See [TASKS.md](TASKS.md) for verified progress and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the research protocol.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check src tests
```

## Ingest daily adjusted prices

```powershell
.\.venv\Scripts\regimecast ingest --provider yahoo --start 2006-01-01 --end 2026-11-09
.\.venv\Scripts\regimecast validate data/raw/REPLACE_WITH_SNAPSHOT_DIRECTORY
```

The end date is exclusive. Choose an end date that excludes incomplete trading sessions. Yahoo access is an unofficial research adapter; verify the underlying provider terms before use. Raw market data is not committed or redistributed. API behavior: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html

For your own licensed history, import a CSV with exactly these required columns:

```text
date,asset,adjusted_close
2024-01-02,SPY,465.25
```

Supply every requested asset for every XNYS session in the requested range, in ascending date order within each asset. The example above illustrates the schema only and is not a complete dataset.

```powershell
.\.venv\Scripts\regimecast ingest --provider csv --csv prices.csv --adjustment "Provider split/dividend adjusted close" --start 2006-01-01 --end 2026-11-09
```

The pipeline fails on missing prices, duplicates, nonpositive values, unexpected sessions, timezone-bearing session labels, or missing boundary observations. Suspicious jumps and unchanged-price streaks are reported for review. It never fills missing prices. Snapshots contain normalized provider observations, a SHA-256 checksum, validation details and download time; local CSV input also gets a source checksum.

Retrospectively adjusted prices are **not** point-in-time vintages. Publication timestamps cannot be reconstructed from current Yahoo history. Revised macro data will be excluded from the primary experiment until historical vintages are available. Keep secrets in environment variables and never add tokens to source or repository URLs.

## Forecast definition

For daily log returns r, the variance target at session t is `252 / 5 * sum(r[t+j]**2 for j in 1..5)`. The dashboard will display `100 * sqrt(variance)` as annualized volatility percent. This daily-return proxy is noisy; it is not an intraday realized-volatility estimator.

## Development baselines

```powershell
.\.venv\Scripts\regimecast backtest data/raw/REPLACE_WITH_SNAPSHOT_DIRECTORY --start 2013-01-01 --end 2024-01-01
```

The command writes an immutable run directory with predictions, features, per-asset metrics and configuration/provenance. Models are historical variance, EWMA, log-HAR and pooled histogram gradient boosting. Features use lagged/current close information only. Forecast issuance is conservatively defined as 16:30 America/New_York on each session. Labels must have fully matured by each monthly refit; training uses the most recent 1,260 eligible sessions. The initial feature warmup is excluded.

No hyperparameters have been tuned. The fixed development defaults leave 2024 onward locked, including the two full calendar years 2024–2025 and the subsequent partial year. Do not select models using that period. Volatility MAE/RMSE are in annualized percentage points; QLIKE uses annualized variance.

## Regime experiments

```powershell
.\.venv\Scripts\regimecast backtest data/raw/REPLACE_WITH_SNAPSHOT_DIRECTORY --gate gmm --states 3
.\.venv\Scripts\regimecast backtest data/raw/REPLACE_WITH_SNAPSHOT_DIRECTORY --gate hmm --states 3
```

Each experiment regenerates a causal historical probability archive using past-only monthly gate refits. State features are aggregate volatility, drawdown, correlation and returns. HMM probabilities use forward filtering. States align to prior training centers using minimum-distance assignment; state IDs are descriptive, not independently observed economic labels. Archives and fit diagnostics accompany each run.

Experts use soft probability sample weights and share the pooled learner family. The mixture combines variance forecasts, and falls back to pooled predictions for unreliable current gates or undersupported experts. Nonconvergence is reported, not suppressed. The extra pooled-plus-probabilities model tests whether gates help without separate experts. Results still require matched, time-dependent uncertainty analysis.

## Free historical macro data and evidence

Create a free FRED API key and store it locally as `FRED_API_KEY=...` in `.env`. The file is ignored by Git. Never paste keys into chat or commit them. Alternatively set the environment variable. All computation and storage are local; no paid services are required.

```powershell
.\.venv\Scripts\regimecast ingest-macro --start 2005-01-01 --end 2026-11-09
.\.venv\Scripts\regimecast backtest data/raw/PRICE_SNAPSHOT --macro-snapshot data/macro/MACRO_SNAPSHOT
.\.venv\Scripts\regimecast analyze artifacts/GMM_RUN artifacts/HMM_RUN --output reports/MY_NEW_ANALYSIS
```

The macro adapter requests sparse new/revised observations in four-year batches, preserving the release dates encoded by FRED. Values become usable on the first trading session strictly after the vintage date. A revision to an older month never displaces a newer observation. Missing macro values stay missing with indicators; boosting handles them natively. Regime gate features remain price-only for the initial controlled macro ablation.

Analysis uses paired date-level QLIKE differences with all assets retained together in circular moving blocks (5/21/63 sessions). Reports include annual results, training-defined stress slices, every-fifth-session non-overlapping targets and worst errors. Confidence intervals are exploratory and unadjusted for multiple comparisons. Existing full-period development evidence is in [reports/development-analysis](reports/development-analysis).
