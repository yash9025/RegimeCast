# RegimeCast

**Regime-Aware Volatility Forecasting**

Does recognizing market regimes improve next-five-session volatility forecasts over a strong pooled model? RegimeCast investigates this question for SPY, QQQ, TLT and GLD with time-ordered evaluation.

## Status

Phase 1 implementation in progress. No forecasting or performance claims yet. See [TASKS.md](TASKS.md) for verified progress and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the research protocol.

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
