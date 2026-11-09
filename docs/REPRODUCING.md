# Reproduction and refresh guide

## Existing prepared workspace

Open the dashboard without retraining:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Run the API separately if needed:

```powershell
.\.venv\Scripts\python.exe -m uvicorn regimecast.api.app:app --host 127.0.0.1 --port 8000
```

Both use `artifacts/catalog.json`. `REGIMECAST_CATALOG` can point to a different local catalog.

## Fresh checkout

Install the pinned environment as shown in README. Add your free FRED key to `.env`. Run commands from the repository root. Replace the example snapshot/run placeholders below with each command's final printed directory.

```powershell
$projectPython = ".\.venv\Scripts\python.exe"
& $projectPython -m regimecast.cli ingest --provider yahoo --start 2006-01-01 --end 2026-11-09
& $projectPython -m regimecast.cli ingest-macro --start 2005-01-01 --end 2026-11-09
```

The price end date is exclusive. These explicit dates reproduce the release scope; provider revisions may prevent identical numbers. Exact numerical reproduction requires the same immutable snapshots and checksums recorded in the released manifests.

Assign your generated paths:

```powershell
$priceSnapshot = "data/raw/YOUR_PRICE_SNAPSHOT"
$macroSnapshot = "data/macro/YOUR_MACRO_SNAPSHOT"
& $projectPython -m regimecast.cli validate $priceSnapshot
& $projectPython -m regimecast.cli backtest $priceSnapshot --gate gmm
& $projectPython -m regimecast.cli backtest $priceSnapshot --gate hmm
& $projectPython -m regimecast.cli backtest $priceSnapshot --macro-snapshot $macroSnapshot
```

These are full research runs and can take substantial time on a laptop. Store the printed paths as `$gmmRun`, `$hmmRun` and `$macroRun`. Do not rerun them simply to open the dashboard.

```powershell
& $projectPython -m regimecast.cli analyze $gmmRun $hmmRun $macroRun --output artifacts/reproduced-analysis/development
```

Analysis outputs must be new directories. The local analysis registry ties them to exact data versions; mismatched released reports are not attached to refreshed forecasts.

## Frozen holdout

Read `configs/holdout-protocol.json` first. Do not change model settings based on holdout results.

```powershell
& $projectPython -m regimecast.cli backtest $priceSnapshot --start 2024-01-01 --end 2026-01-01 --holdout --gate gmm --gate-cache $gmmRun
& $projectPython -m regimecast.cli backtest $priceSnapshot --start 2024-01-01 --end 2026-01-01 --holdout --gate hmm --gate-cache $hmmRun
& $projectPython -m regimecast.cli backtest $priceSnapshot --start 2024-01-01 --end 2026-01-01 --holdout --macro-snapshot $macroSnapshot
```

Analyze those three printed holdout paths together into `artifacts/reproduced-analysis/holdout`. Two/four-state diagnostics use `--start 2020-01-01 --end 2021-01-01 --gate gmm` or `hmm` with `--states 2` or `4`; compare only with the 2020 subset of the three-state runs.

## Prepare latest forecasts

```powershell
& $projectPython -m regimecast.cli forecast $priceSnapshot --macro-snapshot $macroSnapshot --gmm-cache $gmmRun --hmm-cache $hmmRun
```

This prepares fitted models, expert explanations and the dashboard catalog offline. Gate caches require matching price snapshots and model settings. If you download a changed price snapshot, omit the cache arguments to rebuild causal probabilities. Choose an updated exclusive price end date that includes only complete sessions; macro refresh defaults use the current UTC date.

## Local tracking

```powershell
& $projectPython -m regimecast.cli track $gmmRun $hmmRun $macroRun
```

Tracking stores configurations, metrics and metadata in a local SQLite-backed MLflow store under `artifacts/mlflow/`. Registration is idempotent and does not retrain. No paid or remote tracking service is used.

## Validation status

The repository contains data, leakage, filtering, forecasting, API, dashboard and tracking checks. Previously completed checks are recorded in TASKS.md. At the user's request, extended testing stopped at final handoff; a final aggregate suite and separate clean-install verification were not rerun. Raw data and model artifacts are intentionally absent from Git, so a fresh clone cannot show forecasts until artifacts are prepared.
