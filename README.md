# RegimeCast

**Regime-Aware Volatility Forecasting**

A reproducible market-risk research system for SPY, QQQ, TLT and GLD. It predicts annualized volatility over the next five trading sessions and investigates whether probabilistic market regimes improve forecasting.

## Open the completed local dashboard

This workspace already contains downloaded data, evaluated experiments, fitted models and latest forecast artifacts. You do not need to rerun the backtests to open it.

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open [the dashboard](http://localhost:8501). It includes forecasts, state probabilities, historical errors, uncertainty estimates and expert contributions.

For the separate API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn regimecast.api.app:app --host 127.0.0.1 --port 8000
```

API documentation: [localhost:8000/docs](http://localhost:8000/docs). Requests serve saved artifacts; they never train models.

## Research result

The evaluated regime mixtures did **not** outperform the pooled model on the primary metric. This project reports that negative result rather than promising improvement.

| Model | Development QLIKE, 2013–2023 | Locked holdout QLIKE, 2024–2025 |
|---|---:|---:|
| Pooled boosting | 0.4645 | **0.3694** |
| HAR benchmark | **0.4229** | 0.4096 |
| GMM experts | 0.4949 | 0.4163 |
| HMM experts | 0.4984 | 0.3946 |
| Pooled + macro | 0.4715 | 0.4011 |

Lower QLIKE is better. Values average the four assets equally. See the [research report](docs/RESEARCH_REPORT.md) for uncertainty, failure cases and interpretation.

## What is implemented

- Validated adjusted-price ingestion, XNYS session checks, immutable Parquet snapshots and checksums.
- Historical FRED/ALFRED vintages for yields, unemployment and inflation, joined only after conservative release availability.
- Causal returns, volatility, correlations, drawdowns, yield-curve changes and macro features.
- Historical/EWMA/HAR/pooled baselines, GMM and forward-filtered HMM gates, soft-weighted experts and pooled-probability ablations.
- Monthly rolling evaluation with matured labels, matched forecast rows, a frozen holdout and paired date-block uncertainty estimates.
- Local SQLite-backed MLflow tracking, saved operational models, a read-only FastAPI service and a Streamlit dashboard.

Everything runs locally with free libraries and data access. No paid model API, database or hosting is required. Provider usage terms still apply.

## Fresh checkout

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ".[dev,api,dashboard,research]"
```

Copy `.env.example` to `.env` and enter your free FRED API key locally. Never commit or share keys. Market data, models, MLflow databases and `.env` are intentionally excluded from Git. A fresh clone must prepare its own artifacts; follow [Reproducing the project](docs/REPRODUCING.md).

## Documentation

- [Research report](docs/RESEARCH_REPORT.md)
- [Reproduction and refresh instructions](docs/REPRODUCING.md)
- [Model card](docs/MODEL_CARD.md)
- [Data card](docs/DATA_CARD.md)
- [Implementation plan](IMPLEMENTATION_PLAN.md)
- [Delivery tracker and completed checks](TASKS.md)

The application is intended for local research. Public deployment, continuous ingestion and operational monitoring are separate work. Forecasts are not a trading strategy.

## Streamlit Cloud

Deploy `dashboard/app.py` on branch `main`. The dashboard-specific requirements bypass Poetry and omit model-training libraries. A compact public snapshot supplies real saved forecasts, regime probabilities and research results without keys or startup training. See [cloud deployment instructions](docs/STREAMLIT_CLOUD.md).
