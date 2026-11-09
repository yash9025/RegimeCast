# RegimeCast — Implementation Plan

## 1. Research contract

- Universe: SPY, QQQ, TLT, GLD; use a common verified trading calendar.
- Frequency: daily data; forecast issued after the US close with a conservative data availability cutoff. Anything not available by that cutoff enters the next forecast.
- Horizon: next five trading sessions, not an ambiguous calendar week.
- Return: r_t = log(adjusted_close_t / adjusted_close_(t-1)).
- Target variance: v_t = (252 / 5) * sum(r_(t+j)^2, j=1..5). Display volatility as 100 * sqrt(v_t), in annualized percent.
- This daily-return realized variance is a noisy volatility proxy; disclose this limitation.
- Train on log variance with a fixed numerical floor; choose any retransformation calibration using training data only. Mix expert forecasts in variance space, then take the square root for display.
- Prefer a useful negative research result over a misleading performance claim.

## 2. Data and provenance

Start with verified adjusted daily ETF history from 2006 through the last complete session, subject to provider coverage and usage terms. Implement a provider adapter plus local CSV import; verify the provider before selecting it. Cache immutable raw snapshots and processed Parquet files. Do not substitute synthetic observations into research results.

Add 2-year and 10-year Treasury yields, then a small macro set such as inflation and unemployment. FRED/ALFRED access may require a user-provided API key. Store credentials in environment variables, never in source control.

Record provider, instrument, observation date, publication/availability timestamp, vintage, download timestamp, adjustment convention, and checksum. Do not claim adjusted ETF histories are historical vintages unless the provider supplies them; document retrospective adjustment limitations.

Validate unique asset/date keys, sorting, timezones, positive prices, expected sessions, gaps, missingness, stale series, suspicious jumps, and enough observations for each horizon. Do not forward-fill missing ETF prices or targets. Carry macro values forward only after their release, with age and missingness indicators. Never backfill future releases.

For revised macro series, retrieve historical vintages known at the forecast cutoff. If intraday release timing is unavailable, apply a conservative next-session availability rule. If vintage coverage is unavailable, exclude the affected series from the primary experiment and label any exploratory analysis clearly.

Outputs: data manifest, cached snapshots, validation report, reproducible feature table.

## 3. Features

Use only information available by each forecast cutoff:

- Lagged returns and cumulative returns over 1, 5, 10, 21, and 63 sessions.
- Realized volatility over 5, 10, 21, and 63 sessions; EWMA variance; downside variation.
- Drawdowns over 21, 63, and 252 sessions.
- Cross-asset correlations over 21 and 63 sessions.
- Treasury levels, 10Y minus 2Y spread, and 5/21-session changes.
- Released macro changes, release age, and missingness flags.
- Asset identity for pooled models.

Keep price-only and price-plus-macro feature sets explicit. Fit scaling, imputation, and feature selection inside each training fold. Test that modifying future inputs cannot alter earlier features.

## 4. Baselines first

Implement historical variance, EWMA variance, a HAR-style linear baseline, and a pooled gradient-boosted model. The pooled ML model and experts must share features, training eligibility, forecast dates, base learner family, and comparable tuning budgets.

Store every prediction with forecast cutoff, target end date, asset, model, training range, configuration, and data version. Establish the walk-forward pipeline before building regimes.

## 5. Regime estimation

Compare Gaussian mixtures and Gaussian hidden Markov models on a compact shared market-state vector: volatility, drawdown, correlations, and yield changes. Start with three states; test two and four as sensitivity checks.

Fit each estimator and scaler only on the training history for that forecast. HMM forecasts must use forward-filtered probabilities P(state_t | observations through t), never full-sequence smoothing or Viterbi states inferred using later observations.

Generate historical gating probabilities for expert training with an inner chronological walk-forward procedure: each stored probability at date t comes from a regime estimator fitted using data no later than t. Use a warmup period. Log the additional computational cost. Do not retrospectively overwrite archived historical probabilities after refitting.

State numbers are arbitrary. Match states across refits using training-only distribution characteristics and consistent ordering; record assignments. Use neutral state IDs initially. Apply calm/stressed labels only when supported by the state statistics; high uncertainty does not automatically prove a transitional economic state.

Diagnostics: occupancy, effective sample size, persistence, probability entropy, convergence, seed sensitivity, covariance stability, and refit-to-refit changes. Record failed fits and fallback behavior.

## 6. Mixture of experts

Train one expert per state using soft regime probabilities as sample weights. Each expert pools the four assets initially to avoid tiny per-asset state datasets. Use the same boosting model family as the pooled benchmark.

For each asset, predict positive variance with each expert and calculate sum_k p_k(t) * variance_k(t). Define a minimum effective sample size and fall back to the pooled model when an expert or gate is unreliable. Select thresholds using development folds only and log all fallbacks.

Include a pooled-plus-probabilities ablation to distinguish useful regime information from benefits attributable to separate experts. PyTorch is optional after the tabular models establish evidence; it is not required for the first release.

## 7. Time-ordered evaluation

Proposed design, conditional on coverage:

1. Start with five years of training history.
2. Refit monthly and issue daily forecasts using the frozen fitted model and newly available inputs.
3. Use a five-year rolling training window as primary; expanding history as a sensitivity analysis.
4. At every refit include supervised examples only when all five future returns are already observed. Purge label overlap at every inner validation boundary too.
5. Tune with nested time-ordered development folds; keep all four assets for each date in the same fold.
6. Reserve the final two full calendar years as a locked chronological holdout. Freeze design and hyperparameters before evaluating it; scheduled refits may use newly matured labels as they would operationally, without retuning the protocol.
7. Compare all models on identical eligible asset/date rows and report any exclusions.

Primary metric: QLIKE on positive forecast variance, evaluated with a fixed small numerical floor. Also report volatility MAE, volatility RMSE, per-asset results, annual/window results, and worst forecast errors. Define QLIKE as v_actual / v_forecast - log(v_actual / v_forecast) - 1 with documented flooring for zero values.

Estimate uncertainty on paired model loss differences using moving-block or stationary bootstrap over dates, retaining all assets together. Account for overlapping five-session targets and longer dependence; show sensitivity to plausible block lengths. Include a non-overlapping weekly evaluation as a robustness check. Report confidence intervals and win rates across windows; avoid claiming certainty from a single aggregate improvement.

Separate prospective stress classifications based on information available at issuance from retrospective slices based on realized outcomes. Set thresholds from training data and label retrospective analysis explicitly.

## 8. Experiment matrix

Stage experiments to control cost and reduce multiple-comparison fishing:

1. Price-only baselines and pooled ML.
2. Three-state GMM and HMM mixtures versus the same pooled ML.
3. Add point-in-time macro features to both pooled ML and mixtures.
4. Two/four-state sensitivity checks on development data.
5. Pooled-plus-probabilities, expanding-window, seed, and stress-period robustness checks.
6. Select the protocol using development results; run locked holdout evaluation once.

Track configurations, seeds, data hashes, versions, metrics, artifacts, and fit failures with MLflow. Cache causal gating runs by configuration/data version. Explain unequal parameter counts and report runtime as well as accuracy.

## 9. Product

Build FastAPI endpoints for latest forecasts, regime probabilities, historical predictions/errors, model metadata, and data freshness. Serve saved model artifacts; avoid training during API requests.

Use a small Streamlit dashboard initially with four views:

- Overview: annualized volatility forecasts, as-of time, horizon, baseline comparison, stale-data status.
- Regimes: probability timeline, state statistics, uncertainty, refit markers.
- Evaluation: matched model comparisons, date filters, per-asset errors, stress slices, confidence intervals.
- Explanations: expert contributions and training/development-based feature importance, with clear limits on causal interpretation.

Only display prediction intervals if separately calibrated and checked out of sample. Regime probabilities are not forecast confidence intervals. Never display fabricated backtest scores.

## 10. Repository layout

```text
configs/                 assets, features, model settings, fold definitions
src/regimecast/
  data/                  providers, vintages, validation, manifests
  features/              causal transforms and target construction
  models/                baselines, regimes, pooled model, mixture
  evaluation/            purged folds, walk-forward runs, metrics, bootstrap
  api/                   FastAPI application and response schemas
dashboard/               Streamlit views
tests/                   leakage, target, filtering, metric and API checks
data/raw/                immutable cached inputs, ignored by git
data/processed/          versioned derived tables, ignored by git
artifacts/               models and prediction ledgers, ignored by git
reports/                 experiment figures and final research report
notebooks/               exploration consuming reusable package code
```

Use Python, pandas, NumPy, scikit-learn, hmmlearn, PyArrow, MLflow, FastAPI, Streamlit, pytest, and a locked dependency environment. Verify compatible package versions during implementation. Keep reproducible CLI commands for ingest, validate, features, backtest, report, and serve. Provide an explicitly labeled synthetic demo mode for smoke tests only.

## 11. Milestones and acceptance

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Foundation | Package, configuration, data adapters and checks | Four real asset histories validate; provenance recorded |
| 2. Forecasting baseline | Features, labels, baselines and pooled ML | Repeatable walk-forward predictions; target and leakage tests pass |
| 3. Regime research | GMM, filtered HMM, causal gating and experts | Probabilities normalize; no future-dependent historical states; reliable fallbacks |
| 4. Evidence | Ablations, bootstrap and final holdout | Same evaluation rows; uncertainty and failures reported |
| 5. Product | Saved-artifact API and dashboard | Forecasts match artifacts; freshness and errors visible |
| 6. Delivery | Reproduction guide, model/data cards, research report | Clean-install smoke test and complete reproducible run |

A reasonable planning allowance is four to six weeks of focused work, depending on data access and experiment runtime. This is an estimate, not a delivery guarantee.

Completion means a reproducible end-to-end system and an honest answer to the research question. Mixture outperformance is a hypothesis, not a requirement. Record limitations including daily volatility proxy noise, small universe, revised/adjusted history, regime instability, macro availability, and computational cost.

## References

- Time-ordered validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Historical economic vintages: https://fred.stlouisfed.org/docs/api/fred/alfred.html
