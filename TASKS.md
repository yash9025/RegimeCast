# RegimeCast delivery tracker

Each increment is tested before committing and pushing. Checkboxes describe verified delivery, not intended work. Synthetic data is restricted to tests and explicitly labeled demos.

## Phase 1 â€” Foundation
- [x] 1A Package, isolated environment, configuration and CI
- [x] 1B CSV/Yahoo adapters, trading-session validation, immutable Parquet snapshots and manifests
- [x] 1C Four-asset real-data ingestion and validation

## Phase 2 â€” Baselines
- [x] Causal features and exact five-session labels
- [x] Historical/EWMA/HAR/pooled ML models
- [x] Purged walk-forward prediction ledger and leakage tests

## Phase 3 â€” Regimes and experts
- [x] GMM and forward-filtered HMM
- [x] Causal historical probabilities and stable state alignment
- [x] Soft weighted experts, pooled-plus-probabilities and fallbacks

## Phase 4 â€” Evidence
- [x] Matched evaluations, paired date-block bootstrap, stress slices
- [x] Point-in-time macro ingestion and pooled-model ablation
- [x] Development experiments, 2020 regime-count sensitivity, and frozen 2024–2025 holdout

## Phase 5 â€” Product
- [x] Saved-artifact FastAPI and dashboard
- [x] Explanations, freshness, errors, API checks and dashboard rendering

## Phase 6 â€” Delivery
- [x] Model/data cards, reproduction instructions and final report
- [x] Final delivery push
- [ ] Additional clean-environment and final aggregate verification (deferred at explicit user request)

## Execution notes
- Folder renamed by user to RegimeCast.
- Remote initially empty; no existing code to merge.
- Exposed chat token is never used or persisted. Repository operations use configured Git credentials.

### Phase 1A/1B verification
- 16 tests passed; lint, CLI help, and pip compatibility checks passed.
- Upstream NumPy/calendar deprecation warnings remain; no warnings are suppressed.

### Phase 1 completion
- 17 tests pass; lint, CLI smoke checks and dependency checks pass.
- Fixed a holiday/weekend boundary bug found during real ingestion.
- Yahoo retry succeeded: 20,980 observations, four assets, 2006-01-03 through 2026-11-06.
- Checksum reload and complete session coverage verified with no validation warnings.
- Data remains local and ignored by Git; its manifest summary is in reports/phase1-data-validation.json.

### Phase 2A implementation
- 22 tests pass, including future-price invariance, exact target checks, label maturity, matched rows, and holdout protection.
- Features, four baselines, monthly rolling refits and immutable run artifacts implemented.
- Full real-data development integration run is in progress; Phase 2 acceptance awaits its verification.
- Forecast cutoff fixed to 16:30 America/New_York; 2024 onward remains untouched by evaluation.

### Phase 2 completion
- Full development integration verified: 44,288 predictions, 2,768 sessions, 132 monthly refits, 2013-01-02 through 2023-12-29.
- Clean committed rerun produced identical forecast values; forecast timestamps and label-maturity assertions pass.
- 22 unit/integration tests pass. Aggregate metrics are in reports/phase2-baseline-validation.json.
- HAR beats pooled boosting on QLIKE for every asset; no mixture outperformance is assumed.

### Phase 3A implementation
- GMM and HMM gates use monthly past-only fits, training-only scaling and state matching.
- HMM inference uses an explicit forward recursion; no smoothing or Viterbi historical assignments.
- Historical probabilities are generated once per historical cutoff; later observations cannot rewrite them.
- 25 tests pass, including hand-checked filter updates, incremental filtering and gate future-invariance for both estimators.
- Convergence failures and numerical failures are recorded; unreliable gates will trigger pooled-model inference fallback.

### Phase 3B implementation
- Soft-weighted boosting experts mix positive variance predictions; pooled models share the same training rows and features.
- Added pooled-plus-probabilities ablations, expert effective sample-size diagnostics and unreliable-gate fallback.
- 26 tests pass, including variance-space convexity, invalid probability rejection and pooled fallbacks.
- Phase 3 real-data acceptance remains pending the GMM/HMM integration experiments.

### Phase 3 completion
- Both full 2013–2023 real-data experiments completed: 66,432 forecasts each, 132 monthly refits, three regimes.
- All probability sums, historical fit cutoffs and label-maturity checks pass.
- Baseline predictions exactly match the Phase 2 run in both experiments.
- Validation and per-asset errors are in reports/phase3-regime-validation.json.
- Forecasting source stayed unchanged during runs; untracked macro/analysis additions explain dirty-tree provenance.
- Full primary development evaluation is complete; regime-count sensitivity and uncertainty remain Phase 4 work.

### Phase 4A evidence tools
- Added matched-ledger integrity checks and circular date-block bootstrap (5/21/63-session blocks; 2,000 replicates).
- Added annual results, non-overlapping forecast evaluation, worst errors and training-defined prospective stress slices.
- Development intervals are exploratory and unadjusted for multiple comparisons; they are not evidence of profitable trading.
- Free FRED key verified locally; sparse vintage downloader batches four-year release periods to respect API limits.
- Macro release timing and revision parser tests pass. Real macro ingestion remains running.

### Phase 4A verified increment
- 35 tests passed across the full 34-test suite and the additional end-to-end macro test.
- Real FRED/ALFRED snapshot verified: 13,013 vintage records across DGS2, DGS10, UNRATE and CPIAUCSL.
- Release-aware features include levels, changes, missingness, age, inflation year-over-year and yield-curve changes.
- Key remains only in ignored .env; provider errors do not log request URLs.
- Evidence tools and development summaries are complete. Macro ablation, regime sensitivity and locked holdout remain pending.
- GitHub authentication must be restored in the execution environment before queued commits can be pushed.

### Phase 4B macro results and frozen protocol
- Macro features evaluated on the same 2,768 sessions; unchanged baselines pass matched-run checks.
- Overall macro QLIKE difference versus pooled price-only is +0.006979; 21-session paired CI includes zero.
- Protocol frozen in configs/holdout-protocol.json before evaluating 2024–2025. Three states and fixed defaults retained.
- Regime-count diagnostic is limited to 2020 and will be labeled accordingly; it does not choose the holdout model.
- Provenance now captures the source commit, dirty state and source digest at the beginning of each run.
- 36 tests, lint and dependency compatibility checks pass. GitHub device sign-in restored; queued commits pushed.

### Phase 4C runtime improvement
- Added month-boundary continuation of causal gate archives, with data/configuration checks and cache checksums.
- Both GMM and HMM resumed probabilities exactly match full regeneration in tests.
- End-to-end cached backtest preserves all historical probability rows.
- This changes runtime and provenance only; learner defaults and frozen holdout settings stay fixed.

### Phase 4 completion
- Frozen 2024–2025 holdout evaluated once for GMM, HMM and pooled-macro variants; settings unchanged.
- Matched model rows, training-label maturity and identical protocol checksums verified.
- Holdout metrics, paired block intervals, annual/stress results and worst errors are in reports/holdout-analysis.
- Two/three/four-state diagnostics completed for 2020 only, explicitly labeled in reports/regime-count-sensitivity.csv.
- Price and macro hypotheses are reported without requiring outperformance or claiming trading profitability.

### Phase 5A — read-only API
- Forecast, regime, history, metrics, explanation, metadata and freshness endpoints implemented.
- API tests pass for filtering, limits, missing/corrupt artifacts and completed-session freshness.
- Requests serve saved results and never train models.
- Corrected lock generation to use the isolated project environment and include API dependencies in CI.

### Phase 5B — dashboard
- Four views implemented: forecasts, market states, research evidence and explanations.
- Displays forecast as-of dates, stale-data status, uncertainty limitations and explicit synthetic demo labels.
- Streamlit render test passes with four tabs and four forecast cards, using saved test artifacts only.
- Dashboard shipped independently of offline model preparation.

### Phase 5C — offline forecast preparation
- Added monthly-refit operational model preparation with fully matured labels and five-session horizons.
- Saves fitted models, regime archives, expert contributions, feature context, and an atomic artifact catalog.
- End-to-end test validates saved models, contribution sums, label maturity and explicit synthetic-data labeling.
- Live real-data preparation follows this separate tested code commit.

### Phase 6A — experiment tracking
- Added idempotent MLflow registration for existing run artifacts, without retraining.
- Tracking uses a local SQLite database and local artifacts; telemetry is disabled.
- Test verifies completed runs, persisted metrics and duplicate-registration prevention.
- All scientific, API, dashboard and research dependency versions are pinned for CI.

### Refresh consistency safeguards
- Dashboard analyses are registered against exact price/macro data versions; mismatched reports are not attached.
- Reproduced analyses register locally without overwriting released research reports.
- Macro refresh defaults now use the current UTC date; historical experiments still use explicit frozen cutoffs.
- Focused tests cover stale-version rejection and the refresh-date default.

### Final handoff
- Real operational forecasts, fitted models, expert contributions and dashboard catalog prepared locally.
- Six primary development/holdout runs registered in local SQLite-backed MLflow.
- README, research report, model/data cards and reproduction guide completed.
- User requested immediate completion without further extended tests. No final aggregate suite or additional clean-environment run was performed; earlier passing checks are retained above.
- Seed/expanding-window studies and public deployment remain documented extensions.
