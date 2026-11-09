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
- [ ] Matched evaluations, paired date-block bootstrap, stress slices
- [ ] Point-in-time macro ingestion and ablation
- [ ] Development experiments and locked holdout protocol

## Phase 5 â€” Product
- [ ] Saved-artifact FastAPI and dashboard
- [ ] Explanations, freshness, errors, API smoke tests

## Phase 6 â€” Delivery
- [ ] Model/data cards, reproduction instructions and final report
- [ ] Clean environment verification and final push

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
