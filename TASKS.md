# RegimeCast delivery tracker

Each increment is tested before committing and pushing. Checkboxes describe verified delivery, not intended work. Synthetic data is restricted to tests and explicitly labeled demos.

## Phase 1 â€” Foundation
- [x] 1A Package, isolated environment, configuration and CI
- [x] 1B CSV/Yahoo adapters, trading-session validation, immutable Parquet snapshots and manifests
- [ ] 1C Four-asset real-data ingestion and validation

## Phase 2 â€” Baselines
- [ ] Causal features and exact five-session labels
- [ ] Historical/EWMA/HAR/pooled ML models
- [ ] Purged walk-forward prediction ledger and leakage tests

## Phase 3 â€” Regimes and experts
- [ ] GMM and forward-filtered HMM
- [ ] Causal historical probabilities and stable state alignment
- [ ] Soft weighted experts, pooled-plus-probabilities and fallbacks

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
