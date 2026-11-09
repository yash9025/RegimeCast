# Model card

**Purpose:** local research into next-five-session market volatility for SPY, QQQ, TLT and GLD.

**Target and units:** annualized daily-return realized variance; annualized volatility percent is derived for display. State probabilities are not prediction confidence intervals.

**Models:** historical variance, EWMA, log-HAR, pooled histogram gradient boosting, GMM/HMM soft-weighted experts, pooled probability-feature ablations, and a pooled macro-feature model. Boosting uses 100 iterations, 15 maximum leaf nodes, learning rate 0.06, L2 penalty 1.0, disabled random early stopping and seed 42. Training-only smearing calibrates log-variance retransformation.

**Regimes:** three states in primary experiments, monthly historical fits, training-only scaling and state alignment. HMM estimates use forward filtering. State IDs remain neutral because statistical clusters are not economic ground truth.

**Operational preparation:** monthly refit convention, only matured training labels, saved model bundles and atomic catalog publication. Web requests read artifacts and never fit models. Current prepared forecasts are based on the last completed session in the cached price snapshot; inspect their as-of date before use.

**Evaluation:** matched rolling development and frozen chronological holdout. QLIKE is primary, with volatility MAE/RMSE, period results, failure cases and paired date-block uncertainty. The evaluated mixtures underperform the pooled holdout benchmark on mean QLIKE.

**Fallbacks:** undersupported experts use pooled predictions; unreliable current gates trigger pooled-model inference. Fit diagnostics and effective sample sizes are retained.

**Validation completed:** full 36-test suite at the earlier milestone, plus focused checks for later cache, API, dashboard, operational bundle, tracking and data-version safeguards. The final aggregate suite and an additional clean-environment run were not repeated because the user requested immediate completion and no further extended testing. CI configuration is included; no claim of a verified final CI result is made here.

**Known limitations:** retrospective adjusted prices, noisy daily targets, small universe, unstable latent states and limited stress samples. Seed and expanding-window robustness remain future research. Upstream calendar/NumPy deprecation warnings were observed in the pinned environment.

**Deployment boundary:** local research application with pinned dependencies, automated checks and explicit artifact provenance. Public hosting, authentication, scheduling, monitoring, service objectives and live risk-governance approval are outside this release. This is not a certified production risk engine or an investment recommendation.
