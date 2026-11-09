# RegimeCast research report

## Question and design

Can market regimes improve next-five-session volatility forecasts for SPY, QQQ, TLT and GLD? The primary target is annualized variance, `252/5 * sum(next five daily log returns squared)`. Displayed volatility is `100 * sqrt(variance)`.

Price history covers 2006-01-03 through 2026-11-06. After feature warmup, development evaluation uses 2013–2023: 2,768 sessions and 132 monthly refits. The locked holdout uses 2024–2025: 502 sessions. Each refit uses the latest 1,260 eligible training sessions and includes only labels whose full horizon has matured. Forecast issuance is conservatively defined as 16:30 America/New_York.

Models share evaluation rows. Pooled boosting and the experts share price features and learner settings. Historical regime probabilities are generated with past-only monthly fits; HMM inference uses forward filtering. Historical assignments are never replaced with full-history smoothed states. Expert probabilities weight training samples and positive variance forecasts. Unreliable gates or undersupported experts fall back to pooled forecasts.

No hyperparameter search was performed. The holdout dates, three-state specification, seed and learner defaults were frozen in `configs/holdout-protocol.json` before holdout evaluation. Scheduled refits may use newly matured labels without retuning.

## Primary results

| Model | Development mean QLIKE | Holdout mean QLIKE |
|---|---:|---:|
| Pooled boosting | 0.464496 | 0.369364 |
| HAR | 0.422888 | 0.409642 |
| EWMA | 0.443823 | 0.429619 |
| Historical variance | 0.502017 | 0.472851 |
| GMM mixture | 0.494910 | 0.416316 |
| HMM mixture | 0.498372 | 0.394594 |
| Pooled + GMM probabilities | 0.471356 | 0.374119 |
| Pooled + HMM probabilities | 0.483542 | 0.376411 |
| Pooled + macro | 0.471474 | 0.401122 |

Lower is better. HAR leads on aggregate development QLIKE; pooled boosting leads on the locked holdout. Neither separate experts nor probability features establish an improvement over pooled boosting. Macroeconomic features do not establish an overall improvement either, although some per-asset secondary errors improve.

## Time-dependent uncertainty

Paired loss differences are averaged across assets within each date. Circular moving-block bootstrap resamples dates, keeping assets together; 2,000 replicates are reported for 5-, 21- and 63-session blocks. Every-fifth-session evaluation also checks non-overlapping five-session targets.

For 21-session blocks, the holdout differences versus pooled boosting are:

| Candidate | Mean QLIKE difference | Marginal 95% interval |
|---|---:|---:|
| GMM mixture | +0.04695 | [-0.00722, +0.13511] |
| HMM mixture | +0.02523 | [+0.00489, +0.04552] |
| Pooled + macro | +0.03176 | [-0.00181, +0.07026] |

Negative differences would favor the candidate. These are exploratory, unadjusted intervals; no multiple-comparison-adjusted superiority claim is made. The development mixture differences also favor pooled boosting. Complete block-length sensitivity is in the CSV reports.

## Regime-count sensitivity

The predeclared diagnostic compares two, three and four states during **2020 only**, covering 253 sessions. GMM mixture QLIKE is 1.3185, 1.2662 and 1.3213 respectively. HMM mixture QLIKE is 1.3346, 1.2908 and 1.6152. Three states perform best within this limited diagnostic; four HMM states are particularly unstable in forecast error. This does not establish the optimal state count across the full sample. The holdout state count was not changed.

## Failure cases and stress analysis

Prospective stress flags use a trailing-market-variance threshold estimated from the corresponding training window. Separate reports show errors in stressed and ordinary conditions. The holdout has only 25 stress dates under this definition, so those slices are small.

Large errors include the April 2013 gold shock and the early-February 2018 equity volatility event. For example, the GMM mixture issued a GLD forecast on 2013-04-09 whose next-five-session realized variance was much larger than forecast; its QLIKE loss was about 60.07. The HMM mixture also sharply underpredicted the SPY horizon issued on 2018-01-29, with QLIKE about 54.55. Neither issuance date was already classified as stressed by the prospective threshold. Recognizing an existing regime therefore does not reliably anticipate a new shock.

## Limits

Daily squared returns are a noisy realized-volatility proxy. Five-session targets overlap, the asset universe is small, and inferred states are not independently observed economic labels. Adjusted Yahoo histories are retrospective, not historical price vintages. Macro availability is conservative: the first trading session strictly after each vintage date. Results depend on this data snapshot and fixed configuration.

Seed sensitivity, expanding-window alternatives, broader universes, intraday targets and large tuning searches were not completed. They remain extensions, not validated claims. No profitable trading strategy is asserted.

## Evidence files

- `reports/phase1-data-validation.json`: price provenance and validation.
- `reports/phase2-baseline-validation.json`: baseline reproduction checks.
- `reports/phase3-regime-validation.json`: full regime experiments and per-asset scores.
- `reports/macro-analysis/`: development uncertainty, annual errors, stress slices and worst errors.
- `reports/holdout-analysis/`: corresponding locked-holdout results.
- `reports/regime-count-sensitivity.csv`: the limited 2020 diagnostic.

The conclusion is a reproducible negative result: the evaluated regime mixtures do not improve the primary forecast loss over the pooled benchmark.
