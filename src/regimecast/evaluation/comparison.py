"""Paired date-block uncertainty and matched research summaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regimecast.evaluation.metrics import qlike, summarize


def merge_ledgers(paths):
    frames = [pd.read_parquet(Path(path) / "predictions.parquet") for path in paths]
    ledger = pd.concat(frames, ignore_index=True)
    keys = ["date", "asset", "model"]
    duplicates = ledger[ledger.duplicated(keys, keep=False)]
    if (
        not duplicates.empty
        and (duplicates.groupby(keys).forecast_variance.nunique() > 1).any()
    ):
        raise ValueError("Runs disagree on repeated baseline predictions")
    ledger = ledger.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    actuals = ledger.groupby(["date", "asset"]).target_variance.nunique()
    if (actuals > 1).any():
        raise ValueError("Runs disagree on realized targets")
    pairs = ledger.groupby("model").apply(
        lambda g: set(zip(g.date, g.asset)), include_groups=False
    )
    if not all(value == pairs.iloc[0] for value in pairs):
        raise ValueError("Models must have identical asset/date rows")
    return ledger


def paired_block_bootstrap(differences, *, block=21, replicates=2000, seed=42):
    """Circular moving blocks over date-level paired losses, retaining all assets together."""
    values = np.asarray(differences, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or len(values) < block
        or block < 1
    ):
        raise ValueError("Need finite date losses and a valid block length")
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = []
    for _ in range(replicates):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        estimates.append(values[indices].mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_loss_difference": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "block_sessions": block,
        "replicates": replicates,
        "dates": n,
        "seed": seed,
    }


def prospective_stress(ledger, features):
    daily = features.groupby("date").market_variance_21.mean()
    flags = []
    for cutoff, group in ledger.groupby("refit_date"):
        lower, upper = group.train_start.iloc[0], group.train_end.iloc[0]
        history = daily.loc[(daily.index >= lower) & (daily.index <= upper)]
        if history.empty or upper >= cutoff:
            raise ValueError("Invalid past-only stress reference window")
        threshold = float(history.quantile(0.9))
        for date in group.date.unique():
            flags.append(
                {
                    "date": date,
                    "prospective_stress": bool(daily.loc[date] > threshold),
                    "stress_threshold": threshold,
                }
            )
    return ledger.merge(pd.DataFrame(flags), on="date", validate="many_to_one")


def analyze(
    ledger,
    output,
    *,
    benchmark="pooled",
    blocks=(5, 21, 63),
    replicates=2000,
    features=None,
):
    if output.exists():
        raise ValueError("Analysis output directory already exists")
    scored = ledger.copy()
    if features is not None:
        scored = prospective_stress(scored, features)
    scored["qlike"] = qlike(scored.target_variance, scored.forecast_variance)
    date_losses = scored.groupby(["date", "model"]).qlike.mean().unstack("model")
    if benchmark not in date_losses or date_losses.isna().any().any():
        raise ValueError("Incomplete matched model losses")
    comparisons = []
    for model in date_losses:
        if model == benchmark:
            continue
        difference = date_losses[model] - date_losses[benchmark]
        annual = difference.groupby(difference.index.year).mean()
        for block in blocks:
            comparisons.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    **paired_block_bootstrap(
                        difference, block=block, replicates=replicates
                    ),
                    "annual_win_rate": float((annual < 0).mean()),
                }
            )
    annual = (
        scored.assign(year=scored.date.dt.year)
        .groupby(["year", "model"])
        .qlike.mean()
        .reset_index()
    )
    weekly_dates = sorted(scored.date.unique())[::5]
    weekly = scored[scored.date.isin(weekly_dates)]
    worst = (
        scored.sort_values("qlike", ascending=False)
        .groupby("model", sort=False)
        .head(10)
    )
    output.mkdir(parents=True)
    summarize(ledger).to_csv(output / "per_asset_metrics.csv", index=False)
    annual.to_csv(output / "annual_qlike.csv", index=False)
    if "prospective_stress" in scored:
        scored.groupby(["model", "prospective_stress"]).qlike.agg(
            ["mean", "count"]
        ).to_csv(output / "stress_qlike.csv")
    summarize(weekly).to_csv(output / "nonoverlapping_metrics.csv", index=False)
    worst.to_csv(output / "worst_errors.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_bootstrap.csv", index=False)
    (output / "method.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "blocks": list(blocks),
                "replicates": replicates,
                "method": "circular moving-block bootstrap of paired date-average QLIKE differences",
                "interpretation": "Negative differences favor candidate. Intervals are exploratory and unadjusted for multiple comparisons.",
                "nonoverlapping": "Every fifth common session; five-session forward targets do not overlap.",
                "units": "annualized variance QLIKE; volatility errors in percentage points",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(comparisons)
