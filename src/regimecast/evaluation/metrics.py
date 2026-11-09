"""Variance loss and volatility errors; values use annualized decimal units."""

import numpy as np
import pandas as pd

from regimecast.models.baselines import FLOOR


def qlike(actual, forecast):
    actual, forecast = (
        np.asarray(actual, dtype=float),
        np.asarray(forecast, dtype=float),
    )
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Loss inputs must be finite")
    if (actual < 0).any() or (forecast <= 0).any():
        raise ValueError("Actual variance must be nonnegative and forecasts positive")
    ratio = np.maximum(actual, FLOOR) / np.maximum(forecast, FLOOR)
    return ratio - np.log(ratio) - 1


def summarize(ledger):
    rows = []
    for (model, asset), group in ledger.groupby(["model", "asset"]):
        error = 100 * (
            np.sqrt(group.forecast_variance) - np.sqrt(group.target_variance)
        )
        rows.append(
            {
                "model": model,
                "asset": asset,
                "observations": len(group),
                "qlike": float(
                    np.mean(qlike(group.target_variance, group.forecast_variance))
                ),
                "volatility_mae_pct_points": float(np.mean(np.abs(error))),
                "volatility_rmse_pct_points": float(np.sqrt(np.mean(error**2))),
            }
        )
    return pd.DataFrame(rows)
