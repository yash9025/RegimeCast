"""Causal price features and explicitly future-dependent research labels."""

import numpy as np
import pandas as pd

from regimecast.data.prices import validate_prices

HORIZON = 5


def make_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    assets = tuple(sorted(prices.asset.unique()))
    validate_prices(prices, assets)
    wide = prices.pivot(
        index="date", columns="asset", values="adjusted_close"
    ).sort_index()
    returns = np.log(wide).diff()
    market_variance = returns.pow(2).rolling(21).mean().mean(axis=1) * 252
    tables = []
    for asset in assets:
        p, r = wide[asset], returns[asset]
        frame = pd.DataFrame(index=wide.index)
        for window in (1, 5, 10, 21, 63):
            frame[f"return_{window}"] = np.log(p / p.shift(window))
        for window in (5, 10, 21, 63):
            frame[f"variance_{window}"] = r.pow(2).rolling(window).mean() * 252
        frame["ewma_variance"] = (
            r.pow(2).ewm(alpha=0.06, adjust=False, min_periods=63).mean() * 252
        )
        for window in (21, 63, 252):
            frame[f"drawdown_{window}"] = p / p.rolling(window).max() - 1
        frame["downside_21"] = r.clip(upper=0).pow(2).rolling(21).mean() * 252
        for window in (21, 63):
            for other in assets:
                if other != asset:
                    frame[f"correlation_{other}_{window}"] = r.rolling(window).corr(
                        returns[other]
                    )
                else:
                    frame[f"correlation_{other}_{window}"] = 1.0
        frame["market_variance_21"] = market_variance
        for category in assets:
            frame[f"asset_{category}"] = float(asset == category)
        feature_columns = list(frame.columns)
        frame["asset"] = asset
        frame["target_variance"] = (
            sum(r.pow(2).shift(-j) for j in range(1, HORIZON + 1)) * 252 / HORIZON
        )
        frame["target_end"] = pd.Series(wide.index, index=wide.index).shift(-HORIZON)
        tables.append(frame.rename_axis("date").reset_index())
    result = (
        pd.concat(tables, ignore_index=True)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )
    result[feature_columns] = result[feature_columns].replace([np.inf, -np.inf], np.nan)
    # Warmup is excluded; trailing rows with unknown labels remain available for live forecasts.
    return result.dropna(subset=feature_columns).reset_index(drop=True), feature_columns
