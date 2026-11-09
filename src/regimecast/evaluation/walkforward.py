"""Monthly refits with date-level splits and fully matured training labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from regimecast.models.baselines import HAR_FEATURES, LogVarianceModel


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2013-01-01"
    end: str = "2024-01-01"  # exclusive: 2024 onward reserved for locked holdout
    train_sessions: int = 1260
    min_train_sessions: int = 1260
    seed: int = 42

    def __post_init__(self):
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("Backtest start must precede exclusive end")
        if not 1 <= self.min_train_sessions <= self.train_sessions:
            raise ValueError("Invalid training window")


def training_rows(frame, cutoff, config):
    eligible = frame[(frame.date < cutoff) & (frame.target_end <= cutoff)].dropna(
        subset=["target_variance"]
    )
    dates = eligible.date.drop_duplicates().sort_values().tail(config.train_sessions)
    return eligible[eligible.date.isin(dates)]


def backtest(frame, columns, config=BacktestConfig(), gates=None):
    """All models forecast identical dates/assets; no random validation splitting."""
    evaluation = frame[(frame.date >= config.start) & (frame.date < config.end)].dropna(
        subset=["target_variance"]
    )
    if evaluation.empty:
        raise ValueError("No complete targets in evaluation range")
    predictions, fits, expert_diagnostics = [], [], []
    with threadpool_limits(limits=1):
        for month, test in evaluation.groupby(
            evaluation.date.dt.to_period("M"), sort=True
        ):
            cutoff = test.date.min()
            train = training_rows(frame, cutoff, config)
            if train.date.nunique() < config.min_train_sessions:
                raise ValueError(f"Insufficient training history at {cutoff.date()}")
            pooled = LogVarianceModel(columns, seed=config.seed).fit(train)
            har = LogVarianceModel(HAR_FEATURES, linear=True, seed=config.seed).fit(
                train
            )
            forecasts = {
                "historical": test.variance_21.to_numpy(),
                "ewma": test.ewma_variance.to_numpy(),
                "har": har.predict(test),
                "pooled": pooled.predict(test),
            }
            if gates:
                from regimecast.models.mixture import MixtureOfExperts

                for kind, gate in gates.items():
                    indexed = gate.set_index("date")
                    probability_columns = [c for c in gate if c.startswith("regime_")]
                    train_gate = indexed.loc[train.date].reset_index(drop=True)
                    test_gate = indexed.loc[test.date].reset_index(drop=True)
                    weights = train_gate[probability_columns].to_numpy()
                    probabilities = test_gate[probability_columns].to_numpy()
                    mixture = MixtureOfExperts(
                        columns, len(probability_columns), config.seed
                    ).fit(train, weights, pooled)
                    values, _ = mixture.predict(
                        test, probabilities, test_gate.gate_reliable.to_numpy()
                    )
                    forecasts[f"{kind}_mixture"] = values
                    augmented_train, augmented_test = train.copy(), test.copy()
                    for column in probability_columns:
                        augmented_train[column] = train_gate[column].to_numpy()
                        augmented_test[column] = test_gate[column].to_numpy()
                    augmented = LogVarianceModel(
                        columns + probability_columns, seed=config.seed
                    ).fit(augmented_train)
                    forecasts[f"{kind}_pooled_probabilities"] = augmented.predict(
                        augmented_test
                    )
                    expert_diagnostics.append(
                        {
                            "month": str(month),
                            "gate": kind,
                            "experts": mixture.diagnostics,
                            "unreliable_forecasts": int(
                                (~test_gate.gate_reliable).sum()
                            ),
                        }
                    )
            for name, values in forecasts.items():
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(f"Invalid {name} predictions")
                out = test[["date", "asset", "target_end", "target_variance"]].copy()
                out["model"] = name
                out["forecast_variance"] = values
                out["refit_date"] = cutoff
                out["train_start"] = train.date.min()
                out["train_end"] = train.date.max()
                out["train_label_end"] = train.target_end.max()
                predictions.append(out)
            fits.append(
                {
                    "month": str(month),
                    "train_rows": len(train),
                    "train_sessions": train.date.nunique(),
                    "test_rows": len(test),
                }
            )
    ledger = pd.concat(predictions, ignore_index=True)
    ledger["forecast_cutoff"] = (
        pd.DatetimeIndex(ledger.date).tz_localize("America/New_York")
        + pd.Timedelta("16h30min")
    ).tz_convert("UTC")
    if ledger.duplicated(["date", "asset", "model"]).any():
        raise ValueError("Duplicate predictions")
    return ledger, {
        "config": asdict(config),
        "fits": fits,
        "features": columns,
        "expert_diagnostics": expert_diagnostics,
    }
