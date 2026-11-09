"""Positive-variance baselines sharing a common training sample."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FLOOR = 1e-10
HAR_FEATURES = ["variance_5", "variance_21", "variance_63"]


def learner(seed=42):
    return HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.06,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


class LogVarianceModel:
    """Train log variance; training-only smearing corrects retransformation bias."""

    def __init__(self, columns, *, linear=False, seed=42):
        self.columns = list(columns)
        self.linear = linear
        self.model = (
            make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            if linear
            else learner(seed)
        )

    def fit(self, frame, sample_weight=None):
        x = frame[self.columns].to_numpy()
        if self.linear:
            x = np.log(np.maximum(x, FLOOR))
        y = np.log(np.maximum(frame.target_variance.to_numpy(), FLOOR))
        kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.model.fit(x, y, **kwargs)
        residual = y - self.model.predict(x)
        self.smearing = float(
            np.average(np.exp(np.clip(residual, -20, 20)), weights=sample_weight)
        )
        return self

    def predict(self, frame):
        x = frame[self.columns].to_numpy()
        if self.linear:
            x = np.log(np.maximum(x, FLOOR))
        return np.maximum(
            np.exp(np.clip(self.model.predict(x), -25, 10)) * self.smearing, FLOOR
        )
