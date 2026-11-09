"""Causal Gaussian gates with explicit HMM forward filtering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def market_state(prices):
    wide = prices.pivot(
        index="date", columns="asset", values="adjusted_close"
    ).sort_index()
    r = np.log(wide).diff()
    state = pd.DataFrame(index=wide.index)
    state["log_variance"] = np.log(
        r.pow(2).rolling(21).mean().mean(axis=1) * 252 + 1e-10
    )
    state["drawdown"] = (wide / wide.rolling(63).max() - 1).mean(axis=1)
    pairs = [
        r[a].rolling(21).corr(r[b])
        for i, a in enumerate(wide)
        for b in wide.columns[i + 1 :]
    ]
    state["correlation"] = pd.concat(pairs, axis=1).mean(axis=1)
    state["return_5"] = np.log(wide / wide.shift(5)).mean(axis=1)
    return state.replace([np.inf, -np.inf], np.nan).dropna()


def forward_filter(log_emission, startprob, transmat, previous=None):
    """Return P(z_t | x_1..x_t), never backward-smoothed probabilities."""
    log_emission = np.asarray(log_emission)
    if log_emission.ndim != 2 or not len(log_emission):
        raise ValueError("Emissions must be a nonempty observation-by-state matrix")
    tiny = np.finfo(float).tiny
    log_transition = np.log(np.maximum(transmat, tiny))
    posterior = None if previous is None else np.log(np.maximum(previous, tiny))
    result = []
    for emission in log_emission:
        prior = (
            np.log(np.maximum(startprob, tiny))
            if posterior is None
            else logsumexp(posterior[:, None] + log_transition, axis=0)
        )
        posterior = prior + emission
        posterior -= logsumexp(posterior)
        result.append(np.exp(posterior))
    return np.asarray(result)


class GaussianGate:
    def __init__(self, kind="gmm", states=3, seed=42):
        if kind not in {"gmm", "hmm"} or states not in {2, 3, 4}:
            raise ValueError("Gate must be GMM/HMM with 2, 3 or 4 states")
        self.kind, self.states, self.seed = kind, states, seed

    def fit(self, history, previous_centers=None):
        self.scaler = StandardScaler().fit(history)
        x = self.scaler.transform(history)
        if self.kind == "gmm":
            self.model = GaussianMixture(
                self.states,
                covariance_type="full",
                reg_covar=1e-4,
                max_iter=150,
                n_init=2,
                random_state=self.seed,
            ).fit(x)
            converged = bool(self.model.converged_)
            iterations = int(self.model.n_iter_)
        else:
            self.model = GaussianHMM(
                self.states,
                covariance_type="full",
                n_iter=100,
                tol=0.01,
                min_covar=1e-3,
                random_state=self.seed,
            ).fit(x)
            changes = list(self.model.monitor_.history)
            # hmmlearn's converged also reports True at the iteration cap; inspect the gain instead.
            gain = changes[-1] - changes[-2] if len(changes) > 1 else float("inf")
            converged = bool(-0.01 <= gain < 0.01)
            iterations = int(self.model.monitor_.iter)
        raw_centers = self.scaler.inverse_transform(self.model.means_)
        if previous_centers is None:
            self.order = np.argsort(raw_centers[:, 0])
        else:
            distances = (
                (previous_centers[:, None, :] - raw_centers[None, :, :])
                / self.scaler.scale_
            ) ** 2
            rows, columns = linear_sum_assignment(distances.sum(axis=2))
            self.order = columns[np.argsort(rows)]
        self.centers = raw_centers[self.order]
        if not np.isfinite(self.centers).all():
            raise ValueError("Nonfinite regime centers")
        if self.kind == "hmm":
            self.previous = self._hmm_probabilities(x)[-1]
        self.diagnostics = {
            "converged": converged,
            "iterations": iterations,
            "state_order": self.order.tolist(),
            "centers": self.centers.tolist(),
        }
        return self

    def _hmm_probabilities(self, x, previous=None):
        emissions = np.column_stack(
            [
                multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=True)
                for mean, cov in zip(self.model.means_, self.model.covars_)
            ]
        )
        return forward_filter(
            emissions, self.model.startprob_, self.model.transmat_, previous
        )

    def predict(self, observations):
        x = self.scaler.transform(observations)
        if self.kind == "gmm":
            result = self.model.predict_proba(x)
        else:
            result = self._hmm_probabilities(x, self.previous)
            self.previous = result[-1]
        return result[:, self.order]


def causal_probabilities(
    state,
    *,
    kind="gmm",
    states=3,
    seed=42,
    warmup=126,
    window=1260,
    initial_centers=None,
    after=None,
):
    """Fit monthly on strictly earlier state observations; append immutable causal rows."""
    if len(state) <= warmup or warmup < states * 5:
        raise ValueError("Insufficient regime warmup")
    columns = [f"regime_{i}" for i in range(states)]
    rows, diagnostics, previous_centers = [], [], initial_centers
    future = state.iloc[warmup:]
    if after is not None:
        future = future[future.index > pd.Timestamp(after)]
        if future.empty:
            raise ValueError("No new observations after cached archive")
        if future.index[0].to_period("M") == pd.Timestamp(after).to_period("M"):
            raise ValueError("Resume requires a completed month boundary")
    with threadpool_limits(limits=1):
        for month, observations in future.groupby(
            future.index.to_period("M"), sort=True
        ):
            cutoff = observations.index.min()
            history = state.loc[state.index < cutoff].tail(window)
            if not diagnostics or month.month == 1:
                logging.getLogger(__name__).info(
                    "%s gate archive: %s (%d states)", kind, month, states
                )
            fallback_reason = None
            try:
                gate = GaussianGate(kind, states, seed).fit(history, previous_centers)
                probabilities = gate.predict(observations)
                if not np.isfinite(probabilities).all() or not np.allclose(
                    probabilities.sum(axis=1), 1
                ):
                    raise ValueError("Invalid regime probabilities")
                previous_centers = gate.centers
                details = gate.diagnostics
            except (ValueError, np.linalg.LinAlgError) as error:
                probabilities = np.full((len(observations), states), 1 / states)
                fallback_reason = f"{type(error).__name__}: {error}"
                details = {"converged": False}
            result = pd.DataFrame(
                probabilities, index=observations.index, columns=columns
            )
            result["gate_fit_end"] = history.index.max()
            result["gate_reliable"] = bool(
                details["converged"] and fallback_reason is None
            )
            rows.append(result)
            diagnostics.append(
                {
                    "month": str(month),
                    "fit_end": str(history.index.max().date()),
                    "fallback_reason": fallback_reason,
                    **details,
                }
            )
    return pd.concat(rows).rename_axis("date").reset_index(), diagnostics
