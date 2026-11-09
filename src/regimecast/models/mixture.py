"""Soft-weighted experts; all combinations operate in variance space."""

import numpy as np

from regimecast.models.baselines import LogVarianceModel


class MixtureOfExperts:
    def __init__(self, columns, states=3, seed=42, min_effective=200):
        self.columns, self.states, self.seed = columns, states, seed
        self.min_effective = min_effective

    def fit(self, train, probabilities, pooled):
        self.pooled = pooled
        self.experts, self.diagnostics = [], []
        for state in range(self.states):
            weights = probabilities[:, state]
            effective = float(weights.sum() ** 2 / max(np.dot(weights, weights), 1e-12))
            mass = float(weights.sum())
            fallback = effective < self.min_effective or mass < 50
            expert = (
                None
                if fallback
                else LogVarianceModel(self.columns, seed=self.seed).fit(train, weights)
            )
            self.experts.append(expert)
            self.diagnostics.append(
                {
                    "state": state,
                    "effective_rows": effective,
                    "weight_mass": mass,
                    "fallback": fallback,
                }
            )
        return self

    def predict(self, frame, probabilities, reliable=None):
        probabilities = np.asarray(probabilities)
        if (
            probabilities.shape != (len(frame), self.states)
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError("Invalid mixture probability shape or values")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Mixture probabilities must sum to one")
        pooled = self.pooled.predict(frame)
        components = np.column_stack(
            [
                pooled if expert is None else expert.predict(frame)
                for expert in self.experts
            ]
        )
        prediction = (components * probabilities).sum(axis=1)
        if reliable is not None:
            prediction = np.where(reliable, prediction, pooled)
        return prediction, components
