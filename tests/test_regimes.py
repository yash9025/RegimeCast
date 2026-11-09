import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from regimecast.models.regimes import causal_probabilities, forward_filter


def test_filter_is_forward_only_and_matches_manual_update():
    emission = np.log([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    start = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    actual = forward_filter(emission, start, transition)
    np.testing.assert_allclose(actual[0], [0.9, 0.1])
    expected = (actual[0] @ transition) * np.exp(emission[1])
    np.testing.assert_allclose(actual[1], expected / expected.sum())
    altered = emission.copy()
    altered[-1] = [100, -100]
    np.testing.assert_array_equal(
        actual[:2], forward_filter(altered, start, transition)[:2]
    )
    split = forward_filter(emission[2:], start, transition, previous=actual[1])
    np.testing.assert_allclose(actual[2:], split)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert np.isfinite(logsumexp(np.log(actual), axis=1)).all()


@pytest.mark.parametrize("kind", ["gmm", "hmm"])
def test_causal_gates_do_not_rewrite_past(kind):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=150)
    x = pd.DataFrame(rng.normal(size=(150, 4)), index=dates)
    a, diagnostics = causal_probabilities(x, kind=kind, states=2, warmup=60, window=100)
    boundary = dates[120]
    changed = x.copy()
    changed.loc[changed.index > boundary] += 50
    b, _ = causal_probabilities(changed, kind=kind, states=2, warmup=60, window=100)
    pd.testing.assert_frame_equal(a.loc[a.date <= boundary], b.loc[b.date <= boundary])
    assert (a.gate_fit_end < a.date).all()
    np.testing.assert_allclose(a[["regime_0", "regime_1"]].sum(axis=1), 1)
    assert len(diagnostics) > 0


@pytest.mark.parametrize("kind", ["gmm", "hmm"])
def test_archive_resume_matches_full_causal_generation(kind):
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2020-01-01", periods=110)
    frame = pd.DataFrame(rng.normal(size=(110, 4)), index=dates)
    whole, _ = causal_probabilities(frame, kind=kind, states=2, warmup=40, window=100)
    cutoff = pd.Timestamp("2020-03-31")
    first, diagnostics = causal_probabilities(
        frame.loc[:cutoff], kind=kind, states=2, warmup=40, window=100
    )
    tail, _ = causal_probabilities(
        frame,
        kind=kind,
        states=2,
        warmup=40,
        window=100,
        initial_centers=np.asarray(diagnostics[-1]["centers"]),
        after=cutoff,
    )
    pd.testing.assert_frame_equal(whole, pd.concat([first, tail], ignore_index=True))
