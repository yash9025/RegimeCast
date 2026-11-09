import numpy as np
import pandas as pd
import pytest

from regimecast.data.prices import ASSETS, sessions
from regimecast.evaluation.metrics import qlike
from regimecast.evaluation.walkforward import BacktestConfig, backtest, training_rows
from regimecast.features.prices import make_features


@pytest.fixture(scope="module")
def history():
    dates = sessions("2018-01-01", "2021-12-31")
    rng = np.random.default_rng(42)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": a,
                    "adjusted_close": 100
                    * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
                }
            )
            for a in ASSETS
        ],
        ignore_index=True,
    )


def test_exact_target_and_unknown_tail(history):
    frame, _ = make_features(history)
    row = frame.iloc[0]
    prices = history[history.asset.eq(row.asset)].set_index("date").adjusted_close
    r = np.log(prices).diff()
    expected = r.loc[r.index > row.date].iloc[:5].pow(2).mean() * 252
    assert row.target_variance == pytest.approx(expected)
    assert frame.groupby("asset").tail(5).target_variance.isna().all()
    assert row.target_end == prices.index[prices.index.get_loc(row.date) + 5]


def test_future_price_perturbation_cannot_change_features(history):
    original, columns = make_features(history)
    changed = history.copy()
    boundary = pd.Timestamp("2020-01-02")
    changed.loc[changed.date > boundary, "adjusted_close"] *= 2
    perturbed, _ = make_features(changed)
    pd.testing.assert_frame_equal(
        original.loc[original.date <= boundary, columns],
        perturbed.loc[perturbed.date <= boundary, columns],
    )


def test_matured_labels_and_paired_forecasts(history):
    frame, columns = make_features(history)
    config = BacktestConfig(
        start="2021-01-01", end="2021-03-01", train_sessions=300, min_train_sessions=250
    )
    cutoff = pd.Timestamp("2021-01-04")
    train = training_rows(frame, cutoff, config)
    assert train.target_end.max() <= cutoff
    assert train.date.max() < cutoff
    assert train.date.nunique() == 300
    result, _ = backtest(frame, columns, config)
    assert (result.train_label_end <= result.refit_date).all()
    assert (result.train_end < result.refit_date).all()
    keys = [set(zip(g.date, g.asset)) for _, g in result.groupby("model")]
    assert all(k == keys[0] for k in keys)
    changed = frame.copy()
    changed.loc[changed.date >= config.end, "target_variance"] *= 100
    rerun, _ = backtest(changed, columns, config)
    pd.testing.assert_frame_equal(result, rerun)


def test_qlike_contract():
    np.testing.assert_allclose(qlike([1, 2], [1, 2]), 0)
    assert qlike([0], [1])[0] >= 0
    with pytest.raises(ValueError):
        qlike([1], [0])
    with pytest.raises(ValueError):
        qlike([np.nan], [1])


def test_pipeline_holdout_guard(tmp_path):
    from regimecast.pipeline import run_backtest

    with pytest.raises(ValueError, match="locked holdout"):
        run_backtest(
            tmp_path / "nonexistent",
            tmp_path,
            BacktestConfig(start="2023-01-01", end="2025-01-01"),
        )


def test_mixture_is_convex_and_unreliable_gate_falls_back(history):
    from threadpoolctl import threadpool_limits

    from regimecast.models.baselines import LogVarianceModel
    from regimecast.models.mixture import MixtureOfExperts

    frame, columns = make_features(history)
    train = frame.iloc[:600]
    test = frame.iloc[600:610]
    weights = np.tile([0.7, 0.3], (len(train), 1))
    probabilities = np.tile([0.25, 0.75], (len(test), 1))
    with threadpool_limits(limits=1):
        pooled = LogVarianceModel(columns).fit(train)
        mixture = MixtureOfExperts(columns, states=2, min_effective=10).fit(
            train, weights, pooled
        )
        prediction, components = mixture.predict(test, probabilities)
        np.testing.assert_allclose(prediction, (components * probabilities).sum(axis=1))
        assert (prediction >= components.min(axis=1)).all()
        assert (prediction <= components.max(axis=1)).all()
        fallback, _ = mixture.predict(
            test, probabilities, reliable=np.zeros(len(test), dtype=bool)
        )
        np.testing.assert_allclose(fallback, pooled.predict(test))
        small = MixtureOfExperts(columns, states=2, min_effective=10000).fit(
            train, weights, pooled
        )
        assert all(expert is None for expert in small.experts)
        with pytest.raises(ValueError):
            small.predict(test, probabilities * 2)


def test_regime_backtest_uses_identical_baseline_rows(history):
    frame, columns = make_features(history)
    config = BacktestConfig(
        start="2021-01-01", end="2021-02-01", train_sessions=300, min_train_sessions=250
    )
    dates = frame.date.drop_duplicates().sort_values()
    gate = pd.DataFrame(
        {
            "date": dates,
            "regime_0": 0.6,
            "regime_1": 0.4,
            "gate_reliable": True,
            "gate_fit_end": dates - pd.Timedelta("1D"),
        }
    )
    baseline, _ = backtest(frame, columns, config)
    combined, metadata = backtest(frame, columns, config, gates={"test": gate})
    paired = combined[combined.model.isin(baseline.model.unique())].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(baseline, paired)
    assert set(combined.model.unique()) == set(baseline.model.unique()) | {
        "test_mixture",
        "test_pooled_probabilities",
    }
    assert len(metadata["expert_diagnostics"]) == 1


def test_macro_pipeline_keeps_rows_and_records_vintages(history, tmp_path):
    import json

    from regimecast.data.macro import save_vintages
    from regimecast.data.prices import snapshot
    from regimecast.pipeline import run_backtest

    price_path = snapshot(
        history, tmp_path / "data", provider="test", adjustment="synthetic"
    )
    macro = pd.DataFrame(
        {
            "series": ["DGS2", "DGS10"],
            "observation_date": pd.to_datetime(["2018-01-02"] * 2),
            "vintage_date": pd.to_datetime(["2018-01-03"] * 2),
            "value": [2.0, 3.0],
        }
    )
    macro_path = save_vintages(
        macro, tmp_path / "data", provider="synthetic test fixture"
    )
    config = BacktestConfig(
        start="2021-01-01", end="2021-02-01", train_sessions=300, min_train_sessions=250
    )
    directory = run_backtest(
        price_path, tmp_path / "runs", config, macro_snapshot=macro_path
    )
    ledger = pd.read_parquet(directory / "predictions.parquet")
    assert set(ledger.model) == {"historical", "ewma", "har", "pooled_macro"}
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert len(metadata["macro"]["sha256"]) == 64
    assert any(name.startswith("macro_") for name in metadata["features"])
    assert ledger.groupby("model").size().nunique() == 1


def test_holdout_requires_exact_frozen_protocol(tmp_path):
    from regimecast.pipeline import run_backtest

    with pytest.raises(ValueError, match="frozen protocol"):
        run_backtest(
            tmp_path,
            tmp_path,
            BacktestConfig(start="2023-01-01", end="2025-01-01"),
            allow_holdout=True,
        )
