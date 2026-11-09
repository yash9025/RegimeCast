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
