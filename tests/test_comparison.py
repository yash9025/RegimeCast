import numpy as np
import pandas as pd
import pytest

from regimecast.evaluation.comparison import merge_ledgers, paired_block_bootstrap


def test_block_bootstrap_constant_paired_improvement():
    result = paired_block_bootstrap(np.full(100, -0.1), block=10, replicates=200)
    assert result["mean_loss_difference"] == pytest.approx(-0.1)
    assert result["ci95_upper"] == pytest.approx(-0.1)
    assert result["ci95_lower"] == pytest.approx(-0.1)
    assert result == paired_block_bootstrap(
        np.full(100, -0.1), block=10, replicates=200
    )


def test_merge_rejects_unmatched_or_conflicting_forecasts(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "asset": ["SPY"],
            "model": ["pooled"],
            "target_variance": [0.01],
            "forecast_variance": [0.02],
        }
    )
    frame.to_parquet(a / "predictions.parquet", index=False)
    frame.to_parquet(b / "predictions.parquet", index=False)
    assert len(merge_ledgers([a, b])) == 1
    frame.forecast_variance = 0.03
    frame.to_parquet(b / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="disagree"):
        merge_ledgers([a, b])
    frame.model = "mixture"
    frame.date = pd.to_datetime(["2020-01-03"])
    frame.to_parquet(b / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="identical"):
        merge_ledgers([a, b])
