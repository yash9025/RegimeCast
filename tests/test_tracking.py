import json

import pandas as pd

from regimecast.tracking import track_run, tracking_client


def test_tracking_is_local_and_idempotent(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "config": {"seed": 42},
                "data_sha256": "synthetic-fixture",
                "evaluation_kind": "synthetic test fixture",
                "features": ["x"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": "pooled",
                "asset": "SPY",
                "qlike": 0.2,
                "volatility_mae_pct_points": 2.0,
                "volatility_rmse_pct_points": 3.0,
            }
        ]
    ).to_csv(run / "metrics.csv", index=False)
    root = tmp_path / "tracking"
    first = track_run(run, root)
    assert track_run(run, root) == first
    saved = tracking_client(root).get_run(first)
    assert saved.info.status == "FINISHED"
    assert saved.data.metrics["pooled.SPY.qlike"] == 0.2
    assert (root / "mlflow.db").exists()
