import json

import pandas as pd
import pytest

from regimecast.data.prices import ASSETS


@pytest.fixture
def artifact_catalog(tmp_path):
    bundle = {
        "schema_version": 1,
        "demo": True,
        "as_of": "2024-01-02",
        "target_end": "2024-01-09",
        "horizon_sessions": 5,
        "forecasts": [
            {
                "asset": asset,
                "model": "har",
                "forecast_variance": 0.04,
                "volatility_pct": 20.0,
            }
            for asset in ASSETS
        ],
        "regimes": {"hmm": {"probabilities": [0.5, 0.3, 0.2], "reliable": True}},
        "expert_contributions": [],
        "feature_context": [],
        "model_metadata": {},
        "data_metadata": {"kind": "synthetic test fixture"},
    }
    bundle_path = tmp_path / "latest.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03"] * 4),
            "asset": list(ASSETS),
            "model": "har",
            "target_variance": 0.02,
            "forecast_variance": 0.04,
            "target_end": pd.to_datetime(["2023-01-10"] * 4),
        }
    )
    rows.to_parquet(run / "predictions.parquet", index=False)
    regime = tmp_path / "regimes.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03"]),
            "regime_0": [0.5],
            "regime_1": [0.3],
            "regime_2": [0.2],
        }
    ).to_parquet(regime, index=False)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "latest_bundle": str(bundle_path),
                "evaluation_runs": {"development": [str(run)]},
                "regime_history": {"hmm": str(regime), "gmm": str(regime)},
            }
        ),
        encoding="utf-8",
    )
    return catalog
