import json

import pandas as pd
from fastapi.testclient import TestClient

from regimecast.api.app import create_app
from regimecast.api.store import ArtifactStore, freshness


def test_api_serves_saved_artifacts_only(artifact_catalog, monkeypatch):
    from regimecast.models.baselines import LogVarianceModel

    def forbidden(*args, **kwargs):
        raise AssertionError("API requests must never train models")

    monkeypatch.setattr(LogVarianceModel, "fit", forbidden)
    client = TestClient(create_app(artifact_catalog))
    assert client.get("/health").json()["status"] == "ready"
    assert len(client.get("/forecasts").json()["forecasts"]) == 4
    for path in [
        "/regimes",
        "/regimes/history",
        "/metrics",
        "/metadata",
        "/explanations",
        "/freshness",
    ]:
        assert client.get(path).status_code == 200
    assert len(client.get("/history?asset=SPY").json()) == 1
    assert client.get("/history?asset=INVALID").status_code == 422
    assert client.get("/history?limit=10001").status_code == 422
    assert client.get("/metrics?period=holdout").status_code == 503


def test_missing_or_corrupt_artifacts_fail_honestly(artifact_catalog, tmp_path):
    client = TestClient(create_app(tmp_path / "missing.json"))
    assert client.get("/health").json()["status"] == "needs_artifacts"
    assert client.get("/forecasts").status_code == 503
    settings = json.loads(artifact_catalog.read_text())
    from pathlib import Path

    bundle_path = Path(settings["latest_bundle"])
    bundle = json.loads(bundle_path.read_text())
    bundle["forecasts"][0]["forecast_variance"] = -1
    bundle_path.write_text(json.dumps(bundle))
    assert TestClient(create_app(artifact_catalog)).get("/forecasts").status_code == 503


def test_freshness_respects_session_close_and_holidays(artifact_catalog):
    bundle = ArtifactStore(artifact_catalog).latest()
    assert not freshness(bundle, pd.Timestamp("2024-01-03 20:00Z"))["stale"]
    assert freshness(bundle, pd.Timestamp("2024-01-03 22:00Z"))["stale"]
