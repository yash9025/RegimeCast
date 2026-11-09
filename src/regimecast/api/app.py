"""Local artifact-serving API. Requests never train models or retrieve market data."""

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from regimecast.api.store import ArtifactStore, freshness


def records(frame):
    return json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=15)
    )


def create_app(catalog=None):
    store = ArtifactStore(
        catalog or Path(os.environ.get("REGIMECAST_CATALOG", "artifacts/catalog.json"))
    )
    app = FastAPI(
        title="RegimeCast",
        version="0.1.0",
        description="Saved five-session volatility research artifacts",
    )

    def read(callback):
        try:
            return callback()
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(
                503,
                "Required local artifacts are missing or invalid; prepare them using the project setup guide.",
            ) from error

    @app.get("/health")
    def health():
        try:
            store.latest()
            return {"status": "ready", "training_in_requests": False}
        except (OSError, ValueError, KeyError):
            return {"status": "needs_artifacts", "training_in_requests": False}

    @app.get("/forecasts")
    def forecasts():
        bundle = read(store.latest)
        return {
            k: bundle[k]
            for k in ["as_of", "target_end", "horizon_sessions", "forecasts", "demo"]
        }

    @app.get("/freshness")
    def data_freshness():
        return read(lambda: freshness(store.latest()))

    @app.get("/regimes")
    def regimes():
        bundle = read(store.latest)
        return {"as_of": bundle["as_of"], "regimes": bundle["regimes"]}

    @app.get("/regimes/history")
    def regime_history(
        kind: Literal["gmm", "hmm"] = "hmm", limit: int = Query(500, ge=1, le=10000)
    ):
        return read(lambda: records(store.regime_history(kind).tail(limit)))

    @app.get("/history")
    def history(
        period: Literal["development", "holdout"] = "development",
        asset: Literal["SPY", "QQQ", "TLT", "GLD"] | None = None,
        model: str | None = None,
        limit: int = Query(1000, ge=1, le=10000),
    ):
        return read(lambda: records(store.history(period, asset, model).tail(limit)))

    @app.get("/metrics")
    def metrics(period: Literal["development", "holdout"] = "development"):
        return read(lambda: records(store.metrics(period)))

    @app.get("/explanations")
    def explanations():
        bundle = read(store.latest)
        return {
            "as_of": bundle["as_of"],
            "expert_contributions": bundle.get("expert_contributions", []),
            "feature_context": bundle.get("feature_context", []),
            "interpretation": "Descriptive model contributions and observed features; not causal effects or confidence intervals.",
        }

    @app.get("/metadata")
    def metadata():
        bundle = read(store.latest)
        return {
            "as_of": bundle["as_of"],
            "models": bundle.get("model_metadata", {}),
            "data": bundle.get("data_metadata", {}),
            "periods": read(store.periods),
            "demo": bundle["demo"],
        }

    return app


app = create_app()
