"""Read-only access to trusted local forecast and evaluation artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from regimecast.evaluation.comparison import merge_ledgers
from regimecast.evaluation.metrics import summarize


def strict_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def load_bundle(path):
    bundle = strict_json(path)
    if bundle.get("schema_version") != 1 or not bundle.get("forecasts"):
        raise ValueError("Forecast artifact schema is missing or unsupported")
    if pd.Timestamp(bundle["target_end"]) <= pd.Timestamp(bundle["as_of"]):
        raise ValueError("Forecast horizon must follow its as-of date")
    for row in bundle["forecasts"]:
        value = row["forecast_variance"]
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Forecast variance must be finite and positive")
        if not np.isclose(row["volatility_pct"], 100 * np.sqrt(value)):
            raise ValueError("Forecast display units do not match variance")
    for gate in bundle.get("regimes", {}).values():
        probabilities = np.asarray(gate["probabilities"], dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1)
        ):
            raise ValueError("Invalid saved regime probabilities")
    return bundle


def freshness(bundle, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tz is None:
        raise ValueError("Freshness clock must be timezone-aware")
    as_of = pd.Timestamp(bundle["as_of"])
    if as_of.date() > now.date():
        return {
            "stale": True,
            "reason": "Artifact as-of date is in the future",
            "latest_expected_session": None,
        }
    calendar = xcals.get_calendar(
        "XNYS",
        start=as_of - pd.Timedelta("14D"),
        end=now.tz_localize(None).normalize() + pd.Timedelta("14D"),
    )
    complete = calendar.schedule[
        calendar.schedule["close"] + pd.Timedelta("30min") <= now
    ]
    latest = complete.index[-1].tz_localize(None)
    return {
        "stale": bool(as_of < latest),
        "latest_expected_session": str(latest.date()),
        "as_of": str(as_of.date()),
        "checked_at": now.isoformat(),
    }


@lru_cache(maxsize=8)
def _ledger(paths, mtimes):
    return merge_ledgers([Path(p) for p in paths])


class ArtifactStore:
    def __init__(self, catalog=Path("artifacts/catalog.json")):
        self.catalog = Path(catalog)

    def settings(self):
        return strict_json(self.catalog)

    def latest(self):
        return load_bundle(Path(self.settings()["latest_bundle"]))

    def periods(self):
        return list(self.settings().get("evaluation_runs", {}))

    def history(self, period="development", asset=None, model=None):
        runs = self.settings().get("evaluation_runs", {}).get(period)
        if not runs:
            raise ValueError("Requested evaluation period is not ready")
        paths = tuple(str(Path(p).resolve()) for p in runs)
        mtimes = tuple(
            (Path(p) / "predictions.parquet").stat().st_mtime_ns for p in paths
        )
        frame = _ledger(paths, mtimes)
        if asset is not None:
            frame = frame[frame.asset.eq(asset)]
        if model is not None:
            frame = frame[frame.model.eq(model)]
        return frame.copy()

    def metrics(self, period="development"):
        return summarize(self.history(period))

    def regime_history(self, kind):
        path = self.settings().get("regime_history", {}).get(kind)
        if path is None:
            raise ValueError("Regime history is not ready")
        return pd.read_parquet(path)

    def analysis(self, period="development"):
        directory = self.settings().get("analysis", {}).get(period)
        if directory is None:
            raise ValueError("Analysis is not ready")
        return Path(directory)
