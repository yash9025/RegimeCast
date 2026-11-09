"""Offline operational forecast preparation; never called by web requests."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from regimecast.artifacts import matching_analyses
from regimecast.data.macro import load_vintages, macro_features
from regimecast.data.prices import load_snapshot, sessions
from regimecast.evaluation.walkforward import BacktestConfig, training_rows
from regimecast.features.prices import make_features
from regimecast.models.baselines import HAR_FEATURES, LogVarianceModel
from regimecast.models.mixture import MixtureOfExperts
from regimecast.models.regimes import GaussianGate, causal_probabilities, market_state


def operational_archive(prices, kind, cache, data_sha256):
    state = market_state(prices)
    if cache is None:
        archive, diagnostics = causal_probabilities(state, kind=kind)
    else:
        metadata = json.loads((cache / "run.json").read_text(encoding="utf-8"))
        expected = {
            "kind": kind,
            "states": 3,
            "seed": 42,
            "warmup": 126,
            "window": 1260,
        }
        if metadata["gate"] != expected or metadata["data_sha256"] != data_sha256:
            raise ValueError(
                "Operational gate cache does not match data and configuration"
            )
        archive = pd.read_parquet(cache / "regime_probabilities.parquet")
        diagnostics = json.loads(
            (cache / "regime_diagnostics.json").read_text(encoding="utf-8")
        )
        if state.index.max() > archive.date.max():
            centers = next(
                d["centers"] for d in reversed(diagnostics) if "centers" in d
            )
            tail, new_diagnostics = causal_probabilities(
                state,
                kind=kind,
                initial_centers=np.asarray(centers),
                after=archive.date.max(),
            )
            archive = pd.concat([archive, tail], ignore_index=True)
            diagnostics += new_diagnostics
    return archive, diagnostics, state


def discover_evaluations(root, data_sha256):
    selected = {}
    for metadata_path in root.glob("*/run.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("data_sha256") != data_sha256:
            continue
        config = metadata["config"]
        dates = (config["start"], config["end"])
        period = {
            ("2013-01-01", "2024-01-01"): "development",
            ("2024-01-01", "2026-01-01"): "holdout",
        }.get(dates)
        if period is None:
            continue
        gate = metadata.get("gate")
        if gate is not None:
            if gate["states"] != 3:
                continue
            variant = gate["kind"]
        elif "macro" in metadata:
            variant = "macro"
        else:
            continue
        key = (period, variant)
        if key not in selected or metadata["created_at"] > selected[key][0]:
            selected[key] = (
                metadata["created_at"],
                str(metadata_path.parent.resolve()),
            )
    return {
        period: [
            selected[(period, v)][1]
            for v in ["gmm", "hmm", "macro"]
            if (period, v) in selected
        ]
        for period in ["development", "holdout"]
        if any(p == period for p, _ in selected)
    }


def prepare_forecasts(
    snapshot, output, *, macro_snapshot=None, gmm_cache=None, hmm_cache=None
):
    prices = load_snapshot(snapshot)
    data = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    frame, columns = make_features(prices)
    macro_columns = []
    if macro_snapshot is not None:
        macro = macro_features(load_vintages(macro_snapshot), frame.date)
        macro_columns = [c for c in macro if c != "date"]
        frame = frame.merge(macro, on="date", how="left", validate="many_to_one")
    as_of = frame.date.max()
    refit = frame.loc[
        frame.date.dt.to_period("M") == as_of.to_period("M"), "date"
    ].min()
    config = BacktestConfig(
        start=str(refit.date()), end=str((as_of + pd.Timedelta("1D")).date())
    )
    train = training_rows(frame, refit, config)
    if train.date.nunique() < config.min_train_sessions:
        raise ValueError("Insufficient matured operational training history")
    latest = frame[frame.date.eq(as_of)].copy()
    now = datetime.now(UTC)
    directory = (
        output
        / "production"
        / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    )
    directory.mkdir(parents=True, exist_ok=False)
    forecasts, contributions, regimes, models, history_paths = [], [], {}, {}, {}
    with threadpool_limits(limits=1):
        pooled = LogVarianceModel(columns).fit(train)
        har = LogVarianceModel(HAR_FEATURES, linear=True).fit(train)
        values = {
            "historical": latest.variance_21.to_numpy(),
            "ewma": latest.ewma_variance.to_numpy(),
            "har": har.predict(latest),
            "pooled": pooled.predict(latest),
        }
        models.update({"pooled": pooled, "har": har})
        if macro_columns:
            macro_model = LogVarianceModel(columns + macro_columns).fit(train)
            values["pooled_macro"] = macro_model.predict(latest)
            models["pooled_macro"] = macro_model
        for kind, cache in [("gmm", gmm_cache), ("hmm", hmm_cache)]:
            archive, diagnostics, state = operational_archive(
                prices, kind, cache, data["sha256"]
            )
            indexed = archive.set_index("date")
            probability_columns = [f"regime_{i}" for i in range(3)]
            train_gate = indexed.loc[train.date]
            current = indexed.loc[latest.date]
            probabilities = current[probability_columns].to_numpy()
            reliable = current.gate_reliable.to_numpy()
            mixture = MixtureOfExperts(columns).fit(
                train, train_gate[probability_columns].to_numpy(), pooled
            )
            prediction, components = mixture.predict(latest, probabilities, reliable)
            components[~reliable, :] = pooled.predict(latest)[~reliable, None]
            values[kind + "_mixture"] = prediction
            models[kind + "_mixture"] = mixture
            augmented_train, augmented_latest = train.copy(), latest.copy()
            for column in probability_columns:
                augmented_train[column] = train_gate[column].to_numpy()
                augmented_latest[column] = current[column].to_numpy()
            augmented = LogVarianceModel(columns + probability_columns).fit(
                augmented_train
            )
            values[kind + "_pooled_probabilities"] = augmented.predict(augmented_latest)
            models[kind + "_pooled_probabilities"] = augmented
            gate_history = state.loc[state.index < refit].tail(1260)
            previous_centers = next(
                (d["centers"] for d in reversed(diagnostics[:-1]) if "centers" in d),
                None,
            )
            gate = GaussianGate(kind).fit(
                gate_history,
                None if previous_centers is None else np.asarray(previous_centers),
            )
            current_state = state.loc[(state.index >= refit) & (state.index <= as_of)]
            np.testing.assert_allclose(
                gate.predict(current_state)[-1], probabilities[0], rtol=1e-8, atol=1e-10
            )
            models[kind + "_gate"] = gate
            regimes[kind] = {
                "probabilities": probabilities[0].tolist(),
                "reliable": bool(reliable[0]),
                "fit_end": str(current.gate_fit_end.iloc[0].date()),
            }
            path = directory / f"regimes_{kind}.parquet"
            archive.to_parquet(path, index=False)
            history_paths[kind] = str(path.resolve())
            for row, asset in enumerate(latest.asset):
                for k in range(3):
                    contributions.append(
                        {
                            "asset": asset,
                            "gate": kind,
                            "state": k,
                            "probability": float(probabilities[row, k]),
                            "expert_variance": float(components[row, k]),
                            "weighted_variance": float(
                                probabilities[row, k] * components[row, k]
                            ),
                            "fallback": bool(
                                not reliable[row] or mixture.experts[k] is None
                            ),
                        }
                    )
        for model, predictions in values.items():
            for asset, variance in zip(latest.asset, predictions):
                forecasts.append(
                    {
                        "asset": asset,
                        "model": model,
                        "forecast_variance": float(variance),
                        "volatility_pct": float(100 * np.sqrt(variance)),
                    }
                )
    context = []
    for row in latest.itertuples():
        item = {
            "asset": row.asset,
            "trailing_21_session_volatility_pct": 100 * np.sqrt(row.variance_21),
            "five_session_return_pct": 100 * np.expm1(row.return_5),
            "63_session_drawdown_pct": 100 * row.drawdown_63,
        }
        if macro_columns:
            item["10Y_minus_2Y_pct_points"] = float(latest.macro_curve.iloc[0])
            item["inflation_yoy_pct"] = float(latest.macro_CPIAUCSL_yoy_pct.iloc[0])
        context.append(item)
    target_sessions = sessions(
        str(as_of.date()), str((as_of + pd.Timedelta("21D")).date())
    )[1:6]
    joblib.dump(models, directory / "forecasters.joblib")
    bundle = {
        "schema_version": 1,
        "demo": bool(
            "synthetic" in data["provider"].lower()
            or data["provider"].lower() == "test"
        ),
        "as_of": str(as_of.date()),
        "target_end": str(target_sessions[-1].date()),
        "horizon_sessions": 5,
        "forecast_cutoff": (
            as_of.tz_localize("America/New_York") + pd.Timedelta("16h30min")
        )
        .tz_convert("UTC")
        .isoformat(),
        "forecasts": forecasts,
        "regimes": regimes,
        "expert_contributions": contributions,
        "feature_context": context,
        "model_metadata": {
            "refit_date": str(refit.date()),
            "train_start": str(train.date.min().date()),
            "train_end": str(train.date.max().date()),
            "latest_training_label": str(train.target_end.max().date()),
            "states": 3,
            "seed": 42,
            "versions": {
                p: version(p) for p in ["numpy", "pandas", "scikit-learn", "hmmlearn"]
            },
            "model_sha256": hashlib.sha256(
                (directory / "forecasters.joblib").read_bytes()
            ).hexdigest(),
        },
        "data_metadata": {
            "price_sha256": data["sha256"],
            "provider": data["provider"],
            "downloaded_at": data["downloaded_at"],
            "historical_price_vintage": False,
            "adjustment": data["adjustment"],
            "macro_sha256": None
            if macro_snapshot is None
            else json.loads(
                (macro_snapshot / "manifest.json").read_text(encoding="utf-8")
            )["sha256"],
            "macro_availability": "First trading session strictly after each vintage release date",
        },
    }
    # Missing feature context remains explicit JSON null, never an invalid JSON NaN.
    bundle["feature_context"] = json.loads(
        pd.DataFrame(context).to_json(orient="records", double_precision=15)
    )
    bundle_path = directory / "latest.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8"
    )
    catalog_path = output / "catalog.json"
    catalog = {
        "latest_bundle": str(bundle_path.resolve()),
        "regime_history": history_paths,
        "evaluation_runs": discover_evaluations(output, data["sha256"]),
        "analysis": matching_analyses(
            output, data["sha256"], bundle["data_metadata"].get("macro_sha256")
        ),
    }
    temporary = output / "catalog.tmp.json"
    temporary.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    temporary.replace(catalog_path)
    return directory
