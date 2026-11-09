"""Idempotent experiment registration in a local SQLite MLflow store."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def tracking_client(root):
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    from mlflow.tracking import MlflowClient

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri="sqlite:///" + (root / "mlflow.db").as_posix())


def track_run(directory, root=Path("artifacts/mlflow")):
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(directory / "metrics.csv")
    fingerprint = hashlib.sha256((directory / "run.json").read_bytes()).hexdigest()
    client = tracking_client(root)
    experiment = client.get_experiment_by_name("RegimeCast")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            "RegimeCast",
            artifact_location=(Path(root).resolve() / "artifacts").as_uri(),
        )
    )
    existing = client.search_runs(
        [experiment_id],
        filter_string=f"tags.artifact_sha256 = '{fingerprint}'",
        max_results=1,
    )
    if existing:
        return existing[0].info.run_id
    run = client.create_run(
        experiment_id,
        tags={
            "artifact_sha256": fingerprint,
            "data_sha256": metadata.get("data_sha256", "unknown"),
            "evaluation_kind": metadata.get("evaluation_kind", "unknown"),
            "source_commit": metadata.get("git_commit") or "unknown",
        },
    )
    run_id = run.info.run_id
    try:
        for name, value in metadata["config"].items():
            client.log_param(run_id, name, value)
        gate = metadata.get("gate", {})
        client.log_param(run_id, "gate", gate.get("kind", "none"))
        client.log_param(run_id, "states", gate.get("states", 0))
        client.log_param(run_id, "macro", bool(metadata.get("macro")))
        client.log_param(run_id, "feature_count", len(metadata.get("features", [])))
        for row in metrics.to_dict("records"):
            for name in [
                "qlike",
                "volatility_mae_pct_points",
                "volatility_rmse_pct_points",
            ]:
                client.log_metric(
                    run_id, f"{row['model']}.{row['asset']}.{name}", float(row[name])
                )
        client.log_artifact(run_id, str(directory / "run.json"))
        client.log_artifact(run_id, str(directory / "metrics.csv"))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id
