"""Persist reproducible, separately identified development runs."""

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from regimecast.data.prices import load_snapshot
from regimecast.evaluation.metrics import summarize
from regimecast.evaluation.walkforward import BacktestConfig, backtest
from regimecast.features.prices import make_features


def run_backtest(snapshot: Path, output: Path, config: BacktestConfig):
    if config.end > "2024-01-01":
        raise ValueError(
            "2024 onward is locked holdout; development runs must end by 2024-01-01"
        )
    prices = load_snapshot(snapshot)
    features, columns = make_features(prices)
    ledger, metadata = backtest(features, columns, config)
    now = datetime.now(UTC)
    directory = output / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    ledger.to_parquet(directory / "predictions.parquet", index=False)
    features.to_parquet(directory / "features.parquet", index=False)
    summarize(ledger).to_csv(directory / "metrics.csv", index=False)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "created_at": now.isoformat(),
            "data_sha256": manifest["sha256"],
            "snapshot": str(snapshot.resolve()),
            "python": platform.python_version(),
            "versions": {
                p: version(p) for p in ["numpy", "pandas", "scikit-learn", "pyarrow"]
            },
            "horizon_sessions": 5,
            "units": "annualized decimal variance",
            "evaluation_kind": "development; 2024 onward excluded",
            "predictions_sha256": hashlib.sha256(
                (directory / "predictions.parquet").read_bytes()
            ).hexdigest(),
        }
    )
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory
