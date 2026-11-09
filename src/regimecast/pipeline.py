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


def run_backtest(
    snapshot: Path, output: Path, config: BacktestConfig, *, gate_kind=None, states=3
):
    if config.end > "2024-01-01":
        raise ValueError(
            "2024 onward is locked holdout; development runs must end by 2024-01-01"
        )
    prices = load_snapshot(snapshot)
    features, columns = make_features(prices)
    gate = None
    gate_diagnostics = []
    if gate_kind:
        from regimecast.models.regimes import causal_probabilities, market_state

        state = market_state(prices)
        state = state[state.index < config.end]
        gate, gate_diagnostics = causal_probabilities(
            state, kind=gate_kind, states=states, seed=config.seed
        )
    ledger, metadata = backtest(
        features, columns, config, gates=None if gate is None else {gate_kind: gate}
    )
    now = datetime.now(UTC)
    directory = output / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    if gate is not None:
        gate.to_parquet(directory / "regime_probabilities.parquet", index=False)
        (directory / "regime_diagnostics.json").write_text(
            json.dumps(gate_diagnostics, indent=2), encoding="utf-8"
        )
        metadata["gate"] = {
            "kind": gate_kind,
            "states": states,
            "seed": config.seed,
            "warmup": 126,
            "window": 1260,
        }
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
