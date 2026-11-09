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
    snapshot: Path,
    output: Path,
    config: BacktestConfig,
    *,
    gate_kind=None,
    states=3,
    macro_snapshot=None,
    allow_holdout=False,
):
    protocol = None
    if config.end > "2024-01-01":
        if not allow_holdout:
            raise ValueError(
                "2024 onward is locked holdout; development runs must end by 2024-01-01"
            )
        protocol_path = Path("configs/holdout-protocol.json")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = (
            protocol["holdout_start"],
            protocol["holdout_end_exclusive"],
            protocol["train_sessions"],
            protocol["seed"],
            protocol["states"],
        )
        actual = (config.start, config.end, config.train_sessions, config.seed, states)
        if (
            actual != expected
            or config.min_train_sessions != protocol["train_sessions"]
        ):
            raise ValueError("Holdout configuration differs from the frozen protocol")
    source_metadata = {}
    try:
        source_metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        source_metadata["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        source_metadata["git_commit"] = None
    source_metadata["source_sha256"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))
    ).hexdigest()
    prices = load_snapshot(snapshot)
    features, columns = make_features(prices)
    macro_manifest = None
    if macro_snapshot is not None:
        from regimecast.data.macro import load_vintages, macro_features

        macro = macro_features(load_vintages(macro_snapshot), features.date)
        macro_columns = [c for c in macro if c != "date"]
        features = features.merge(macro, on="date", how="left", validate="many_to_one")
        columns = columns + macro_columns
        macro_manifest = json.loads(
            (macro_snapshot / "manifest.json").read_text(encoding="utf-8")
        )
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
    if macro_manifest is not None:
        ledger["model"] = ledger.model.where(
            ledger.model.isin(["historical", "ewma", "har"]), ledger.model + "_macro"
        )
        metadata["macro"] = {
            "sha256": macro_manifest["sha256"],
            "availability": macro_manifest["availability"],
            "gate_features": "price-only, held fixed for controlled macro forecasting ablation",
        }
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
    metadata.update(source_metadata)
    if protocol is not None:
        metadata["evaluation_kind"] = "locked chronological holdout; frozen defaults"
        metadata["holdout_protocol"] = protocol
        metadata["protocol_sha256"] = hashlib.sha256(
            Path("configs/holdout-protocol.json").read_bytes()
        ).hexdigest()
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory
