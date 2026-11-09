"""Version-aware registration of research analyses for the local dashboard."""

import json
from pathlib import Path


def register_analysis(
    runs, directory, registry=Path("artifacts/analysis-registry.json")
):
    metadata = [
        json.loads((Path(run) / "run.json").read_text(encoding="utf-8")) for run in runs
    ]
    prices = {m["data_sha256"] for m in metadata}
    macros = {m["macro"]["sha256"] for m in metadata if "macro" in m}
    ranges = {(m["config"]["start"], m["config"]["end"]) for m in metadata}
    if len(prices) != 1 or len(macros) > 1 or len(ranges) != 1:
        raise ValueError("Analysis runs must share data versions and evaluation dates")
    period = {
        ("2013-01-01", "2024-01-01"): "development",
        ("2024-01-01", "2026-01-01"): "holdout",
    }.get(next(iter(ranges)), "custom")
    entry = {
        "price_sha256": next(iter(prices)),
        "macro_sha256": next(iter(macros)) if macros else None,
        "period": period,
        "path": str(Path(directory).resolve()),
    }
    registry = Path(registry)
    previous = (
        json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    )
    previous = [e for e in previous if e["path"] != entry["path"]] + [entry]
    registry.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    temporary.replace(registry)
    return entry


def matching_analyses(
    root,
    price_sha256,
    macro_sha256=None,
    release_catalog=Path("configs/analysis-catalog.json"),
):
    entries = []
    for registry in [Path(release_catalog), Path(root) / "analysis-registry.json"]:
        if registry.exists():
            entries += json.loads(registry.read_text(encoding="utf-8"))
    return {
        e["period"]: str(Path(e["path"]).resolve())
        for e in entries
        if e["price_sha256"] == price_sha256
        and e.get("macro_sha256") == macro_sha256
        and Path(e["path"]).exists()
    }
