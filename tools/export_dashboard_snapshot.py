"""Export public display artifacts; excludes credentials, raw prices and models."""
import argparse
import json
from pathlib import Path

import pandas as pd


def export(catalog_path, output):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    latest = json.loads(Path(catalog["latest_bundle"]).read_text(encoding="utf-8"))
    fields = ["schema_version", "demo", "as_of", "target_end", "horizon_sessions",
              "forecast_cutoff", "forecasts", "regimes", "expert_contributions", "feature_context"]
    public = {k: latest[k] for k in fields if k in latest}
    public["model_metadata"] = {k: latest.get("model_metadata", {})[k] for k in
        ["refit_date", "train_start", "train_end", "latest_training_label", "states", "seed", "versions", "model_sha256"]
        if k in latest.get("model_metadata", {})}
    public["data_metadata"] = {k: latest.get("data_metadata", {})[k] for k in
        ["provider", "downloaded_at", "price_sha256", "macro_sha256", "historical_price_vintage", "adjustment", "macro_availability"]
        if k in latest.get("data_metadata", {})}
    public["publication_note"] = "Published research snapshot · Saved forecasts, not a live market feed."
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(public, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    histories = {}
    for kind, path in catalog.get("regime_history", {}).items():
        frame = pd.read_parquet(path).tail(504)
        columns = [c for c in frame if c == "date" or c.startswith("regime_") or c == "gate_reliable"]
        name = f"regimes_{kind}.parquet"
        frame[columns].to_parquet(output / name, index=False)
        histories[kind] = name
    # These aggregate evaluation reports are already published in the repository.
    settings = {
        "relative_paths": True, "latest_bundle": "latest.json",
        "regime_history": histories,
        "evaluation_runs": {"development": [], "holdout": []},
        "metrics": {"development": "../../reports/macro-analysis/per_asset_metrics.csv",
                    "holdout": "../../reports/holdout-analysis/per_asset_metrics.csv"},
        "analysis": {"development": "../../reports/macro-analysis",
                     "holdout": "../../reports/holdout-analysis"},
    }
    (output / "catalog.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    print(f"Published snapshot: {output} (as of {public['as_of']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("dashboard/snapshot"))
    args = parser.parse_args()
    export(args.catalog, args.output)
