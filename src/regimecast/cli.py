"""Command line entry point; errors result in nonzero exit status."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from regimecast.data.prices import (
    ASSETS,
    download_yahoo,
    load_snapshot,
    read_csv,
    snapshot,
    validate_prices,
)


def main():
    logger = logging.getLogger("regimecast")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    parser = argparse.ArgumentParser(
        description="RegimeCast reproducible research pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Validate and cache a price snapshot")
    ingest.add_argument("--provider", choices=["csv", "yahoo"], required=True)
    ingest.add_argument("--csv", type=Path)
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end", required=True, help="Exclusive end date")
    ingest.add_argument(
        "--adjustment", help="Required description of CSV adjustment convention"
    )
    ingest.add_argument("--assets", nargs="+", default=list(ASSETS))
    ingest.add_argument("--output", type=Path, default=Path("data"))
    validate = commands.add_parser(
        "validate", help="Verify snapshot checksum and coverage"
    )
    validate.add_argument("snapshot", type=Path)
    backtest_parser = commands.add_parser(
        "backtest", help="Run price-only development baselines"
    )
    backtest_parser.add_argument("snapshot", type=Path)
    backtest_parser.add_argument("--start", default="2013-01-01")
    backtest_parser.add_argument(
        "--end", default="2024-01-01", help="Exclusive; holdout begins 2024"
    )
    backtest_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    backtest_parser.add_argument("--gate", choices=["gmm", "hmm"])
    backtest_parser.add_argument("--gate-cache", type=Path)
    backtest_parser.add_argument("--states", type=int, choices=[2, 3, 4], default=3)
    backtest_parser.add_argument("--macro-snapshot", type=Path)
    backtest_parser.add_argument(
        "--holdout", action="store_true", help="Run only the frozen holdout protocol"
    )
    macro_parser = commands.add_parser(
        "ingest-macro", help="Download historical FRED vintages using a local key"
    )
    macro_parser.add_argument("--start", default="2005-01-01")
    macro_parser.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    macro_parser.add_argument("--output", type=Path, default=Path("data"))
    analyze_parser = commands.add_parser(
        "analyze", help="Analyze matched prediction ledgers"
    )
    analyze_parser.add_argument("runs", nargs="+", type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    forecast_parser = commands.add_parser(
        "forecast", help="Prepare offline latest forecasts and saved model artifacts"
    )
    forecast_parser.add_argument("snapshot", type=Path)
    forecast_parser.add_argument("--macro-snapshot", type=Path)
    forecast_parser.add_argument("--gmm-cache", type=Path)
    forecast_parser.add_argument("--hmm-cache", type=Path)
    forecast_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    track_parser = commands.add_parser(
        "track", help="Register saved runs in local SQLite-backed MLflow"
    )
    track_parser.add_argument("runs", nargs="+", type=Path)
    track_parser.add_argument("--output", type=Path, default=Path("artifacts/mlflow"))
    args = parser.parse_args()
    try:
        if args.command == "track":
            from regimecast.tracking import track_run

            for run in args.runs:
                print(track_run(run, args.output))
            return
        if args.command == "forecast":
            from regimecast.production import prepare_forecasts

            print(
                prepare_forecasts(
                    args.snapshot,
                    args.output,
                    macro_snapshot=args.macro_snapshot,
                    gmm_cache=args.gmm_cache,
                    hmm_cache=args.hmm_cache,
                )
            )
            return
        if args.command == "ingest-macro":
            from regimecast.data.macro import download_vintages, save_vintages

            print(save_vintages(download_vintages(args.start, args.end), args.output))
            return
        if args.command == "analyze":
            from regimecast.evaluation.comparison import analyze, merge_ledgers

            analyze(
                merge_ledgers(args.runs),
                args.output,
                features=pd.read_parquet(args.runs[0] / "features.parquet"),
            )
            from regimecast.artifacts import register_analysis

            register_analysis(args.runs, args.output)
            print(args.output)
            return
        if args.command == "backtest":
            from regimecast.evaluation.walkforward import BacktestConfig
            from regimecast.pipeline import run_backtest

            print(
                run_backtest(
                    args.snapshot,
                    args.output,
                    BacktestConfig(start=args.start, end=args.end),
                    gate_kind=args.gate,
                    states=args.states,
                    macro_snapshot=args.macro_snapshot,
                    allow_holdout=args.holdout,
                    gate_cache=args.gate_cache,
                )
            )
            return
        if args.command == "validate":
            frame = load_snapshot(args.snapshot)
            print(
                json.dumps(
                    asdict(validate_prices(frame, tuple(frame.asset.unique()))),
                    indent=2,
                )
            )
            return
        if args.provider == "csv":
            if args.csv is None or not args.adjustment:
                parser.error("CSV ingestion requires --csv and --adjustment")
            frame = read_csv(args.csv)
            source_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
            adjustment = args.adjustment
        else:
            frame = download_yahoo(args.start, args.end, args.assets)
            source_hash = None
            adjustment = "Yahoo Adj Close; retrospective split/dividend adjustments"
        last_date = str((pd.Timestamp(args.end) - pd.Timedelta(days=1)).date())
        directory = snapshot(
            frame,
            args.output,
            provider=args.provider,
            adjustment=adjustment,
            start=args.start,
            end=last_date,
            assets=args.assets,
            source_sha256=source_hash,
        )
        print(directory)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"RegimeCast: {error}\n")


if __name__ == "__main__":
    main()
