"""Price ingestion with explicit session coverage and immutable provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ASSETS = ("SPY", "QQQ", "TLT", "GLD")


@dataclass
class ValidationReport:
    rows: int
    assets: list[str]
    start: str
    end: str
    warnings: list[str]


def sessions(start: str, end: str) -> pd.DatetimeIndex:
    """Inclusive XNYS session labels, including ad-hoc closures."""
    calendar = xcals.get_calendar(
        "XNYS",
        start=pd.Timestamp(start) - pd.Timedelta("14D"),
        end=pd.Timestamp(end) + pd.Timedelta("14D"),
    )
    return calendar.sessions_in_range(start, end).tz_localize(None)


def validate_prices(
    frame: pd.DataFrame, assets=ASSETS, *, start=None, end=None, min_sessions=6
) -> ValidationReport:
    required = {"date", "asset", "adjusted_close"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("No prices supplied")
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Dates must be nonmissing timezone-naive session labels")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError(
            "Dates must be midnight session labels, not intraday timestamps"
        )
    if frame.asset.isna().any() or set(frame.asset) != set(assets):
        raise ValueError("Asset universe does not match the requested assets")
    if frame.assign(date=dates).duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate asset/date keys")
    price = pd.to_numeric(frame.adjusted_close, errors="coerce")
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")
    start = pd.Timestamp(start) if start is not None else dates.min()
    end = pd.Timestamp(end) if end is not None else dates.max()
    if start > end:
        raise ValueError("Start must not follow end")
    expected = sessions(str(start.date()), str(end.date()))
    if len(expected) < min_sessions:
        raise ValueError(f"Need at least {min_sessions} market sessions")
    warnings = []
    for asset in assets:
        mask = frame.asset.eq(asset)
        observed = pd.DatetimeIndex(dates[mask])
        if not observed.is_monotonic_increasing:
            raise ValueError(f"Unsorted dates for {asset}")
        if not observed.equals(expected):
            missing = expected.difference(observed)
            extra = observed.difference(expected)
            raise ValueError(
                f"Session coverage mismatch for {asset}: "
                f"{len(missing)} missing, {len(extra)} unexpected"
            )
        values = price[mask].reset_index(drop=True)
        if values.pct_change().abs().gt(0.3).any():
            warnings.append(f"{asset}: adjusted daily move above 30%; inspect source")
        if values.diff().eq(0).rolling(5).sum().ge(5).any():
            warnings.append(
                f"{asset}: five consecutive unchanged prices; inspect staleness"
            )
    return ValidationReport(
        len(frame), list(assets), str(start.date()), str(end.date()), warnings
    )


def read_csv(path: Path) -> pd.DataFrame:
    """Strict canonical CSV: date,asset,adjusted_close. No filling or sorting."""
    return pd.read_csv(path, parse_dates=["date"])


def download_yahoo(start: str, end: str, assets=ASSETS) -> pd.DataFrame:
    """End is exclusive, as in Yahoo's API. Download each symbol explicitly."""
    import yfinance as yf

    frames = []
    for asset in assets:
        raw = yf.download(
            asset,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            multi_level_index=False,
            timeout=30,
        )
        if raw is None or raw.empty or "Adj Close" not in raw:
            raise ValueError(f"No adjusted prices returned for {asset}")
        index = pd.DatetimeIndex(raw.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frames.append(
            pd.DataFrame(
                {
                    "date": index,
                    "asset": asset,
                    "adjusted_close": raw["Adj Close"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def snapshot(
    frame: pd.DataFrame,
    root: Path,
    *,
    provider: str,
    adjustment: str,
    start=None,
    end=None,
    assets=ASSETS,
    source_sha256: str | None = None,
) -> Path:
    """Persist a validated normalized raw snapshot; never overwrite a run."""
    report = validate_prices(frame, assets, start=start, end=end)
    now = datetime.now(UTC)
    import uuid

    directory = (
        root / "raw" / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "prices.parquet"
    frame.to_parquet(path, index=False)
    manifest = {
        "schema_version": 1,
        "provider": provider,
        "downloaded_at": now.isoformat(),
        "adjustment": adjustment,
        "historical_vintage": False,
        "availability": "Session close prices retrieved retrospectively; not a point-in-time archive",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "validation": asdict(report),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return directory


def load_snapshot(directory: Path) -> pd.DataFrame:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "prices.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    frame = pd.read_parquet(path)
    report = manifest["validation"]
    validate_prices(frame, report["assets"], start=report["start"], end=report["end"])
    return frame
