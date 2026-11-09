"""Historical FRED vintages with conservative next-session availability."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from regimecast.data.prices import sessions

SERIES = ("DGS2", "DGS10", "UNRATE", "CPIAUCSL")


def fred_key(env_file=Path(".env")):
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "FRED_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise ValueError(
            "Configure FRED_API_KEY in your environment or local .env file"
        )
    return key


def parse_sparse_observations(observations, series):
    records = []
    for observation in observations:
        for name, value in observation.items():
            if name.startswith(series + "_"):
                records.append(
                    {
                        "series": series,
                        "observation_date": pd.Timestamp(observation["date"]),
                        "vintage_date": pd.to_datetime(
                            name[len(series) + 1 :], format="%Y%m%d"
                        ),
                        "value": np.nan if value == "." else float(value),
                    }
                )
    return records


def download_vintages(start="2005-01-01", end="2026-11-09", series=SERIES):
    key = fred_key()
    records = []
    for name in series:
        batch_start = pd.Timestamp(start)
        final = pd.Timestamp(end)
        while batch_start <= final:
            batch_end = min(
                batch_start + pd.DateOffset(years=4) - pd.Timedelta("1D"), final
            )
            offset = 0
            while True:
                parameters = {
                    "api_key": key,
                    "file_type": "json",
                    "series_id": name,
                    "observation_start": start,
                    "observation_end": end,
                    "realtime_start": str(batch_start.date()),
                    "realtime_end": str(batch_end.date()),
                    "output_type": 3,
                    "limit": 100000,
                    "offset": offset,
                }
                try:
                    response = requests.get(
                        "https://api.stlouisfed.org/fred/series/observations",
                        params=parameters,
                        timeout=60,
                    )
                except requests.RequestException:
                    raise ValueError(
                        f"FRED transport failure for {name}; retry later"
                    ) from None
                if response.status_code != 200:
                    raise ValueError(
                        f"FRED rejected {name} request (HTTP {response.status_code})"
                    )
                payload = response.json()
                observations = payload.get("observations", [])
                records.extend(parse_sparse_observations(observations, name))
                offset += len(observations)
                if not observations or offset >= int(payload["count"]):
                    break
            batch_start = batch_end + pd.Timedelta("1D")
    if not records:
        raise ValueError("FRED returned no historical vintage changes")
    result = pd.DataFrame(records)
    validate_vintages(result)
    return result


def validate_vintages(frame):
    required = {"series", "observation_date", "vintage_date", "value"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Missing vintage schema or observations")
    if frame[list(required - {"value"})].isna().any().any():
        raise ValueError("Missing vintage identifiers")
    if frame.duplicated(["series", "observation_date", "vintage_date"]).any():
        raise ValueError("Duplicate vintage keys")
    if (frame.vintage_date < frame.observation_date).any():
        raise ValueError("Vintage cannot precede its observation period")
    if np.isinf(pd.to_numeric(frame.value, errors="raise")).any():
        raise ValueError("Infinite macro observations")


def save_vintages(frame, root=Path("data"), provider="FRED/ALFRED"):
    validate_vintages(frame)
    now = datetime.now(UTC)
    directory = (
        root / "macro" / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "vintages.parquet"
    frame.to_parquet(path, index=False)
    manifest = {
        "provider": provider,
        "downloaded_at": now.isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "series": sorted(frame.series.unique().tolist()),
        "availability": "First XNYS session strictly after vintage date; no intraday release timing assumed",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return directory


def load_vintages(directory):
    path = directory / "vintages.parquet"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Macro snapshot checksum mismatch")
    frame = pd.read_parquet(path)
    validate_vintages(frame)
    return frame


def macro_features(vintages, dates):
    """Keep a historical observation dictionary; old revisions cannot replace newer periods."""
    validate_vintages(vintages)
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    calendar = sessions(
        str(min(vintages.vintage_date.min(), dates.min()).date()),
        str(
            (max(vintages.vintage_date.max(), dates.max()) + pd.Timedelta("14D")).date()
        ),
    )
    result = pd.DataFrame(index=dates)
    for series in sorted(vintages.series.unique()):
        events = (
            vintages[vintages.series.eq(series)]
            .sort_values(["vintage_date", "observation_date"])
            .copy()
        )
        positions = calendar.searchsorted(
            pd.DatetimeIndex(events.vintage_date), side="right"
        )
        events["available_session"] = calendar[positions]
        records = list(events.itertuples(index=False))
        known, offset, levels, ages, growth = {}, 0, [], [], []
        for date in dates:
            while offset < len(records) and records[offset].available_session <= date:
                row = records[offset]
                known[row.observation_date] = (row.value, row.available_session)
                offset += 1
            if not known:
                levels.append(np.nan)
                ages.append(np.nan)
                growth.append(np.nan)
                continue
            latest = max(known)
            value, availability = known[latest]
            levels.append(value)
            ages.append((date - availability).days)
            previous = known.get(latest - pd.DateOffset(years=1), (np.nan, None))[0]
            growth.append(
                100 * (value / previous - 1)
                if previous and np.isfinite(previous)
                else np.nan
            )
        prefix = f"macro_{series}"
        result[prefix] = levels
        result[prefix + "_age_days"] = ages
        result[prefix + "_missing"] = result[prefix].isna().astype(float)
        result[prefix + "_change_21"] = result[prefix].diff(21)
        if series == "CPIAUCSL":
            result[prefix + "_yoy_pct"] = growth
    if {"macro_DGS2", "macro_DGS10"}.issubset(result):
        result["macro_curve"] = result.macro_DGS10 - result.macro_DGS2
        for window in (5, 21):
            result[f"macro_curve_change_{window}"] = result.macro_curve.diff(window)
    return result.rename_axis("date").reset_index()
