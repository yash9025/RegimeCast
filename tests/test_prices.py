import json

import numpy as np
import pandas as pd
import pytest

from regimecast.data.prices import (
    ASSETS,
    load_snapshot,
    sessions,
    snapshot,
    validate_prices,
)


@pytest.fixture
def prices():
    dates = sessions("2024-01-02", "2024-02-01")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "adjusted_close": 100 * np.exp(np.arange(len(dates)) * 0.001),
                }
            )
            for asset in ASSETS
        ],
        ignore_index=True,
    )


def test_holidays_and_ad_hoc_closures():
    assert pd.Timestamp("2024-01-15") not in sessions("2024-01-12", "2024-01-17")
    assert pd.Timestamp("2012-10-29") not in sessions("2012-10-26", "2012-11-02")


def test_valid_prices_and_immutable_roundtrip(prices, tmp_path):
    assert validate_prices(prices).rows == len(prices)
    first = snapshot(prices, tmp_path, provider="test", adjustment="synthetic")
    second = snapshot(prices, tmp_path, provider="test", adjustment="synthetic")
    assert first != second
    pd.testing.assert_frame_equal(load_snapshot(first), prices)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["historical_vintage"] is False
    with (first / "prices.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(first)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing",
        "negative",
        "infinite",
        "nan",
        "unsorted",
        "intraday",
        "timezone",
        "universe",
    ],
)
def test_reject_invalid_inputs(prices, kind):
    frame = prices.copy()
    if kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == "missing":
        frame = frame.drop(index=4)
    elif kind in {"negative", "infinite", "nan"}:
        frame.loc[0, "adjusted_close"] = {
            "negative": -1,
            "infinite": np.inf,
            "nan": np.nan,
        }[kind]
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "intraday":
        frame.loc[0, "date"] += pd.Timedelta(hours=1)
    elif kind == "timezone":
        frame.date = frame.date.dt.tz_localize("UTC")
    else:
        frame.loc[0, "asset"] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_prices(frame)


def test_requested_boundaries_catch_truncated_provider(prices):
    truncated = prices[prices.date > prices.date.min()]
    with pytest.raises(ValueError, match="coverage"):
        validate_prices(truncated, start="2024-01-02", end="2024-02-01")


def test_no_partial_snapshot_on_validation_failure(prices, tmp_path):
    with pytest.raises(ValueError):
        snapshot(
            prices.iloc[1:],
            tmp_path,
            provider="test",
            adjustment="test",
            start="2024-01-02",
            end="2024-02-01",
        )
    assert not list(tmp_path.iterdir())


def test_warnings_are_reported(prices):
    prices.loc[prices.asset.eq("SPY"), "adjusted_close"] = 100.0
    assert any("SPY" in warning for warning in validate_prices(prices).warnings)


def test_download_adapter_preserves_adjusted_prices(monkeypatch, prices):
    import yfinance

    from regimecast.data.prices import download_yahoo

    calls = []

    def fake_download(asset, **kwargs):
        calls.append(kwargs)
        data = prices[prices.asset.eq(asset)].set_index("date")
        return pd.DataFrame({"Adj Close": data.adjusted_close, "Close": 999.0})

    monkeypatch.setattr(yfinance, "download", fake_download)
    result = download_yahoo("2024-01-02", "2024-02-02")
    pd.testing.assert_frame_equal(result, prices)
    assert len(calls) == 4
    assert all(not call["auto_adjust"] and not call["threads"] for call in calls)


def test_csv_cli_roundtrip(monkeypatch, prices, tmp_path, capsys):
    import sys

    from regimecast.cli import main

    source = tmp_path / "prices.csv"
    prices.to_csv(source, index=False)
    output = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regimecast",
            "ingest",
            "--provider",
            "csv",
            "--csv",
            str(source),
            "--adjustment",
            "synthetic test fixture",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-02",
            "--output",
            str(output),
        ],
    )
    main()
    capsys.readouterr()
    directory = next((output / "raw").iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["source_sha256"]) == 64
    monkeypatch.setattr(sys, "argv", ["regimecast", "validate", str(directory)])
    main()
    assert json.loads(capsys.readouterr().out)["rows"] == len(prices)
