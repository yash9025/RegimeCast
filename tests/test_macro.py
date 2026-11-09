import numpy as np
import pandas as pd
import pytest

from regimecast.data.macro import fred_key, macro_features, validate_vintages


def test_revisions_are_available_only_after_release_and_preserve_newer_periods():
    records = pd.DataFrame(
        {
            "series": ["UNRATE"] * 3,
            "observation_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01"]
            ),
            "vintage_date": pd.to_datetime(["2024-02-02", "2024-03-01", "2024-03-05"]),
            "value": [3.5, 3.7, 99.0],
        }
    )
    dates = pd.bdate_range("2024-02-01", "2024-03-08")
    features = macro_features(records, dates).set_index("date")
    assert np.isnan(features.loc["2024-02-02", "macro_UNRATE"])
    assert features.loc["2024-02-05", "macro_UNRATE"] == 3.5
    assert features.loc["2024-03-01", "macro_UNRATE"] == 3.5
    assert features.loc["2024-03-04", "macro_UNRATE"] == 3.7
    assert features.loc["2024-03-06", "macro_UNRATE"] == 3.7
    changed = records.copy()
    changed.loc[2, "value"] = -100
    pd.testing.assert_frame_equal(
        macro_features(records, dates), macro_features(changed, dates)
    )


def test_secret_loading_and_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    path = tmp_path / ".env"
    with pytest.raises(ValueError, match="Configure FRED_API_KEY"):
        fred_key(path)
    path.write_text('FRED_API_KEY="test-only-placeholder"\n', encoding="utf-8")
    assert fred_key(path) == "test-only-placeholder"
    monkeypatch.setenv("FRED_API_KEY", "environment-placeholder")
    assert fred_key(path) == "environment-placeholder"


def test_reject_vintage_before_observation():
    frame = pd.DataFrame(
        {
            "series": ["UNRATE"],
            "observation_date": pd.to_datetime(["2024-01-01"]),
            "vintage_date": pd.to_datetime(["2023-01-01"]),
            "value": [3.0],
        }
    )
    with pytest.raises(ValueError, match="precede"):
        validate_vintages(frame)


def test_sparse_vintage_schema_preserves_true_release_dates():
    from regimecast.data.macro import parse_sparse_observations

    rows = parse_sparse_observations(
        [
            {"date": "2005-01-03"},
            {"date": "2005-06-27", "DGS2_20050706": "3.600", "DGS2_20050707": "."},
        ],
        "DGS2",
    )
    assert len(rows) == 2
    assert rows[0]["vintage_date"] == pd.Timestamp("2005-07-06")
    assert rows[0]["value"] == 3.6
    assert np.isnan(rows[1]["value"])


def test_latest_observation_revision_is_not_applied_early():
    frame = pd.DataFrame(
        {
            "series": ["UNRATE"] * 2,
            "observation_date": pd.to_datetime(["2024-01-01"] * 2),
            "vintage_date": pd.to_datetime(["2024-02-02", "2024-02-09"]),
            "value": [3.5, 3.6],
        }
    )
    features = macro_features(
        frame, pd.bdate_range("2024-02-01", "2024-02-13")
    ).set_index("date")
    assert features.loc["2024-02-09", "macro_UNRATE"] == 3.5
    assert features.loc["2024-02-12", "macro_UNRATE"] == 3.6


def test_macro_refresh_default_uses_current_utc_date(monkeypatch):
    from datetime import UTC, datetime

    from regimecast.data import macro

    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 2, tzinfo=UTC)

    class Response:
        status_code = 200

        def json(self):
            return {
                "count": 1,
                "observations": [{"date": "2025-06-01", "UNRATE_20250706": "4.0"}],
            }

    requests = []

    def get(url, params, timeout):
        requests.append(params.copy())
        return Response()

    monkeypatch.setattr(macro, "datetime", Clock)
    monkeypatch.setattr(macro, "fred_key", lambda: "synthetic-placeholder")
    monkeypatch.setattr(macro.requests, "get", get)
    macro.download_vintages(start="2025-01-01", series=("UNRATE",))
    assert requests[0]["observation_end"] == "2026-01-02"
    assert requests[0]["realtime_end"] == "2026-01-02"
