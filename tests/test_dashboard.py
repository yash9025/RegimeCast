from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_saved_research(artifact_catalog, monkeypatch):
    monkeypatch.setenv("REGIMECAST_CATALOG", str(artifact_catalog))
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "dashboard/app.py"
    ).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 4
    assert app.title[0].value == "One week ahead."
    assert len(app.metric) == 4
    assert any("DEMO" in warning.value for warning in app.warning)
