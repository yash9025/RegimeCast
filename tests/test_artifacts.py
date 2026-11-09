import json

import pytest

from regimecast.artifacts import matching_analyses, register_analysis


def test_analysis_registry_never_serves_mismatched_data(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    metadata = {
        "data_sha256": "price-A",
        "macro": {"sha256": "macro-A"},
        "config": {"start": "2013-01-01", "end": "2024-01-01"},
    }
    (run / "run.json").write_text(json.dumps(metadata))
    output = tmp_path / "analysis"
    output.mkdir()
    root = tmp_path / "artifacts"
    register_analysis([run], output, root / "analysis-registry.json")
    assert matching_analyses(root, "price-A", "macro-A", tmp_path / "absent")[
        "development"
    ] == str(output.resolve())
    assert matching_analyses(root, "price-B", "macro-A", tmp_path / "absent") == {}
    assert matching_analyses(root, "price-A", "macro-B", tmp_path / "absent") == {}
    other = tmp_path / "other"
    other.mkdir()
    metadata["data_sha256"] = "price-B"
    (other / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="share data versions"):
        register_analysis([run, other], output, root / "analysis-registry.json")
