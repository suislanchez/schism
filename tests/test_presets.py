"""Tests for the preset system."""

import json

import pytest

from schism.presets.manager import (
    delete_preset,
    get_preset,
    import_preset,
    list_presets,
    save_preset,
)


@pytest.fixture(autouse=True)
def clean_user_presets(tmp_path, monkeypatch):
    """Use a temp directory for user presets during tests."""
    test_presets = tmp_path / "presets"
    test_presets.mkdir()
    monkeypatch.setattr("schism.presets.manager.PRESETS_DIR", test_presets)
    yield test_presets


def test_list_presets_includes_defaults():
    presets = list_presets()
    names = [p["name"] for p in presets]
    assert "Creative" in names
    assert "Formal" in names
    assert "Sarcastic" in names
    assert "Concise" in names


def test_list_presets_defaults_have_source():
    presets = list_presets()
    for p in presets:
        if p["name"] in ["Creative", "Formal", "Sarcastic", "Concise"]:
            assert p["source"] == "default"


def test_get_preset_by_name():
    preset = get_preset("Creative")
    assert preset is not None
    assert "sliders" in preset
    assert "creativity" in preset["sliders"]


def test_get_preset_case_insensitive():
    preset = get_preset("creative")
    assert preset is not None
    assert preset["name"] == "Creative"


def test_get_preset_nonexistent():
    assert get_preset("nonexistent_preset_xyz") is None


def test_save_preset(clean_user_presets):
    data = {
        "name": "Test Preset",
        "description": "A test",
        "sliders": {"creativity": 0.5, "humor": -0.3},
    }
    path = save_preset(data)
    assert path.exists()

    # Should appear in list
    presets = list_presets()
    names = [p["name"] for p in presets]
    assert "Test Preset" in names


def test_save_preset_invalid():
    with pytest.raises(ValueError):
        save_preset({"name": "Bad"})  # Missing sliders

    with pytest.raises(ValueError):
        save_preset({"sliders": {}})  # Missing name


def test_delete_preset(clean_user_presets):
    save_preset(
        {
            "name": "Deletable",
            "sliders": {"humor": 0.9},
        }
    )
    assert get_preset("Deletable") is not None

    success = delete_preset("Deletable")
    assert success
    assert get_preset("Deletable") is None


def test_delete_nonexistent(clean_user_presets):
    assert delete_preset("nope") is False


def test_import_preset(clean_user_presets, tmp_path):
    preset_data = {
        "name": "Imported",
        "description": "From file",
        "sliders": {"confidence": 0.7},
    }
    file_path = tmp_path / "import_me.json"
    file_path.write_text(json.dumps(preset_data))

    result = import_preset(str(file_path))
    assert result["name"] == "Imported"
    assert get_preset("Imported") is not None


def test_preset_sliders_are_valid():
    """All default presets should have valid slider values between -1 and 1."""
    presets = list_presets()
    for p in presets:
        if p["source"] == "default":
            for feature, value in p["sliders"].items():
                assert -1.0 <= value <= 1.0, (
                    f"Preset '{p['name']}' has invalid value {value} for {feature}"
                )
