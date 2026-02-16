"""Preset management - load, save, list, and share personality configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from schism.engine.loader import SCHISM_HOME

PRESETS_DIR = SCHISM_HOME / "presets"
DEFAULTS_DIR = Path(__file__).parent / "defaults"


def ensure_presets_dir():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_preset(data: dict) -> bool:
    """Validate preset JSON structure."""
    required = {"name", "sliders"}
    return required.issubset(data.keys()) and isinstance(data["sliders"], dict)


def list_presets() -> list[dict]:
    """List all available presets (defaults + user-created)."""
    presets = []

    # Load built-in defaults
    if DEFAULTS_DIR.exists():
        for path in sorted(DEFAULTS_DIR.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
                data["source"] = "default"
                data["path"] = str(path)
                presets.append(data)

    # Load user presets
    ensure_presets_dir()
    for path in sorted(PRESETS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
            data["source"] = "user"
            data["path"] = str(path)
            presets.append(data)

    return presets


def get_preset(name: str) -> Optional[dict]:
    """Get a specific preset by name."""
    for preset in list_presets():
        if preset["name"].lower() == name.lower():
            return preset
    return None


def save_preset(data: dict) -> Path:
    """Save a new user preset."""
    if not _validate_preset(data):
        raise ValueError("Invalid preset: must have 'name' and 'sliders' fields")

    ensure_presets_dir()
    filename = data["name"].lower().replace(" ", "_") + ".json"
    path = PRESETS_DIR / filename

    # Don't include transient fields
    save_data = {
        "name": data["name"],
        "description": data.get("description", ""),
        "model": data.get("model", "any"),
        "sliders": data["sliders"],
        "version": data.get("version", 1),
    }

    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)

    return path


def delete_preset(name: str) -> bool:
    """Delete a user preset. Cannot delete defaults."""
    ensure_presets_dir()
    for path in PRESETS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
            if data.get("name", "").lower() == name.lower():
                path.unlink()
                return True
    return False


def import_preset(file_path: str) -> dict:
    """Import a preset from an external JSON file."""
    with open(file_path) as f:
        data = json.load(f)

    if not _validate_preset(data):
        raise ValueError("Invalid preset file")

    save_preset(data)
    return data
