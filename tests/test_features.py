"""Tests for the feature system."""

import json
import pytest

from schism.engine.features import (
    get_default_features,
    get_feature_names,
    get_feature_info,
    get_all_features,
    save_custom_feature,
    load_custom_features,
    DEFAULT_FEATURES,
)
from schism.engine.loader import FEATURES_DIR


@pytest.fixture(autouse=True)
def clean_features(tmp_path, monkeypatch):
    test_features = tmp_path / "features"
    test_features.mkdir()
    monkeypatch.setattr("schism.engine.features.FEATURES_DIR", test_features)
    monkeypatch.setattr("schism.engine.loader.FEATURES_DIR", test_features)
    yield test_features


def test_default_features_exist():
    features = get_default_features()
    expected = {"creativity", "formality", "humor", "confidence", "verbosity", "empathy", "technical"}
    assert expected == set(features.keys())


def test_feature_names_sorted():
    names = get_feature_names()
    assert names == sorted(names)


def test_each_feature_has_prompts():
    for name, feat in DEFAULT_FEATURES.items():
        assert "positive" in feat, f"{name} missing positive prompts"
        assert "negative" in feat, f"{name} missing negative prompts"
        assert "description" in feat, f"{name} missing description"
        assert len(feat["positive"]) >= 3, f"{name} needs at least 3 positive prompts"
        assert len(feat["negative"]) >= 3, f"{name} needs at least 3 negative prompts"


def test_get_feature_info():
    info = get_feature_info("creativity")
    assert info is not None
    assert "description" in info


def test_get_feature_info_nonexistent():
    assert get_feature_info("nonexistent") is None


def test_save_and_load_custom_feature(clean_features):
    custom = {
        "description": "How poetic the response is",
        "positive": ["Be very poetic", "Use beautiful imagery"],
        "negative": ["Be prosaic", "Use plain language"],
    }
    save_custom_feature("poetry", custom)

    loaded = load_custom_features()
    assert "poetry" in loaded
    assert loaded["poetry"]["description"] == "How poetic the response is"


def test_get_all_features_merges_custom(clean_features):
    custom = {
        "description": "Test feature",
        "positive": ["pos1"],
        "negative": ["neg1"],
    }
    save_custom_feature("custom_test", custom)

    all_features = get_all_features()
    assert "creativity" in all_features  # default
    assert "custom_test" in all_features  # custom


def test_positive_negative_prompts_are_strings():
    for name, feat in DEFAULT_FEATURES.items():
        for p in feat["positive"]:
            assert isinstance(p, str), f"{name}: positive prompt is not a string"
        for p in feat["negative"]:
            assert isinstance(p, str), f"{name}: negative prompt is not a string"
