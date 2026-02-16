"""Tests for the steering engine (unit tests with mocked models)."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from schism.engine.steering import (
    extract_activations,
    extract_steering_vector_contrastive,
    apply_steering,
    _vector_cache_key,
    save_vector,
    load_cached_vector,
)
from schism.engine.loader import VECTORS_DIR


@pytest.fixture(autouse=True)
def clean_vectors(tmp_path, monkeypatch):
    test_vectors = tmp_path / "vectors"
    test_vectors.mkdir()
    monkeypatch.setattr("schism.engine.steering.VECTORS_DIR", test_vectors)
    monkeypatch.setattr("schism.engine.loader.VECTORS_DIR", test_vectors)
    yield test_vectors


def _make_mock_model(hidden_dim=256, num_layers=24):
    """Create a minimal mock model for testing."""
    model = MagicMock()
    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        hooks = []

        def register_hook(fn, _hooks=hooks):
            _hooks.append(fn)
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle

        layer.register_forward_hook = register_hook
        layer._hooks = hooks
        layers.append(layer)

    model.model.layers = layers
    return model


def _make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token_id = 2
    return tokenizer


def test_vector_cache_key_deterministic():
    key1 = _vector_cache_key("model", "feat", ["a", "b"], ["c", "d"])
    key2 = _vector_cache_key("model", "feat", ["a", "b"], ["c", "d"])
    assert key1 == key2


def test_vector_cache_key_order_independent():
    key1 = _vector_cache_key("model", "feat", ["b", "a"], ["d", "c"])
    key2 = _vector_cache_key("model", "feat", ["a", "b"], ["c", "d"])
    assert key1 == key2


def test_vector_cache_key_different_for_different_inputs():
    key1 = _vector_cache_key("model", "feat", ["a"], ["b"])
    key2 = _vector_cache_key("model", "feat", ["x"], ["y"])
    assert key1 != key2


def test_save_and_load_vector(clean_vectors):
    vec = torch.randn(256)
    save_vector(vec, "test_model", "test_feat", "abc123")

    loaded = load_cached_vector("test_model", "test_feat", "abc123")
    assert loaded is not None
    assert torch.allclose(vec, loaded)


def test_load_nonexistent_vector(clean_vectors):
    assert load_cached_vector("nope", "nope", "nope") is None


def test_apply_steering_context_manager():
    """Test that apply_steering properly adds and removes hooks."""
    model = _make_mock_model()
    vec = torch.randn(256)

    layer = model.model.layers[12]
    with apply_steering(model, [(vec, 0.5)], layer_idx=12):
        assert layer.register_forward_hook.called

    # Hook handle.remove should have been called on exit


def test_apply_steering_empty_vectors():
    """No-op when no vectors provided."""
    model = _make_mock_model()
    with apply_steering(model, [], layer_idx=12):
        pass  # Should not raise


def test_apply_steering_multiple_vectors():
    """Multiple vectors should all get applied."""
    model = _make_mock_model()
    vec1 = torch.randn(256)
    vec2 = torch.randn(256)

    with apply_steering(model, [(vec1, 0.3), (vec2, 0.7)], layer_idx=12):
        assert model.model.layers[12].register_forward_hook.called
