"""Core steering logic: extract vectors, apply hooks at inference time."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Optional

import torch

from schism.engine.loader import VECTORS_DIR, ensure_dirs


def _get_layer(model, layer_idx: int):
    """Get a specific transformer layer from the model, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError("Unsupported model architecture - cannot find layers")


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Run prompts through model, capture activations at a specific layer.

    Returns tensor of shape (num_prompts, hidden_dim) - mean-pooled over tokens.
    """
    all_activations = []

    for prompt in prompts:
        captured = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured["act"] = output[0].detach()
            else:
                captured["act"] = output.detach()

        layer = _get_layer(model, layer_idx)
        handle = layer.register_forward_hook(hook_fn)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Mean pool over token positions (excluding padding)
        act = captured["act"]  # (1, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled = (act * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
        all_activations.append(pooled.squeeze(0))

    return torch.stack(all_activations)  # (num_prompts, hidden_dim)


def extract_steering_vector_sae(
    model,
    tokenizer,
    sae,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Extract a steering vector using SAE-based feature decomposition (FGAA-style).

    1. Get activations for positive and negative prompts
    2. Encode through SAE to get sparse features
    3. Compute mean difference in feature space
    4. Decode back to activation space
    """
    pos_acts = extract_activations(model, tokenizer, positive_prompts, layer_idx, device)
    neg_acts = extract_activations(model, tokenizer, negative_prompts, layer_idx, device)

    # Encode into SAE feature space
    pos_features = sae.encode(pos_acts)
    neg_features = sae.encode(neg_acts)

    # Mean difference in feature space
    feature_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)

    # Decode back to activation space
    steering_vector = sae.decode(feature_diff.unsqueeze(0)).squeeze(0)

    return steering_vector


def extract_steering_vector_contrastive(
    model,
    tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Extract a steering vector using simple contrastive activation addition (CAA).

    Used when no SAE is available (e.g., Llama models).
    Computes mean activation difference directly in activation space.
    """
    pos_acts = extract_activations(model, tokenizer, positive_prompts, layer_idx, device)
    neg_acts = extract_activations(model, tokenizer, negative_prompts, layer_idx, device)

    steering_vector = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)

    # Normalize to unit length for consistent scaling
    steering_vector = steering_vector / steering_vector.norm()

    return steering_vector


def extract_steering_vector(
    model,
    tokenizer,
    sae,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Extract a steering vector, using SAE if available, else contrastive."""
    if sae is not None:
        return extract_steering_vector_sae(
            model,
            tokenizer,
            sae,
            positive_prompts,
            negative_prompts,
            layer_idx,
            device,
        )
    return extract_steering_vector_contrastive(
        model,
        tokenizer,
        positive_prompts,
        negative_prompts,
        layer_idx,
        device,
    )


@contextmanager
def apply_steering(
    model,
    vectors: list[tuple[torch.Tensor, float]],
    layer_idx: int,
):
    """Context manager that applies steering vectors during inference.

    Args:
        model: The language model
        vectors: List of (steering_vector, alpha) tuples
        layer_idx: Which layer to hook into
    """
    if not vectors:
        yield
        return

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            modified = hidden.clone()
            for vec, alpha in vectors:
                v = vec.to(hidden.device).to(hidden.dtype)
                modified = modified + alpha * v.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            modified = output.clone()
            for vec, alpha in vectors:
                v = vec.to(output.device).to(output.dtype)
                modified = modified + alpha * v.unsqueeze(0).unsqueeze(0)
            return modified

    layer = _get_layer(model, layer_idx)
    handle = layer.register_forward_hook(steering_hook)
    try:
        yield
    finally:
        handle.remove()


def _vector_cache_key(
    model_name: str,
    feature_name: str,
    positive_prompts: list[str],
    negative_prompts: list[str],
) -> str:
    """Generate a deterministic cache key for a steering vector."""
    data = json.dumps(
        {
            "model": model_name,
            "feature": feature_name,
            "positive": sorted(positive_prompts),
            "negative": sorted(negative_prompts),
        },
        sort_keys=True,
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def save_vector(vector: torch.Tensor, model_name: str, feature_name: str, cache_key: str):
    """Save a computed steering vector to disk."""
    ensure_dirs()
    path = VECTORS_DIR / f"{model_name}_{feature_name}_{cache_key}.pt"
    torch.save(vector.cpu(), path)
    return path


def load_cached_vector(
    model_name: str, feature_name: str, cache_key: str
) -> Optional[torch.Tensor]:
    """Load a cached steering vector if it exists."""
    path = VECTORS_DIR / f"{model_name}_{feature_name}_{cache_key}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    return None
