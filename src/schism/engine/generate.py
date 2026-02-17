"""Text generation with steering hooks applied."""

from __future__ import annotations

from typing import AsyncIterator

import torch

from schism.engine.features import get_all_features
from schism.engine.loader import load_model
from schism.engine.steering import (
    _vector_cache_key,
    apply_steering,
    extract_steering_vector,
    load_cached_vector,
    save_vector,
)


def _get_or_compute_vector(
    model_name: str,
    feature_name: str,
    model_data: dict,
) -> torch.Tensor:
    """Get a cached steering vector or compute it fresh."""
    features = get_all_features()
    if feature_name not in features:
        raise ValueError(f"Unknown feature: {feature_name}")

    feature = features[feature_name]
    cache_key = _vector_cache_key(
        model_name,
        feature_name,
        feature["positive"],
        feature["negative"],
    )

    # Try cache first
    cached = load_cached_vector(model_name, feature_name, cache_key)
    if cached is not None:
        return cached

    # Compute fresh
    vector = extract_steering_vector(
        model=model_data["model"],
        tokenizer=model_data["tokenizer"],
        sae=model_data["sae"],
        positive_prompts=feature["positive"],
        negative_prompts=feature["negative"],
        layer_idx=model_data["config"]["steering_layer"],
        device=model_data["device"],
    )

    save_vector(vector, model_name, feature_name, cache_key)
    return vector


def build_steering_vectors(
    model_name: str,
    sliders: dict[str, float],
    model_data: dict,
) -> list[tuple[torch.Tensor, float]]:
    """Build list of (vector, alpha) tuples from slider config.

    Args:
        model_name: Name of the model
        sliders: Dict of {feature_name: alpha_value} where alpha is -1.0 to 1.0
        model_data: Loaded model data from loader
    """
    vectors = []
    for feature_name, alpha in sliders.items():
        if abs(alpha) < 0.01:  # Skip near-zero sliders
            continue
        vector = _get_or_compute_vector(model_name, feature_name, model_data)
        vectors.append((vector, alpha))
    return vectors


def generate_sync(
    model_name: str,
    prompt: str,
    sliders: dict[str, float],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Synchronous generation with steering. Returns full text."""
    tokens = list(generate_sync_stream(model_name, prompt, sliders, max_tokens, temperature))
    return "".join(tokens)


def generate_sync_stream(
    model_name: str,
    prompt: str,
    sliders: dict[str, float],
    max_tokens: int = 256,
    temperature: float = 0.7,
):
    """Synchronous streaming generation with steering. Yields tokens one at a time."""
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    layer_idx = model_data["config"]["steering_layer"]

    vectors = build_steering_vectors(model_name, sliders, model_data)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    with apply_steering(model, vectors, layer_idx):
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            input_ids = torch.cat([input_ids, next_token], dim=-1)


async def generate_stream(
    model_name: str,
    prompt: str,
    sliders: dict[str, float],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> AsyncIterator[str]:
    """Async streaming generation with steering. Yields tokens one at a time."""
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    layer_idx = model_data["config"]["steering_layer"]

    vectors = build_steering_vectors(model_name, sliders, model_data)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    with apply_steering(model, vectors, layer_idx):
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            input_ids = torch.cat([input_ids, next_token], dim=-1)


async def generate_comparison(
    model_name: str,
    prompt: str,
    sliders: dict[str, float],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> AsyncIterator[dict]:
    """Generate both steered and vanilla responses, yielding interleaved tokens.

    Yields dicts: {"token": str, "side": "steered"|"vanilla"}
    """
    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    layer_idx = model_data["config"]["steering_layer"]

    vectors = build_steering_vectors(model_name, sliders, model_data)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    steered_ids = inputs["input_ids"].clone()
    vanilla_ids = inputs["input_ids"].clone()

    for _ in range(max_tokens):
        # Steered generation step
        with apply_steering(model, vectors, layer_idx):
            with torch.no_grad():
                s_out = model(input_ids=steered_ids)
                s_logits = s_out.logits[:, -1, :] / max(temperature, 0.01)
                s_probs = torch.softmax(s_logits, dim=-1)
                s_next = torch.multinomial(s_probs, num_samples=1)

        # Vanilla generation step
        with torch.no_grad():
            v_out = model(input_ids=vanilla_ids)
            v_logits = v_out.logits[:, -1, :] / max(temperature, 0.01)
            v_probs = torch.softmax(v_logits, dim=-1)
            v_next = torch.multinomial(v_probs, num_samples=1)

        s_done = s_next.item() == tokenizer.eos_token_id
        v_done = v_next.item() == tokenizer.eos_token_id

        if not s_done:
            s_text = tokenizer.decode(s_next[0], skip_special_tokens=True)
            yield {"token": s_text, "side": "steered"}
            steered_ids = torch.cat([steered_ids, s_next], dim=-1)

        if not v_done:
            v_text = tokenizer.decode(v_next[0], skip_special_tokens=True)
            yield {"token": v_text, "side": "vanilla"}
            vanilla_ids = torch.cat([vanilla_ids, v_next], dim=-1)

        if s_done and v_done:
            break
