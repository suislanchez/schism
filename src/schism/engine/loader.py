"""Model and SAE loading with device auto-detection and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

SCHISM_HOME = Path.home() / ".schism"
MODELS_DIR = SCHISM_HOME / "models"
VECTORS_DIR = SCHISM_HOME / "vectors"
FEATURES_DIR = SCHISM_HOME / "features"

MODEL_REGISTRY = {
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "steering_layer": 12,
        "has_sae": True,
    },
    "gemma-2-2b-it": {
        "hf_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "steering_layer": 12,
        "has_sae": True,
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "sae_release": None,
        "sae_id": None,
        "steering_layer": 16,
        "has_sae": False,
    },
    "llama-3.2-3b-instruct": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "sae_release": None,
        "sae_id": None,
        "steering_layer": 16,
        "has_sae": False,
    },
}

# Singleton cache for loaded models
_loaded_models: dict[str, dict] = {}


def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dirs():
    """Create schism cache directories if they don't exist."""
    for d in [SCHISM_HOME, MODELS_DIR, VECTORS_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_model(
    model_name: str,
    device: Optional[str] = None,
    on_progress: Optional[callable] = None,
) -> dict:
    """Load a model and its tokenizer. Returns cached if already loaded.

    Args:
        model_name: Name from MODEL_REGISTRY
        device: Force a specific device (auto-detected if None)
        on_progress: Optional callback(stage: str, detail: str) for progress updates

    Returns dict with keys: model, tokenizer, config, device, sae (or None)
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Available models: {available}")

    ensure_dirs()
    config = MODEL_REGISTRY[model_name]
    device = device or get_device()

    def _progress(stage, detail=""):
        if on_progress:
            on_progress(stage, detail)

    _progress("tokenizer", f"Loading tokenizer for {config['hf_id']}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}'. "
            f"Make sure you have internet access and the model ID '{config['hf_id']}' is correct. "
            f"For gated models (like Llama), run: huggingface-cli login\n"
            f"Original error: {e}"
        ) from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _progress("model", f"Loading model weights for {config['hf_id']} (this may take a few minutes)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"],
            torch_dtype=torch.float32,  # MPS requires FP32
            device_map=device if device == "cuda" else None,
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}'. "
            f"For gated models (like Llama), you need to:\n"
            f"  1. Accept the license at https://huggingface.co/{config['hf_id']}\n"
            f"  2. Run: huggingface-cli login\n"
            f"Original error: {e}"
        ) from e

    if device != "cuda":
        _progress("device", f"Moving model to {device}")
        model = model.to(device)
    model.eval()

    sae = None
    if config["has_sae"] and config["sae_release"]:
        _progress("sae", f"Loading SAE: {config['sae_release']}")
        try:
            sae = load_sae(config["sae_release"], config["sae_id"], device)
        except Exception as e:
            # SAE is optional - warn but don't fail
            import warnings

            warnings.warn(
                f"Failed to load SAE for {model_name}: {e}. "
                f"Falling back to contrastive steering (no SAE)."
            )

    _progress("done", "Model ready")

    result = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "device": device,
        "sae": sae,
    }
    _loaded_models[model_name] = result
    return result


def load_sae(release: str, sae_id: str, device: str):
    """Load a pre-trained Sparse Autoencoder from SAELens."""
    from sae_lens import SAE

    sae, cfg, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    return sae


def unload_model(model_name: str):
    """Unload a model from memory."""
    if model_name in _loaded_models:
        del _loaded_models[model_name]
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def list_available_models() -> list[dict]:
    """List all models with their download/load status."""
    results = []
    for name, config in MODEL_REGISTRY.items():
        results.append(
            {
                "name": name,
                "hf_id": config["hf_id"],
                "has_sae": config["has_sae"],
                "loaded": name in _loaded_models,
            }
        )
    return results
