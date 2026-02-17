# Schism

Reshape how AI thinks, one slider at a time.

Schism is a local activation steering tool for LLMs. It lets you control model personality — creativity, humor, formality, confidence, and more — using visual sliders or CLI flags. No fine-tuning, no cloud, no ML knowledge required.

It works by injecting steering vectors into a model's hidden states at inference time, using either Sparse Autoencoders (SAE) or contrastive activation extraction.

## Quick Start

```bash
pip install schism

# Download a model (~5GB)
schism download gemma-2-2b

# Launch the web UI
schism
```

Open `http://localhost:6660`, drag the sliders, type a prompt, and watch the model's personality shift in real time.

## CLI Usage

```bash
# Generate with personality sliders
schism steer "Explain quantum physics" --humor 0.9 --creativity 0.8 --formality -0.6

# Use a preset
schism steer "Tell me about the ocean" --preset pirate

# Side-by-side comparison: vanilla vs steered
schism compare "What is love?" --preset sarcastic

# Pre-compute steering vectors for faster first generation
schism warmup gemma-2-2b
```

## Steering Dimensions

Each slider maps to a steering vector computed from contrastive prompt pairs:

| Slider | Description |
|--------|-------------|
| `creativity` | Imaginative and expressive vs plain and literal |
| `formality` | Professional and academic vs casual and relaxed |
| `humor` | Witty and funny vs serious and dry |
| `confidence` | Assertive and decisive vs hesitant and uncertain |
| `verbosity` | Detailed and thorough vs brief and concise |
| `empathy` | Warm and compassionate vs cold and analytical |
| `technical` | Expert jargon vs simple language |

All sliders range from `-1.0` to `+1.0`. Create custom features with `schism features create`.

## Presets

11 built-in personality presets:

**Creative** · **Formal** · **Sarcastic** · **Concise** · **Pirate** · **ELI5** · **Academic** · **Noir Detective** · **Zen Master** · **Chaos** · **Robot**

```bash
# List all presets
schism presets list

# Show preset details
schism presets show pirate

# Export for sharing
schism presets export pirate -o pirate.json
```

## Supported Models

| Model | HuggingFace ID | SAE Support |
|-------|---------------|-------------|
| `gemma-2-2b` | `google/gemma-2-2b` | Yes (Gemma Scope) |
| `gemma-2-2b-it` | `google/gemma-2-2b-it` | Yes (Gemma Scope) |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B` | No (contrastive) |
| `llama-3.2-3b-instruct` | `meta-llama/Llama-3.2-3B-Instruct` | No (contrastive) |

Models with SAE support use Sparse Autoencoders for more interpretable steering. Models without SAE fall back to contrastive activation extraction, which still works well.

## How It Works

1. **Contrastive prompts** define each steering dimension (e.g. "be creative" vs "be plain")
2. **Activation extraction** captures the model's internal representations for each set of prompts
3. **Steering vectors** are computed as the difference between positive and negative activations
4. **At inference time**, these vectors are injected into the model's hidden states at a target layer, scaled by the slider value

Based on research from [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) and [FGAA (2025)](https://arxiv.org/abs/2501.09929).

## Docker

```bash
docker build -t schism .
docker run -p 6660:6660 schism
```

For GPU support:

```bash
docker run --gpus all -p 6660:6660 schism
```

## Development

```bash
git clone https://github.com/suislanchez/schism.git
cd schism
pip install -e ".[test]"
pytest tests/ -v
```

## License

MIT
