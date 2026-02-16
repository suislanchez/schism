"""Feature discovery and management - maps SAE features to human-readable sliders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from schism.engine.loader import FEATURES_DIR, ensure_dirs

# Default behavioral features with contrastive prompt pairs.
# These work with both SAE-based and contrastive extraction.
DEFAULT_FEATURES = {
    "creativity": {
        "description": "How creative and imaginative the responses are",
        "positive": [
            "Be extremely creative, imaginative, and think outside the box.",
            "Write something wildly original and unexpected.",
            "Use vivid metaphors, unusual analogies, and inventive ideas.",
            "Approach this with maximum creativity and artistic flair.",
            "Generate the most imaginative and novel response possible.",
        ],
        "negative": [
            "Be plain, literal, and straightforward. No creativity needed.",
            "Write something completely ordinary and predictable.",
            "Use only basic language with no figurative speech.",
            "Approach this in the most boring, standard way possible.",
            "Generate a dry, factual response with no embellishment.",
        ],
    },
    "formality": {
        "description": "How formal and professional the tone is",
        "positive": [
            "Respond in an extremely formal, professional, and academic tone.",
            "Use sophisticated vocabulary and complex sentence structures.",
            "Write as if addressing a board of directors or academic panel.",
            "Maintain the highest level of professional decorum.",
            "Communicate with utmost formality and precision.",
        ],
        "negative": [
            "Respond in a super casual, chill, laid-back way.",
            "Use slang, contractions, and simple everyday language.",
            "Write like you're texting a close friend.",
            "Keep it totally relaxed and informal.",
            "Talk like you're hanging out at a bar with buddies.",
        ],
    },
    "humor": {
        "description": "How humorous and witty the responses are",
        "positive": [
            "Be hilarious, witty, and make people laugh.",
            "Include clever jokes, puns, and humorous observations.",
            "Respond with maximum humor and comedic timing.",
            "Be the funniest version of yourself possible.",
            "Make every sentence entertaining and amusing.",
        ],
        "negative": [
            "Be completely serious and humorless.",
            "No jokes, no wit, no levity whatsoever.",
            "Respond with absolute seriousness and gravity.",
            "Treat everything with utmost seriousness.",
            "Be as dry and serious as a legal document.",
        ],
    },
    "confidence": {
        "description": "How assertive and confident the tone is",
        "positive": [
            "Be extremely confident, assertive, and decisive.",
            "State things with absolute certainty and authority.",
            "Show no hesitation or doubt in your responses.",
            "Speak with the confidence of a world-leading expert.",
            "Be bold, direct, and unequivocal in every statement.",
        ],
        "negative": [
            "Be very uncertain, hesitant, and tentative.",
            "Hedge every statement with qualifiers and disclaimers.",
            "Express doubt and uncertainty about everything.",
            "Be wishy-washy and noncommittal in your responses.",
            "Start every point with 'I'm not sure, but...' or 'Maybe...'",
        ],
    },
    "verbosity": {
        "description": "How detailed and lengthy the responses are",
        "positive": [
            "Give an extremely detailed, thorough, and comprehensive response.",
            "Explain everything in great depth with many examples.",
            "Be as verbose and elaborate as possible.",
            "Cover every angle, nuance, and edge case in detail.",
            "Write a very long, detailed, and exhaustive answer.",
        ],
        "negative": [
            "Be extremely brief and concise. Use as few words as possible.",
            "Give the shortest possible answer.",
            "Be terse and to the point. No extra words.",
            "Respond in one sentence or less if possible.",
            "Minimize word count. Every word must earn its place.",
        ],
    },
    "empathy": {
        "description": "How emotionally warm and empathetic the tone is",
        "positive": [
            "Be extremely warm, caring, empathetic, and compassionate.",
            "Show deep emotional understanding and sensitivity.",
            "Respond with warmth, kindness, and genuine concern.",
            "Be nurturing, supportive, and emotionally present.",
            "Express deep empathy and emotional intelligence.",
        ],
        "negative": [
            "Be cold, detached, and purely analytical.",
            "Show no emotion or personal warmth whatsoever.",
            "Respond like a machine - factual and impersonal.",
            "Be completely dispassionate and emotionally neutral.",
            "Strip all warmth and feeling from your response.",
        ],
    },
    "technical": {
        "description": "How technical and jargon-heavy the responses are",
        "positive": [
            "Use highly technical language and domain-specific jargon.",
            "Explain things at an expert/PhD level with technical precision.",
            "Include technical details, specifications, and precise terminology.",
            "Write for a highly technical audience with deep expertise.",
            "Be as technically detailed and precise as possible.",
        ],
        "negative": [
            "Explain like I'm five years old. No jargon at all.",
            "Use the simplest possible language anyone could understand.",
            "Avoid any technical terms or complex vocabulary.",
            "Write for someone with zero technical background.",
            "Keep it simple enough for a complete beginner.",
        ],
    },
}


def get_default_features() -> dict:
    """Return the default feature definitions."""
    return DEFAULT_FEATURES


def get_feature_names() -> list[str]:
    """Return sorted list of available feature names."""
    return sorted(DEFAULT_FEATURES.keys())


def get_feature_info(feature_name: str) -> Optional[dict]:
    """Get info about a specific feature."""
    return DEFAULT_FEATURES.get(feature_name)


def save_custom_feature(name: str, feature_def: dict):
    """Save a user-defined custom feature."""
    ensure_dirs()
    path = FEATURES_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(feature_def, f, indent=2)


def load_custom_features() -> dict:
    """Load all user-defined custom features."""
    ensure_dirs()
    features = {}
    for path in FEATURES_DIR.glob("*.json"):
        with open(path) as f:
            features[path.stem] = json.load(f)
    return features


def get_all_features() -> dict:
    """Get both default and custom features."""
    features = dict(DEFAULT_FEATURES)
    features.update(load_custom_features())
    return features
