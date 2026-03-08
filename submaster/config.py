from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    download_url: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        name="tiny",
        filename="ggml-tiny.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true",
        description="Fastest and smallest model.",
    ),
    "base": ModelSpec(
        name="base",
        filename="ggml-base.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true",
        description="Balanced quality for short and medium clips.",
    ),
    "small": ModelSpec(
        name="small",
        filename="ggml-small.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true",
        description="Better accuracy with moderate resource usage.",
    ),
    "medium": ModelSpec(
        name="medium",
        filename="ggml-medium.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
        description="High accuracy for difficult audio.",
    ),
    "large": ModelSpec(
        name="large",
        filename="ggml-large.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin?download=true",
        description="Largest standard multilingual model.",
    ),
    "turbo": ModelSpec(
        name="turbo",
        filename="ggml-large-v3-turbo.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true",
        description="Large-v3-turbo optimized for faster inference.",
    ),
}

DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = "auto"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_THREADS = 4
