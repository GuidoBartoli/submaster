from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Describe a downloadable model artifact.

    :param name: Human-readable model identifier exposed through the CLI.
    :type name: str
    :param filename: Local filename used when storing the model on disk.
    :type filename: str
    :param download_url: Remote URL used to fetch the model binary.
    :type download_url: str
    :param description: Short explanation of the model tradeoffs.
    :type description: str
    """

    name: str
    filename: str
    download_url: str
    description: str


# Whisper transcription models exposed by the CLI.
WHISPER_MODEL_SPECS: dict[str, ModelSpec] = {
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
        filename="ggml-large-v3.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
        description="Largest standard multilingual model (large-v3).",
    ),
    "turbo": ModelSpec(
        name="turbo",
        filename="ggml-large-v3-turbo.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true",
        description="Large-v3-turbo optimized for faster inference.",
    ),
}

# Tencent HY-MT translation models used by the optional translation stage.
TRANSLATION_MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(
        name="small",
        filename="HY-MT1.5-1.8B-Q4_K_M.gguf",
        download_url=(
            "https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/resolve/main/"
            "HY-MT1.5-1.8B-Q4_K_M.gguf?download=true"
        ),
        description=(
            "Tencent HY-MT1.5 1.8B in Q4_K_M GGUF format. "
            "Chosen as the default translation model because the int4 quantization "
            "is small enough for typical local machines while still preserving "
            "good subtitle translation quality."
        ),
    ),
    "large": ModelSpec(
        name="large",
        filename="HY-MT1.5-7B-Q4_K_M.gguf",
        download_url=(
            "https://huggingface.co/tencent/HY-MT1.5-7B-GGUF/resolve/main/"
            "HY-MT1.5-7B-Q4_K_M.gguf?download=true"
        ),
        description=(
            "Tencent HY-MT1.5 7B in Q4_K_M GGUF format. "
            "Uses the same int4 quantization to keep the model practical for "
            "offline local inference while improving translation quality."
        ),
    ),
}

# Backwards-compatible alias used by the existing whisper-only code path and tests.
MODEL_SPECS = WHISPER_MODEL_SPECS

# Default CLI values shared across modules.
DEFAULT_MODEL = "base"
DEFAULT_TRANSLATION_MODEL = "small"
DEFAULT_LANGUAGE = "auto"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_THREADS = 4

# llama.cpp runtime defaults tuned for subtitle translation prompts.
LLAMA_CONTEXT_SIZE = 4_096
LLAMA_MAX_BATCH_CUES = 12
LLAMA_MAX_BATCH_CHARS = 1_600
LLAMA_N_GPU_LAYERS_CPU = 0
LLAMA_N_GPU_LAYERS_ALL = 999
LLAMA_TEMPERATURE = 0.7
LLAMA_TOP_K = 20
LLAMA_TOP_P = 0.6
LLAMA_REPEAT_PENALTY = 1.05

# Published language labels used to normalize user-facing translation targets.
TRANSLATION_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "bo": "Tibetan",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
}
