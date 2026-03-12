from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .config import (
    MODEL_SPECS,
    TRANSLATION_MODEL_SPECS,
    VAD_MODEL_SPECS,
    WHISPER_MODEL_SPECS,
    ModelSpec,
)
from .console import Console, format_bytes
from .errors import SubmasterError


def resolve_named_model_spec(
    name: str,
    model_specs: dict[str, ModelSpec],
    model_label: str = "model",
) -> ModelSpec:
    """Resolve a model name within a specific model registry.

    :param name: User-provided model name to resolve.
    :type name: str
    :param model_specs: Mapping of valid model names to model metadata.
    :type model_specs: dict[str, ModelSpec]
    :param model_label: Label used in user-facing validation errors.
    :type model_label: str
    :returns: Matching model specification.
    :rtype: ModelSpec
    :raises SubmasterError: If the requested model name is unknown.
    """
    # Normalize the key once so lookups behave consistently across CLI input variations.
    normalized = name.lower().strip()
    if normalized in model_specs:
        return model_specs[normalized]

    for spec in model_specs.values():
        if spec.filename.lower() == normalized:
            return spec

    choices = ", ".join(dict.fromkeys(spec.name for spec in model_specs.values()))
    raise SubmasterError(
        f"Unknown {model_label} '{name}'. Choose one of: {choices}."
    )


def resolve_model_spec(name: str) -> ModelSpec:
    """Resolve a whisper model name from the built-in whisper registry.

    :param name: Whisper model name from the CLI or tests.
    :type name: str
    :returns: Matching whisper model specification.
    :rtype: ModelSpec
    """
    return resolve_named_model_spec(name, WHISPER_MODEL_SPECS)


def resolve_translation_model_spec(name: str) -> ModelSpec:
    """Resolve a translation model name from the built-in translation registry.

    :param name: Translation model name from the CLI or tests.
    :type name: str
    :returns: Matching translation model specification.
    :rtype: ModelSpec
    """
    return resolve_named_model_spec(name, TRANSLATION_MODEL_SPECS, model_label="translation model")


def resolve_vad_model_spec(name: str) -> ModelSpec:
    """Resolve a VAD model name from the built-in VAD registry.

    :param name: VAD model name from the CLI.
    :type name: str
    :returns: Matching VAD model specification.
    :rtype: ModelSpec
    """
    return resolve_named_model_spec(name, VAD_MODEL_SPECS, model_label="VAD model")


def ensure_model_available(
    name: str,
    models_dir: Path,
    console: Console,
    model_specs: dict[str, ModelSpec] | None = None,
    model_label: str = "model",
) -> Path:
    """Ensure that a requested model exists locally, downloading it if needed.

    :param name: Model name to resolve and prepare.
    :type name: str
    :param models_dir: Directory where model files are stored.
    :type models_dir: pathlib.Path
    :param console: Console used for status output and progress updates.
    :type console: Console
    :param model_specs: Optional registry overriding the default model mapping.
    :type model_specs: dict[str, ModelSpec] | None
    :param model_label: User-facing label for logs and errors.
    :type model_label: str
    :returns: Path to the ready-to-use model file.
    :rtype: pathlib.Path
    :raises SubmasterError: If the model cannot be resolved or downloaded.
    """
    # Resolve the requested model first so all later steps work with canonical metadata.
    specs = model_specs or MODEL_SPECS
    spec = resolve_named_model_spec(name, specs, model_label=model_label)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / spec.filename

    # Reuse an existing non-empty model file instead of downloading again.
    if model_path.exists() and model_path.stat().st_size > 0:
        console.success(f"{model_label.capitalize()} '{spec.name}' ready: {model_path}")
        return model_path

    # Always write to a temporary part file so interrupted downloads do not look valid.
    tmp_path = model_path.with_suffix(f"{model_path.suffix}.part")
    if tmp_path.exists():
        tmp_path.unlink()

    console.note(f"Downloading {model_label} '{spec.name}' to {model_path}")
    request = urllib.request.Request(
        spec.download_url,
        headers={"User-Agent": "submaster/0.1 (+https://github.com/ggml-org/whisper.cpp)"},
    )

    try:
        # Stream the payload to disk and report byte progress as the download advances.
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0")) or None
            progress = console.progress("download", total=total, unit="B")
            downloaded = 0
            # Stream to disk first so incomplete downloads never masquerade as valid models.
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                extra = format_bytes(downloaded)
                progress.update(downloaded, extra=extra)
            progress.finish(downloaded, extra=format_bytes(downloaded))
        os.replace(tmp_path, model_path)
    except Exception as exc:
        # Remove the partial file on failure so the next run starts cleanly.
        if tmp_path.exists():
            tmp_path.unlink()
        raise SubmasterError(f"Failed to download {model_label} '{spec.name}': {exc}") from exc

    console.success(f"{model_label.capitalize()} '{spec.name}' downloaded successfully.")
    return model_path


def ensure_translation_model_available(name: str, models_dir: Path, console: Console) -> Path:
    """Ensure that a translation model exists locally.

    :param name: Translation model name to prepare.
    :type name: str
    :param models_dir: Directory where translation models are stored.
    :type models_dir: pathlib.Path
    :param console: Console used for status output and progress updates.
    :type console: Console
    :returns: Path to the ready-to-use translation model file.
    :rtype: pathlib.Path
    :raises SubmasterError: If the model cannot be resolved or downloaded.
    """
    return ensure_model_available(
        name,
        models_dir,
        console,
        model_specs=TRANSLATION_MODEL_SPECS,
        model_label="translation model",
    )


def ensure_vad_model_available(name: str, models_dir: Path, console: Console) -> Path:
    """Ensure that a VAD model exists locally.

    :param name: VAD model name to prepare.
    :type name: str
    :param models_dir: Directory where VAD models are stored.
    :type models_dir: pathlib.Path
    :param console: Console used for status output and progress updates.
    :type console: Console
    :returns: Path to the ready-to-use VAD model file.
    :rtype: pathlib.Path
    :raises SubmasterError: If the model cannot be resolved or downloaded.
    """
    return ensure_model_available(
        name,
        models_dir,
        console,
        model_specs=VAD_MODEL_SPECS,
        model_label="VAD model",
    )
