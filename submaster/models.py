from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .config import MODEL_SPECS, TRANSLATION_MODEL_SPECS, WHISPER_MODEL_SPECS, ModelSpec
from .console import Console, format_bytes
from .errors import SubmasterError


def resolve_named_model_spec(
    name: str,
    model_specs: dict[str, ModelSpec],
    model_label: str = "model",
) -> ModelSpec:
    normalized = name.lower().strip()
    try:
        return model_specs[normalized]
    except KeyError as exc:
        choices = ", ".join(model_specs)
        raise SubmasterError(
            f"Unknown {model_label} '{name}'. Choose one of: {choices}."
        ) from exc


def resolve_model_spec(name: str) -> ModelSpec:
    return resolve_named_model_spec(name, WHISPER_MODEL_SPECS)


def resolve_translation_model_spec(name: str) -> ModelSpec:
    return resolve_named_model_spec(name, TRANSLATION_MODEL_SPECS, model_label="translation model")


def ensure_model_available(
    name: str,
    models_dir: Path,
    console: Console,
    model_specs: dict[str, ModelSpec] | None = None,
    model_label: str = "model",
) -> Path:
    specs = model_specs or MODEL_SPECS
    spec = resolve_named_model_spec(name, specs, model_label=model_label)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / spec.filename
    if model_path.exists() and model_path.stat().st_size > 0:
        console.success(f"{model_label.capitalize()} '{spec.name}' ready: {model_path}")
        return model_path

    tmp_path = model_path.with_suffix(f"{model_path.suffix}.part")
    if tmp_path.exists():
        tmp_path.unlink()

    console.note(f"Downloading {model_label} '{spec.name}' to {model_path}")
    request = urllib.request.Request(
        spec.download_url,
        headers={"User-Agent": "submaster/0.1 (+https://github.com/ggml-org/whisper.cpp)"},
    )

    try:
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
        if tmp_path.exists():
            tmp_path.unlink()
        raise SubmasterError(f"Failed to download {model_label} '{spec.name}': {exc}") from exc

    console.success(f"{model_label.capitalize()} '{spec.name}' downloaded successfully.")
    return model_path


def ensure_translation_model_available(name: str, models_dir: Path, console: Console) -> Path:
    return ensure_model_available(
        name,
        models_dir,
        console,
        model_specs=TRANSLATION_MODEL_SPECS,
        model_label="translation model",
    )
