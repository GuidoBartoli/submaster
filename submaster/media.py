from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_SAMPLE_RATE
from .console import Console
from .errors import SubmasterError


def _run_probe(input_path: Path, entries: str, target: str) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        entries,
        target,
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SubmasterError(result.stderr.strip() or "ffprobe failed.")
    return result.stdout


def detect_media_type(input_path: Path) -> str:
    output = _run_probe(input_path, "stream=codec_type", "-show_streams")
    payload = json.loads(output or "{}")
    streams = payload.get("streams", [])
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    return "video" if has_video else "audio"


def probe_duration_seconds(input_path: Path) -> float | None:
    output = _run_probe(input_path, "format=duration", "-show_format")
    payload = json.loads(output or "{}")
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        return None
    try:
        return float(duration)
    except (TypeError, ValueError):
        return None


def create_work_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="submaster-"))


def extract_audio(
    source_path: Path,
    destination_path: Path,
    console: Console,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    duration = probe_duration_seconds(source_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(destination_path),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    progress = console.progress("ffmpeg", total=duration, unit="s")
    latest_seconds = 0.0
    stderr_lines: list[str] = []

    assert process.stdout is not None
    assert process.stderr is not None

    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if not line:
            continue
        line = line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"out_time_ms", "out_time_us"}:
            try:
                # ffmpeg emits microsecond progress values on the machine-readable stream.
                divisor = 1_000_000 if key == "out_time_ms" else 1_000_000
                latest_seconds = max(latest_seconds, float(value) / divisor)
                progress.update(latest_seconds)
            except ValueError:
                continue
        elif key == "progress" and value == "end":
            final_seconds = duration if duration is not None else latest_seconds
            progress.finish(final_seconds)

    stderr_lines = process.stderr.read().splitlines()
    return_code = process.wait()
    if return_code != 0:
        progress.finish(latest_seconds)
        error_message = "\n".join(line for line in stderr_lines if line.strip()) or "ffmpeg failed."
        raise SubmasterError(error_message)

    if not destination_path.exists():
        raise SubmasterError("ffmpeg finished without producing a WAV file.")

    console.success(f"Prepared audio: {destination_path}")
    return destination_path
