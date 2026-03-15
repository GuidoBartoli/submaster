from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_SAMPLE_RATE
from .console import Console
from .errors import SubmasterError


def _run_probe(input_path: Path, entries: str, target: str) -> str:
    """Run `ffprobe` and return the raw JSON payload.

    :param input_path: Media file to inspect.
    :type input_path: pathlib.Path
    :param entries: `ffprobe` field selector passed to `-show_entries`.
    :type entries: str
    :param target: `ffprobe` target flag such as `-show_streams`.
    :type target: str
    :returns: Standard output emitted by `ffprobe`.
    :rtype: str
    :raises SubmasterError: If `ffprobe` exits with a non-zero status.
    """
    # Keep the probe command machine-readable so callers can parse JSON reliably.
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


def has_video_stream(input_path: Path) -> bool:
    """Check whether a media file contains at least one video stream.

    :param input_path: Media file to inspect.
    :type input_path: pathlib.Path
    :returns: `True` when a video stream is present, otherwise `False`.
    :rtype: bool
    :raises SubmasterError: If probing the file fails.
    """
    # Ask ffprobe for stream types only; the CLI just needs to know whether video exists.
    output = _run_probe(input_path, "stream=codec_type", "-show_streams")
    payload = json.loads(output or "{}")
    streams = payload.get("streams", [])
    return any(stream.get("codec_type") == "video" for stream in streams)


def probe_duration_seconds(input_path: Path) -> float | None:
    """Extract the media duration in seconds when available.

    :param input_path: Media file to inspect.
    :type input_path: pathlib.Path
    :returns: Duration in seconds, or `None` when it cannot be parsed.
    :rtype: float | None
    :raises SubmasterError: If probing the file fails.
    """
    # Duration is optional metadata, so parsing errors degrade to `None` instead of hard failure.
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
    """Create a temporary working directory for intermediate media files.

    :returns: Newly created temporary directory path.
    :rtype: pathlib.Path
    """
    return Path(tempfile.mkdtemp(prefix="submaster-"))


def _format_ffmpeg_time(milliseconds: int) -> str:
    """Format a millisecond timestamp for ffmpeg CLI arguments.

    :param milliseconds: Timestamp value in milliseconds.
    :type milliseconds: int
    :returns: Timestamp rendered as fractional seconds.
    :rtype: str
    """
    return f"{milliseconds / 1_000:.3f}"


def extract_audio(
    source_path: Path,
    destination_path: Path,
    console: Console,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    clip_start_ms: int | None = None,
    clip_duration_ms: int | None = None,
) -> Path:
    """Extract normalized mono WAV audio from the input media file.

    :param source_path: Source media path containing the audio track.
    :type source_path: pathlib.Path
    :param destination_path: WAV path to create.
    :type destination_path: pathlib.Path
    :param console: Console used for progress and success output.
    :type console: Console
    :param sample_rate: Output sample rate for the normalized WAV file.
    :type sample_rate: int
    :param clip_start_ms: Optional source offset in milliseconds.
    :type clip_start_ms: int | None
    :param clip_duration_ms: Optional extracted duration in milliseconds.
    :type clip_duration_ms: int | None
    :returns: Path to the generated WAV file.
    :rtype: pathlib.Path
    :raises SubmasterError: If `ffmpeg` fails or does not create the destination file.
    """
    if clip_start_ms is not None and clip_start_ms < 0:
        raise SubmasterError("Clip start time cannot be negative.")
    if clip_duration_ms is not None and clip_duration_ms <= 0:
        raise SubmasterError("Clip duration must be greater than zero.")

    # Probe the duration first so the ffmpeg progress bar can show real-time completion.
    duration = (
        clip_duration_ms / 1_000
        if clip_duration_ms is not None
        else probe_duration_seconds(source_path)
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if clip_start_ms is not None:
        command.extend(["-ss", _format_ffmpeg_time(clip_start_ms)])
    command.extend(["-i", str(source_path)])
    if clip_duration_ms is not None:
        command.extend(["-t", _format_ffmpeg_time(clip_duration_ms)])
    command.extend(
        [
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
    )

    # Use machine-readable progress output so the console can render an updating progress bar.
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

    # Parse ffmpeg key=value progress lines until the subprocess exits.
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

    # Collect stderr only after the streaming loop so we preserve the progress experience.
    stderr_lines = process.stderr.read().splitlines()
    return_code = process.wait()
    if return_code != 0:
        progress.finish(latest_seconds)
        error_message = "\n".join(line for line in stderr_lines if line.strip()) or "ffmpeg failed."
        raise SubmasterError(error_message)

    # Treat a missing output file as a hard failure even if ffmpeg exited cleanly.
    if not destination_path.exists():
        raise SubmasterError("ffmpeg finished without producing a WAV file.")

    console.info(f"Prepared audio: {destination_path}")
    return destination_path
