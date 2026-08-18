from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_SAMPLE_RATE
from .console import Console
from .errors import SubmasterError


# Expected chapter line format: "HH:MM:SS Chapter title"
_CHAPTER_RE = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\s+(?P<title>.+)$"
)


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
    :param console: Console used for progress and status output.
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


def parse_chapters(chapters_path: Path) -> list[dict[str, int | str]]:
    """Parse a plain-text chapter file into a list of chapter dicts.

    :param chapters_path: Path to a text file with lines of the form ``HH:MM:SS Title``.
    :type chapters_path: pathlib.Path
    :returns: List of dicts with ``title`` (str) and ``start`` (int milliseconds) keys.
    :rtype: list[dict[str, int | str]]
    :raises SubmasterError: If any line is malformed or the file contains no chapters.
    """
    chapters: list[dict[str, int | str]] = []
    for raw_line in chapters_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _CHAPTER_RE.fullmatch(line)
        if match is None:
            raise SubmasterError(f"Wrong chapter format: '{raw_line}'")

        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        # The regex only checks digit count, not value range — reject out-of-range fields manually
        if minutes > 59 or seconds > 59:
            raise SubmasterError(f"Wrong chapter format: '{raw_line}'")

        # ffmetadata timestamps are expressed in milliseconds
        start_ms = ((hours * 3_600) + (minutes * 60) + seconds) * 1_000
        chapters.append({"title": match.group("title"), "start": start_ms})

    if not chapters:
        raise SubmasterError("No chapters found in the chapter file.")

    return chapters


def _build_chapter_metadata(chapters: list[dict[str, int | str]], duration_ms: int) -> str:
    """Return the ffmetadata chapter block to append to the container metadata."""
    blocks: list[str] = []
    for index, chapter in enumerate(chapters):
        # Last chapter ends at the media duration; all others end 1 ms before the next chapter starts
        next_start = duration_ms if index == len(chapters) - 1 else int(chapters[index + 1]["start"]) - 1
        blocks.append(
            "\n".join(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={chapter['start']}",
                    f"END={next_start}",
                    f"title={chapter['title']}",
                ]
            )
        )
    # Leading newline separates chapters from the global metadata header above
    return "\n" + "\n".join(blocks) + "\n"


def _export_existing_metadata(input_path: Path, metadata_path: Path) -> None:
    """Dump the container metadata of a video to an ffmetadata file."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-f", "ffmetadata", str(metadata_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # ffmpeg can exit 0 yet still fail to write the file when the container has no metadata
    if result.returncode != 0 or not metadata_path.exists():
        raise SubmasterError("Unable to extract video metadata.")


def _write_chapter_video(input_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Copy a video file while replacing its container metadata."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(metadata_path),
            # Use the second input (ffmetadata file) as the metadata source
            "-map_metadata",
            "1",
            # Stream-copy avoids re-encoding; only the container metadata changes
            "-codec",
            "copy",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SubmasterError("Unable to write chapter-embedded video.")


def embed_chapters(
    input_path: Path,
    chapters_path: Path,
    output_path: Path,
    console: Console,
) -> None:
    """Embed chapter timestamps from a text file into a copy of the video.

    :param input_path: Source video file.
    :type input_path: pathlib.Path
    :param chapters_path: Plain-text chapter file (``HH:MM:SS Title`` per line).
    :type chapters_path: pathlib.Path
    :param output_path: Destination video file to create.
    :type output_path: pathlib.Path
    :param console: Console used for progress and status output.
    :type console: Console
    :raises SubmasterError: If chapter parsing or any ffmpeg step fails.
    """
    chapters = parse_chapters(chapters_path)
    console.info(f"Embedding {len(chapters)} chapter(s) into '{output_path.name}'.")

    duration_s = probe_duration_seconds(input_path)
    if duration_s is None:
        raise SubmasterError("Unable to read video duration.")
    duration_ms = int(duration_s * 1_000)

    chapter_metadata = _build_chapter_metadata(chapters, duration_ms)

    with tempfile.TemporaryDirectory(prefix="submaster-chapters-") as temp_dir:
        metadata_path = Path(temp_dir) / "metadata.txt"
        # Start from the video's existing metadata so non-chapter tags are preserved
        _export_existing_metadata(input_path, metadata_path)
        with metadata_path.open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(chapter_metadata)
        _write_chapter_video(input_path, metadata_path, output_path)

    console.info(f"Chapter-embedded video written to {output_path}")
