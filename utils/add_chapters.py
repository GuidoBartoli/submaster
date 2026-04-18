#!/usr/bin/env python3
"""Embed chapter metadata into a video file from a plain-text chapter list."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


CHAPTER_RE = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\s+(?P<title>.+)$"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the standalone chapter utility."""
    parser = argparse.ArgumentParser(
        prog="add_chapters.py",
        description="Read chapters from a text file and embed them into a video with ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input video path.")
    parser.add_argument("chapters", help="Text file containing chapter timestamps and titles.")
    parser.add_argument(
        "output",
        help="Output video path. Adds .mp4 automatically if no suffix is provided.",
    )
    return parser


def parse_chapters(chapters_path: Path) -> list[dict[str, int | str]]:
    """Load chapter definitions from the plain-text chapter file."""
    chapters: list[dict[str, int | str]] = []
    for raw_line in chapters_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = CHAPTER_RE.fullmatch(line)
        if match is None:
            raise SystemExit(f"# Wrong chapter format: '{raw_line}'")

        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        if minutes > 59 or seconds > 59:
            raise SystemExit(f"# Wrong chapter format: '{raw_line}'")

        start_ms = ((hours * 3_600) + (minutes * 60) + seconds) * 1_000
        chapters.append({"title": match.group("title"), "start": start_ms})

    if not chapters:
        raise SystemExit("# No chapters found!")

    return chapters


def require_ffmpeg_tool(tool_name: str) -> None:
    """Exit early if a required ffmpeg executable is missing."""
    if shutil.which(tool_name) is None:
        raise SystemExit(f"# {tool_name} not installed!")


def extract_duration_ms(input_path: Path) -> int:
    """Return the media duration in milliseconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit("# Unable to read video duration!")
    return int(float(result.stdout.strip()) * 1_000)


def build_chapter_metadata(chapters: list[dict[str, int | str]], duration_ms: int) -> str:
    """Build the ffmetadata chapter block appended to the original metadata."""
    blocks: list[str] = []
    for index, chapter in enumerate(chapters):
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
    return "\n" + "\n".join(blocks) + "\n"


def export_existing_metadata(input_path: Path, metadata_path: Path) -> None:
    """Dump the source container metadata to a temporary ffmetadata file."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-f", "ffmetadata", str(metadata_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not metadata_path.exists():
        raise SystemExit("# Unable to extract metadata!")


def write_output_video(input_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Copy the input video while replacing metadata with the updated ffmetadata file."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(metadata_path),
            "-map_metadata",
            "1",
            "-codec",
            "copy",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("# Unable to update metadata!")


def main() -> int:
    """Run the standalone chapter embedding workflow."""
    args = build_parser().parse_args()

    input_path = Path(args.input).expanduser()
    chapters_path = Path(args.chapters).expanduser()
    output_path = Path(args.output).expanduser()
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mp4")

    if not input_path.exists():
        raise SystemExit("# Input video not found!")
    if not chapters_path.exists():
        raise SystemExit("# Chapter file not found!")

    require_ffmpeg_tool("ffmpeg")
    require_ffmpeg_tool("ffprobe")

    print(f'> Input video: "{input_path.name}"')
    print("- Extracting original metadata...")

    chapters = parse_chapters(chapters_path)
    print(f'- {len(chapters)} chapters found in "{chapters_path.name}":')
    for index, chapter in enumerate(chapters, start=1):
        print(f"  [{index:02d}] {chapter['title']} [start:{chapter['start']}]")

    print("- Computing chapter timestamps...")
    duration_ms = extract_duration_ms(input_path)
    chapter_metadata = build_chapter_metadata(chapters, duration_ms)

    with tempfile.TemporaryDirectory(prefix="submaster-chapters-") as temp_dir:
        metadata_path = Path(temp_dir) / "metadata.txt"
        export_existing_metadata(input_path, metadata_path)
        with metadata_path.open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(chapter_metadata)

        print("- Writing updated metadata...")
        write_output_video(input_path, metadata_path, output_path)

    print(f'> Output video: "{output_path.name}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
