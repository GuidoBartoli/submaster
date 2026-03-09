from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SubmasterError


TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})[,.](?P<millis>\d{3})"
)


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: list[str]


def _timestamp_to_ms(raw: str) -> int:
    match = TIMESTAMP_RE.fullmatch(raw.strip())
    if not match:
        raise SubmasterError(f"Invalid SRT timestamp: '{raw}'.")
    parts = {name: int(value) for name, value in match.groupdict().items()}
    return (
        parts["hours"] * 3_600_000
        + parts["minutes"] * 60_000
        + parts["seconds"] * 1_000
        + parts["millis"]
    )


def format_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise SubmasterError("Negative timestamps are not valid in SRT.")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt(raw_srt: str) -> list[Cue]:
    normalized = raw_srt.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SubmasterError("Generated SRT file is empty.")

    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", normalized)
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timestamp_index = 0 if "-->" in lines[0] else 1
        if timestamp_index >= len(lines):
            raise SubmasterError(f"Malformed SRT block: '{block}'.")
        times = lines[timestamp_index]
        if "-->" not in times:
            raise SubmasterError(f"Missing cue timing line: '{block}'.")
        start_raw, end_raw = [part.strip() for part in times.split("-->", 1)]
        start_ms = _timestamp_to_ms(start_raw)
        end_ms = _timestamp_to_ms(end_raw)
        if end_ms < start_ms:
            raise SubmasterError("Cue end timestamp precedes start timestamp.")
        text = lines[timestamp_index + 1 :]
        if not text:
            raise SubmasterError("Cue text is missing.")
        cues.append(Cue(start_ms=start_ms, end_ms=end_ms, text=text))

    if not cues:
        raise SubmasterError("No subtitle cues were parsed from the generated SRT file.")
    return cues


def normalize_srt(raw_srt: str) -> str:
    cues = parse_srt(raw_srt)
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        # Rebuild every cue so numbering, timestamp separators, and line endings stay consistent.
        text = "\r\n".join(line.rstrip() for line in cue.text)
        blocks.append(
            f"{index}\r\n"
            f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}\r\n"
            f"{text}"
        )
    return "\r\n\r\n".join(blocks) + "\r\n"
