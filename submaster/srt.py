from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SubmasterError


# Match standard `HH:MM:SS,mmm` or `HH:MM:SS.mmm` SRT timestamps.
TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})[,.](?P<millis>\d{3})"
)


@dataclass(frozen=True)
class Cue:
    """Represent a single subtitle cue in milliseconds and rendered text.

    :param start_ms: Cue start timestamp in milliseconds.
    :type start_ms: int
    :param end_ms: Cue end timestamp in milliseconds.
    :type end_ms: int
    :param text: Cue text split into individual subtitle lines.
    :type text: list[str]
    """

    start_ms: int
    end_ms: int
    text: list[str]


def _timestamp_to_ms(raw: str) -> int:
    """Convert an SRT timestamp string into milliseconds.

    :param raw: Raw timestamp string from an SRT timing line.
    :type raw: str
    :returns: Parsed timestamp in milliseconds.
    :rtype: int
    :raises SubmasterError: If the timestamp does not match the expected format.
    """
    # Reject malformed timestamps early so downstream cue parsing stays simple.
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
    """Format a millisecond timestamp using canonical SRT syntax.

    :param milliseconds: Timestamp value to format.
    :type milliseconds: int
    :returns: Timestamp rendered as `HH:MM:SS,mmm`.
    :rtype: str
    :raises SubmasterError: If a negative timestamp is provided.
    """
    # SRT does not allow negative cue boundaries.
    if milliseconds < 0:
        raise SubmasterError("Negative timestamps are not valid in SRT.")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt(raw_srt: str) -> list[Cue]:
    """Parse raw SRT text into structured subtitle cues.

    :param raw_srt: Raw subtitle file contents.
    :type raw_srt: str
    :returns: Parsed subtitle cues in source order.
    :rtype: list[Cue]
    :raises SubmasterError: If the input is empty or contains malformed cues.
    """
    # Normalize line endings before splitting into cue blocks so Windows and Unix files parse identically.
    normalized = raw_srt.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SubmasterError("Generated SRT file is empty.")

    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", normalized)

    # Parse each block independently so numbering, timing, and text validation stay local.
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

    # Reject an all-empty parse result to surface bad generator output clearly.
    if not cues:
        raise SubmasterError("No subtitle cues were parsed from the generated SRT file.")
    return cues


def render_srt(cues: list[Cue]) -> str:
    """Render structured cues back into canonical SRT text.

    :param cues: Subtitle cues to render.
    :type cues: list[Cue]
    :returns: Normalized SRT text with CRLF line endings.
    :rtype: str
    """
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


def render_transcript(cues: list[Cue]) -> str:
    """Render subtitle cues as plain dialogue text without timestamps.

    :param cues: Subtitle cues to flatten into transcript lines.
    :type cues: list[Cue]
    :returns: Plain-text transcript with one normalized dialogue line per cue.
    :rtype: str
    """
    transcript_lines = [" ".join(line.strip() for line in cue.text if line.strip()) for cue in cues]
    return "\n".join(line for line in transcript_lines if line) + "\n"


def shift_cues(cues: list[Cue], offset_ms: int) -> list[Cue]:
    """Return subtitle cues shifted by a fixed timestamp offset.

    :param cues: Subtitle cues to shift.
    :type cues: list[Cue]
    :param offset_ms: Offset in milliseconds to add to every cue boundary.
    :type offset_ms: int
    :returns: New cue list with shifted timestamps.
    :rtype: list[Cue]
    :raises SubmasterError: If shifting would produce a negative timestamp.
    """
    shifted: list[Cue] = []
    for cue in cues:
        start_ms = cue.start_ms + offset_ms
        end_ms = cue.end_ms + offset_ms
        if start_ms < 0 or end_ms < 0:
            raise SubmasterError("Shifted cue timestamps cannot be negative.")
        shifted.append(Cue(start_ms=start_ms, end_ms=end_ms, text=cue.text.copy()))
    return shifted


def normalize_srt(raw_srt: str, offset_ms: int = 0) -> str:
    """Parse and re-render SRT text into the project's canonical format.

    :param raw_srt: Raw subtitle file contents to normalize.
    :type raw_srt: str
    :param offset_ms: Optional millisecond offset added to every cue boundary.
    :type offset_ms: int
    :returns: Canonically formatted SRT text.
    :rtype: str
    :raises SubmasterError: If the input cannot be parsed as valid SRT.
    """
    return render_srt(shift_cues(parse_srt(raw_srt), offset_ms))


def render_transcript_from_srt(raw_srt: str, offset_ms: int = 0) -> str:
    """Parse SRT text and render a plain-text transcript.

    :param raw_srt: Raw subtitle file contents to convert.
    :type raw_srt: str
    :param offset_ms: Optional millisecond offset applied before rendering.
    :type offset_ms: int
    :returns: Transcript text with timestamps removed.
    :rtype: str
    :raises SubmasterError: If the input cannot be parsed as valid SRT.
    """
    return render_transcript(shift_cues(parse_srt(raw_srt), offset_ms))
