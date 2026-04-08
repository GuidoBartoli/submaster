from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SubmasterError


# Match standard `HH:MM:SS,mmm` or `HH:MM:SS.mmm` SRT timestamps.
TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})[,.](?P<millis>\d{3})"
)
_TRANSCRIPT_WHITESPACE_RE = re.compile(r"\s+")
_SUBTITLE_BREAK_RE = re.compile(r"[,;:.!?]+$")
_WORD_TOKEN_RE = re.compile(r"\S+")
_MAX_SUBTITLE_LINE_CHARS = 42
_MAX_SUBTITLE_LINES = 2
_MAX_SUBTITLE_CUE_CHARS = _MAX_SUBTITLE_LINE_CHARS * _MAX_SUBTITLE_LINES
_TARGET_SUBTITLE_CUE_CHARS = 64
_MIN_SUBTITLE_SPLIT_CHARS = 28
_MIN_SUBTITLE_CUE_MS = 900


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


def _normalize_cue_text_lines(lines: list[str]) -> str:
    """Collapse subtitle lines into a single normalized prose string."""
    return _TRANSCRIPT_WHITESPACE_RE.sub(
        " ",
        " ".join(line.strip() for line in lines if line.strip()),
    ).strip()


def _wrap_subtitle_lines(text: str) -> list[str]:
    """Wrap subtitle prose into at most two readable lines when possible."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and len(candidate) > _MAX_SUBTITLE_LINE_CHARS:
            lines.append(" ".join(current))
            current = [word]
            continue
        current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines


def _can_render_as_single_cue(text: str) -> bool:
    """Check whether a cue fits comfortably within the subtitle layout limits."""
    wrapped = _wrap_subtitle_lines(text)
    return len(wrapped) <= _MAX_SUBTITLE_LINES and all(
        len(line) <= _MAX_SUBTITLE_LINE_CHARS for line in wrapped
    )


def _is_preferred_split_point(word: str) -> bool:
    """Prefer splitting after punctuation that naturally marks a short pause."""
    return bool(_SUBTITLE_BREAK_RE.search(word))


def _split_long_cue_text(text: str) -> list[str]:
    """Split oversized subtitle prose into multiple readable cue-sized chunks."""
    words = _WORD_TOKEN_RE.findall(text)
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        current_words.append(word)
        current_text = " ".join(current_words)
        current_length = len(current_text)

        next_is_safe_break = _is_preferred_split_point(word) and current_length >= _MIN_SUBTITLE_SPLIT_CHARS
        if current_length >= _TARGET_SUBTITLE_CUE_CHARS and next_is_safe_break:
            chunks.append(current_text)
            current_words = []
            continue

        if current_length > _MAX_SUBTITLE_CUE_CHARS and len(current_words) > 1:
            overflow_word = current_words.pop()
            chunks.append(" ".join(current_words))
            current_words = [overflow_word]

    if current_words:
        chunks.append(" ".join(current_words))

    # Merge tiny tail fragments back into the previous chunk when they would render poorly alone.
    if len(chunks) >= 2 and len(chunks[-1]) < (_MIN_SUBTITLE_SPLIT_CHARS // 2):
        merged_tail = f"{chunks[-2]} {chunks[-1]}".strip()
        if len(merged_tail) <= (_MAX_SUBTITLE_CUE_CHARS + (_MIN_SUBTITLE_SPLIT_CHARS // 2)):
            chunks[-2] = merged_tail
            chunks.pop()

    fitted_chunks: list[str] = []
    for chunk in chunks:
        if _can_render_as_single_cue(chunk):
            fitted_chunks.append(chunk)
            continue
        fitted_chunks.extend(_split_chunk_to_fit(chunk))
    return fitted_chunks


def _split_chunk_to_fit(text: str) -> list[str]:
    """Force a long chunk to fit by splitting on word boundaries near the midpoint."""
    words = text.split()
    if len(words) <= 1:
        return [text]

    midpoint = len(words) // 2
    split_index = midpoint
    while split_index > 1:
        candidate = " ".join(words[:split_index])
        if _can_render_as_single_cue(candidate):
            break
        split_index -= 1
    if split_index <= 1:
        split_index = midpoint

    left = " ".join(words[:split_index]).strip()
    right = " ".join(words[split_index:]).strip()
    segments: list[str] = []
    if left:
        if _can_render_as_single_cue(left):
            segments.append(left)
        else:
            segments.extend(_split_chunk_to_fit(left))
    if right:
        if _can_render_as_single_cue(right):
            segments.append(right)
        else:
            segments.extend(_split_chunk_to_fit(right))
    return segments


def _allocate_segment_durations(
    start_ms: int,
    end_ms: int,
    segments: list[str],
) -> list[tuple[int, int]]:
    """Allocate cue timings proportionally across split subtitle segments."""
    total_duration = end_ms - start_ms
    if total_duration <= 0 or len(segments) <= 1:
        return [(start_ms, end_ms)]

    weights = [max(1, len(segment)) for segment in segments]
    total_weight = sum(weights)
    remaining_start = start_ms
    remaining_duration = total_duration
    remaining_weight = total_weight
    timings: list[tuple[int, int]] = []

    for index, weight in enumerate(weights):
        is_last = index == len(weights) - 1
        if is_last:
            segment_end = end_ms
        else:
            proportional = round(remaining_duration * (weight / remaining_weight))
            min_duration = min(
                _MIN_SUBTITLE_CUE_MS,
                max(1, remaining_duration - (len(weights) - index - 1)),
            )
            segment_duration = max(min_duration, proportional)
            max_end = end_ms - (len(weights) - index - 1)
            segment_end = min(max_end, remaining_start + segment_duration)
        timings.append((remaining_start, segment_end))
        remaining_duration = end_ms - segment_end
        remaining_weight -= weight
        remaining_start = segment_end

    return timings


def _segment_cue(cue: Cue) -> list[Cue]:
    """Split an oversized cue into smaller subtitle-friendly cues when needed."""
    normalized_text = _normalize_cue_text_lines(cue.text)
    if not normalized_text:
        return [cue]

    if _can_render_as_single_cue(normalized_text):
        return [Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=_wrap_subtitle_lines(normalized_text))]

    segments = _split_long_cue_text(normalized_text)
    if len(segments) <= 1:
        return [Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=_wrap_subtitle_lines(normalized_text))]

    timings = _allocate_segment_durations(cue.start_ms, cue.end_ms, segments)
    segmented_cues: list[Cue] = []
    for (segment_start, segment_end), segment_text in zip(timings, segments, strict=True):
        segmented_cues.append(
            Cue(
                start_ms=segment_start,
                end_ms=segment_end,
                text=_wrap_subtitle_lines(segment_text),
            )
        )
    return segmented_cues


def resegment_cues(cues: list[Cue]) -> list[Cue]:
    """Rewrite cues into subtitle-friendly chunks and line lengths."""
    segmented: list[Cue] = []
    for cue in cues:
        segmented.extend(_segment_cue(cue))
    return segmented


def render_transcript(cues: list[Cue]) -> str:
    """Render subtitle cues as plain dialogue text without timestamps.

    :param cues: Subtitle cues to flatten into transcript lines.
    :type cues: list[Cue]
    :returns: Plain-text transcript with one normalized dialogue sentence per line.
    :rtype: str
    """
    transcript_text = " ".join(
        " ".join(line.strip() for line in cue.text if line.strip()) for cue in cues
    )
    normalized_text = _TRANSCRIPT_WHITESPACE_RE.sub(" ", transcript_text).strip()
    if not normalized_text:
        return ""
    transcript_lines = _split_transcript_sentences(normalized_text)
    return "\n".join(transcript_lines) + "\n"


def _split_transcript_sentences(text: str) -> list[str]:
    """Split transcript prose into readable sentence-like lines."""
    lines: list[str] = []
    current_line: list[str] = []
    index = 0

    while index < len(text):
        character = text[index]
        current_line.append(character)
        index += 1

        if character not in ".!?":
            continue

        while index < len(text) and text[index] in "\"')]}":
            current_line.append(text[index])
            index += 1

        while index < len(text) and text[index].isspace():
            index += 1

        lines.append("".join(current_line).strip())
        current_line = []

    trailing_line = "".join(current_line).strip()
    if trailing_line:
        lines.append(trailing_line)
    return lines


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
    return render_srt(resegment_cues(shift_cues(parse_srt(raw_srt), offset_ms)))


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
