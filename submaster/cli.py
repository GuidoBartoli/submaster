from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import (
    DEFAULT_CLEANUP_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_VAD_MODEL,
    DEFAULT_WHISPER_MAX_CONTEXT,
    MODEL_SPECS,
    TRANSLATION_MODEL_SPECS,
    VAD_MODEL_SPECS,
)
from .console import Console
from .errors import SubmasterError
from .llama_cpp import LlamaCppRunner
from .media import create_work_dir, embed_chapters, extract_audio, has_video_stream
from .models import (
    ensure_cleanup_model_available,
    ensure_model_available,
    ensure_translation_model_available,
    ensure_vad_model_available,
)
from .srt import normalize_srt, render_transcript_from_srt
from .transcript_cleanup import TranscriptCleaner
from .translation import SubtitleTranslator
from .whisper_cpp import WhisperCppRunner


_RANGE_INTEGER_RE = re.compile(r"^\d+$")
_RANGE_SECONDS_RE = re.compile(r"^(?P<seconds>\d+)(?:[,.](?P<millis>\d{1,3}))?$")


@dataclass
class ProcessingResources:
    """Bundle reusable runtime state for one CLI invocation."""

    whisper_runner: WhisperCppRunner
    whisper_model_path: Path
    vad_model_path: Path | None
    translator: SubtitleTranslator | None
    transcript_cleaner: TranscriptCleaner | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the `submaster` executable.

    :returns: Configured parser for the CLI entrypoint.
    :rtype: argparse.ArgumentParser
    """
    # Group the parser definition up front so the rest of the module can reuse a
    # single source of truth for CLI help text and defaults.
    parser = argparse.ArgumentParser(
        prog="submaster",
        description=(
            "Create synchronized .srt subtitles from a video file, or batch-process "
            "every video file in a folder, using whisper.cpp with optional offline "
            "translation and transcript cleanup via llama.cpp."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core input/output options define the source media file and final
    # subtitle location.
    parser.add_argument(
        "input",
        help="Path to a video file, or a folder when --batch is set.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output .srt file path or directory. Defaults to the input stem next to "
            "the source file. In --batch mode this must be a directory."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat the input path as a folder and process every contained video file.",
    )

    # Whisper-specific options control transcription quality, language
    # handling, and runtime selection.
    parser.add_argument(
        "-m",
        "--wmodel",
        default=DEFAULT_MODEL,
        choices=tuple(MODEL_SPECS),
        help="Whisper model to use.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code for whisper.cpp. Use 'auto' to let the model detect it.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "gpu"),
        help="Prefer GPU if available, or force cpu/gpu.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Folder used to cache ggml whisper models.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_WHISPER_MAX_CONTEXT,
        help=(
            "Maximum prior-text tokens fed back into whisper.cpp. "
            "Use 0 to disable rolling text context; use -1 to restore the upstream default."
        ),
    )
    parser.add_argument(
        "--vad-model",
        choices=tuple(VAD_MODEL_SPECS),
        help=(
            "Enable whisper.cpp VAD with the selected model. "
            f"Use '{DEFAULT_VAD_MODEL}' for the current default VAD model."
        ),
    )
    parser.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help=(
            "Only transcribe and translate the given time range. "
            "Accepted formats: SS, MM:SS, or HH:MM:SS with optional .mmm or ,mmm."
        ),
    )

    # Runtime flags tune performance and output handling without changing
    # transcript content.
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or DEFAULT_THREADS) - 1),
        help="CPU threads passed to whisper.cpp.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination SRT if it already exists.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the normalized WAV file next to the generated subtitle.",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Also write a plain-text transcription next to the generated subtitle.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Polish the transcription text with a local Qwen cleanup model. Has no effect without --transcribe.",
    )
    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Display the final whisper.cpp timing summary.",
    )
    parser.add_argument(
        "--show-model-info",
        action="store_true",
        help="Display whisper.cpp model-load metadata after transcription.",
    )

    # Translation options stay optional and are only used when the caller
    # asks for a second-language subtitle file.
    parser.add_argument(
        "--translate",
        help="Translate the generated subtitles into the given target language code or name.",
    )
    parser.add_argument(
        "--tmodel",
        default=DEFAULT_TRANSLATION_MODEL,
        choices=tuple(TRANSLATION_MODEL_SPECS),
        help="Tencent HY-MT translation model size to use when --translate is set.",
    )
    parser.add_argument(
        "--chapters",
        metavar="FILE",
        help=(
            "Path to a plain-text chapter file (one 'HH:MM:SS Title' entry per line). "
            "When set, a chapter-embedded copy of the input video is written next to it "
            "with a '_chapters' suffix. Cannot be used with --batch."
        ),
    )
    return parser


def parse_time_value(raw_value: str) -> int:
    """Parse a user-facing time value into milliseconds.

    :param raw_value: Time expression such as `75`, `01:15`, or `00:01:15.250`.
    :type raw_value: str
    :returns: Parsed millisecond value.
    :rtype: int
    :raises SubmasterError: If the value is malformed.
    """
    normalized = raw_value.strip()
    if not normalized:
        raise SubmasterError("Time values cannot be empty.")

    parts = normalized.split(":")
    if len(parts) > 3:
        raise SubmasterError(
            f"Invalid time value '{raw_value}'. Use SS, MM:SS, or HH:MM:SS."
        )

    last_match = _RANGE_SECONDS_RE.fullmatch(parts[-1])
    if not last_match:
        raise SubmasterError(
            f"Invalid time value '{raw_value}'. Use optional milliseconds as .mmm or ,mmm."
        )

    seconds = int(last_match.group("seconds"))
    millis_raw = last_match.group("millis") or ""
    millis = int(millis_raw.ljust(3, "0")) if millis_raw else 0

    hours = 0
    minutes = 0

    if len(parts) == 2:
        if not _RANGE_INTEGER_RE.fullmatch(parts[0]):
            raise SubmasterError(f"Invalid time value '{raw_value}'.")
        minutes = int(parts[0])
        if seconds >= 60:
            raise SubmasterError(f"Seconds must be below 60 in '{raw_value}'.")
    elif len(parts) == 3:
        if not _RANGE_INTEGER_RE.fullmatch(parts[0]) or not _RANGE_INTEGER_RE.fullmatch(parts[1]):
            raise SubmasterError(f"Invalid time value '{raw_value}'.")
        hours = int(parts[0])
        minutes = int(parts[1])
        if minutes >= 60 or seconds >= 60:
            raise SubmasterError(
                f"Minutes and seconds must be below 60 in '{raw_value}'."
            )

    total_seconds = hours * 3_600 + minutes * 60 + seconds
    return total_seconds * 1_000 + millis


def resolve_clip_range(raw_range: list[str] | None) -> tuple[int, int] | None:
    """Resolve the optional `--range` argument into millisecond bounds.

    :param raw_range: Raw parser values for `--range`, if provided.
    :type raw_range: list[str] | None
    :returns: Inclusive start / exclusive end millisecond bounds, or `None`.
    :rtype: tuple[int, int] | None
    :raises SubmasterError: If the range is malformed or empty.
    """
    if raw_range is None:
        return None

    start_ms = parse_time_value(raw_range[0])
    end_ms = parse_time_value(raw_range[1])
    if end_ms <= start_ms:
        raise SubmasterError("Range end must be greater than range start.")
    return start_ms, end_ms


def resolve_output_path(source_path: Path, requested_output: str | None) -> Path:
    """Resolve the destination subtitle path from the user-provided output value.

    :param source_path: Source media path used to derive the default SRT filename.
    :type source_path: pathlib.Path
    :param requested_output: Raw `--output` value supplied by the user, if any.
    :type requested_output: str | None
    :returns: Normalized subtitle path with an `.srt` suffix.
    :rtype: pathlib.Path
    """
    # With no explicit output, keep the subtitle next to the source file
    # using the same stem.
    if not requested_output:
        return source_path.with_suffix(".srt")

    # Expand user shortcuts first, then track whether the input was intended
    # as a directory path.
    output_path = Path(requested_output).expanduser()
    trailing_separators = {sep for sep in (os.sep, os.altsep, "/", "\\") if sep}
    has_trailing_separator = any(requested_output.endswith(sep) for sep in trailing_separators)
    directory_output_path = output_path

    # Preserve a non-existent directory target when the caller explicitly
    # ends the path with a separator.
    if has_trailing_separator and not output_path.exists():
        trimmed_output = requested_output.rstrip("/\\")
        if trimmed_output:
            directory_output_path = Path(trimmed_output).expanduser()

    # Existing directories always receive a derived subtitle filename.
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{source_path.stem}.srt"

    # Non-SRT outputs are treated either as directory intents or rewritten
    # with the expected extension.
    if output_path.suffix.lower() != ".srt":
        return (
            directory_output_path / f"{source_path.stem}.srt"
            if has_trailing_separator
            else output_path.with_suffix(".srt")
        )

    # A valid .srt path can be used as-is.
    return output_path


def resolve_batch_output_dir(requested_output: str | None) -> Path | None:
    """Resolve the optional batch output root directory.

    :param requested_output: Raw `--output` value supplied by the user, if any.
    :type requested_output: str | None
    :returns: Output directory used for batch subtitles, or `None` to keep them beside each source.
    :rtype: pathlib.Path | None
    :raises SubmasterError: If the user points batch mode at a single `.srt` file.
    """
    if not requested_output:
        return None

    output_path = Path(requested_output).expanduser()
    trailing_separators = {sep for sep in (os.sep, os.altsep, "/", "\\") if sep}
    has_trailing_separator = any(requested_output.endswith(sep) for sep in trailing_separators)

    if output_path.exists():
        if not output_path.is_dir():
            raise SubmasterError(
                "Batch mode requires --output to be a directory path, not a single .srt file."
            )
        return output_path

    if output_path.suffix.lower() == ".srt" and not has_trailing_separator:
        raise SubmasterError(
            "Batch mode requires --output to be a directory path, not a single .srt file."
        )

    if has_trailing_separator:
        trimmed_output = requested_output.rstrip("/\\")
        if trimmed_output:
            return Path(trimmed_output).expanduser()
    return output_path


def resolve_input_path(raw_input: str) -> Path:
    """Resolve a CLI input argument into a local filesystem path.

    :param raw_input: Raw positional input argument provided by the caller.
    :type raw_input: str
    :returns: Resolved local path.
    :rtype: pathlib.Path
    :raises SubmasterError: If an SMB URL is malformed or not mounted locally.
    """
    parsed = urlparse(raw_input)
    if parsed.scheme.lower() != "smb":
        return Path(raw_input).expanduser().resolve()

    host = parsed.hostname
    relative_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if host is None or not relative_parts:
        raise SubmasterError(
            f"Malformed SMB URL. Expected smb://host/share/path, got: {raw_input}"
        )

    share = relative_parts[0]
    subpath_parts = relative_parts[1:]
    gvfs_base = Path(
        f"/run/user/{os.getuid()}/gvfs/smb-share:server={host},share={share}"
    )
    candidate_path = gvfs_base.joinpath(*subpath_parts).resolve()
    if candidate_path.exists():
        return candidate_path

    raise SubmasterError(
        "SMB URL is not available as a local mount path. "
        f"Expected mounted location: {candidate_path}"
    )


def discover_batch_inputs(input_dir: Path) -> tuple[list[Path], int]:
    """Find direct child files that expose a video stream.

    :param input_dir: Folder whose regular files should be probed with `ffprobe`.
    :type input_dir: pathlib.Path
    :returns: Resolved video file paths and a count of skipped non-video or unreadable files.
    :rtype: tuple[list[pathlib.Path], int]
    """
    media_paths: list[Path] = []
    skipped_count = 0

    for candidate in sorted(input_dir.iterdir(), key=lambda path: path.name.lower()):
        if not candidate.is_file():
            continue
        try:
            if has_video_stream(candidate):
                media_paths.append(candidate.resolve())
            else:
                skipped_count += 1
        except SubmasterError:
            skipped_count += 1

    return media_paths, skipped_count


def resolve_batch_output_path(source_path: Path, output_dir: Path | None) -> Path:
    """Build the destination subtitle path for one batch input."""
    if output_dir is None:
        return source_path.with_suffix(".srt")
    return output_dir / f"{source_path.stem}.srt"


def build_batch_jobs(
    input_paths: list[Path],
    output_dir: Path | None,
) -> list[tuple[Path, Path]]:
    """Pair each batch input with its resolved subtitle output path.

    :param input_paths: Video files selected for batch processing.
    :type input_paths: list[pathlib.Path]
    :param output_dir: Optional shared output directory.
    :type output_dir: pathlib.Path | None
    :returns: Ordered `(input_path, output_path)` pairs.
    :rtype: list[tuple[pathlib.Path, pathlib.Path]]
    :raises SubmasterError: If two inputs would write the same subtitle file.
    """
    jobs: list[tuple[Path, Path]] = []
    seen_outputs: dict[Path, Path] = {}

    for input_path in input_paths:
        output_path = resolve_batch_output_path(input_path, output_dir).resolve()
        conflicting_input = seen_outputs.get(output_path)
        if conflicting_input is not None:
            raise SubmasterError(
                "Batch mode would write multiple inputs to the same subtitle path: "
                f"{conflicting_input.name} and {input_path.name} -> {output_path}"
            )
        seen_outputs[output_path] = input_path
        jobs.append((input_path, output_path))

    return jobs


def build_processing_summary(
    args: argparse.Namespace,
    clip_range: tuple[int, int] | None,
) -> str:
    """Render a one-line summary for the current transcription job."""
    summary = f"Whisper: {args.wmodel} | Language: {args.language} | Device: {args.device}"
    if clip_range is not None:
        summary += f" | Range: {args.range[0]} -> {args.range[1]}"
    if args.max_context != DEFAULT_WHISPER_MAX_CONTEXT:
        summary += f" | Max context: {args.max_context}"
    if args.vad_model:
        summary += f" | VAD: {args.vad_model}"
    if args.transcribe:
        transcript_mode = "raw + cleanup" if args.cleanup else "raw"
        summary += f" | Transcribe: {transcript_mode}"
    if args.translate:
        summary += f" | Translate to: {args.translate} ({args.tmodel})"
    return summary


def build_transcription_output_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    """Build the raw and cleaned transcription sidecar paths for one media file."""
    base_name = input_path.stem
    raw_path = output_path.with_name(f"{base_name}_transcript.txt")
    cleaned_path = output_path.with_name(f"{base_name}_cleanup.txt")
    return raw_path, cleaned_path


def build_translation_output_path(output_path: Path, language_code: str) -> Path:
    """Build the translated subtitle sidecar path for one target language."""
    normalized_code = language_code.strip().lower() or "translated"
    return output_path.with_name(f"{output_path.stem}_{normalized_code}.srt")


def prepare_processing_resources(
    args: argparse.Namespace,
    models_dir: Path,
    console: Console,
) -> ProcessingResources:
    """Resolve models and native runners that can be reused across jobs."""
    whisper_runner = WhisperCppRunner(console=console)
    whisper_model_path = ensure_model_available(args.wmodel, models_dir, console)
    vad_model_path = (
        ensure_vad_model_available(args.vad_model, models_dir, console)
        if args.vad_model
        else None
    )

    llama_runner: LlamaCppRunner | None = None
    if args.translate or (args.transcribe and args.cleanup):
        llama_runner = LlamaCppRunner(console=console)

    translator: SubtitleTranslator | None = None
    if args.translate:
        translation_model_path = ensure_translation_model_available(
            args.tmodel,
            models_dir,
            console,
        )
        translator = SubtitleTranslator(
            console=console,
            runner=llama_runner,
            model_path=translation_model_path,
            target_language=args.translate,
            source_language=args.language,
            requested_device=args.device,
            threads=args.threads,
        )

    transcript_cleaner: TranscriptCleaner | None = None
    if args.transcribe and args.cleanup:
        cleanup_model_path = ensure_cleanup_model_available(
            DEFAULT_CLEANUP_MODEL,
            models_dir,
            console,
        )
        transcript_cleaner = TranscriptCleaner(
            console=console,
            runner=llama_runner,
            model_path=cleanup_model_path,
            requested_device=args.device,
            threads=args.threads,
        )

    return ProcessingResources(
        whisper_runner=whisper_runner,
        whisper_model_path=whisper_model_path,
        vad_model_path=vad_model_path,
        translator=translator,
        transcript_cleaner=transcript_cleaner,
    )


def process_media_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    console: Console,
    resources: ProcessingResources,
    clip_range: tuple[int, int] | None,
    *,
    job_index: int | None = None,
    job_total: int | None = None,
    skip_video_validation: bool = False,
) -> None:
    """Run the full subtitle pipeline for one media file."""
    work_dir: Path | None = None
    transcript_path, cleanup_path = build_transcription_output_paths(input_path, output_path)
    translated_output_path = (
        build_translation_output_path(output_path, resources.translator.target_language.code)
        if resources.translator is not None
        else None
    )

    try:
        if output_path.exists() and not args.overwrite:
            raise SubmasterError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )
        if translated_output_path is not None and translated_output_path.exists() and not args.overwrite:
            raise SubmasterError(
                f"Translated subtitle file already exists: {translated_output_path}. Use --overwrite to replace it."
            )
        if args.transcribe and transcript_path.exists() and not args.overwrite:
            raise SubmasterError(
                f"Transcription file already exists: {transcript_path}. Use --overwrite to replace it."
            )
        if args.transcribe and args.cleanup and cleanup_path.exists() and not args.overwrite:
            raise SubmasterError(
                f"Cleanup file already exists: {cleanup_path}. Use --overwrite to replace it."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detail = f"'{input_path.name}' -> '{output_path.name}'"
        if job_index is not None and job_total is not None:
            detail = f"[{job_index}/{job_total}] {detail}"
            console.emphasis(detail)
        else:
            console.line(detail)

        if not skip_video_validation and not has_video_stream(input_path):
            raise SubmasterError("Input must contain a video stream.")
        console.info(build_processing_summary(args, clip_range))

        work_dir = create_work_dir()
        # Keep intermediate filenames simple so native runtimes never have to
        # deal with user-facing punctuation or long source names.
        audio_path = work_dir / "input.wav"
        console.info("Extracting audio track from video.")
        clip_start_ms = clip_range[0] if clip_range is not None else None
        clip_duration_ms = (
            clip_range[1] - clip_range[0]
            if clip_range is not None
            else None
        )
        extract_audio(
            input_path,
            audio_path,
            console,
            clip_start_ms=clip_start_ms,
            clip_duration_ms=clip_duration_ms,
        )

        output_base = work_dir / "transcript"
        raw_srt_path = resources.whisper_runner.run(
            audio_path=audio_path,
            model_path=resources.whisper_model_path,
            output_base=output_base,
            language=args.language,
            requested_device=args.device,
            threads=args.threads,
            max_context=args.max_context,
            vad_model_path=resources.vad_model_path,
            show_timings=args.show_timings,
            show_model_info=args.show_model_info,
        )

        raw_srt = raw_srt_path.read_text(encoding="utf-8")
        normalized_srt = normalize_srt(
            raw_srt,
            offset_ms=clip_start_ms or 0,
        )
        translated_srt = (
            resources.translator.translate_srt(normalized_srt)
            if resources.translator is not None
            else None
        )
        transcript_text = (
            render_transcript_from_srt(
                raw_srt,
                offset_ms=clip_start_ms or 0,
            )
            if args.transcribe
            else None
        )
        cleaned_transcript_text = (
            resources.transcript_cleaner.clean_text(transcript_text)
            if transcript_text is not None and resources.transcript_cleaner is not None
            else None
        )

        output_path.write_text(normalized_srt, encoding="utf-8", newline="")
        if translated_srt is not None and translated_output_path is not None:
            translated_output_path.write_text(translated_srt, encoding="utf-8", newline="")
        if transcript_text is not None:
            transcript_path.write_text(transcript_text, encoding="utf-8", newline="")
        if cleaned_transcript_text is not None:
            cleanup_path.write_text(cleaned_transcript_text, encoding="utf-8", newline="")

        if args.keep_audio:
            kept_audio = output_path.with_name(f"{output_path.stem}.normalized.wav")
            shutil.copy2(audio_path, kept_audio)
            console.success(f"Kept normalized audio at {kept_audio}")

        console.success(f"Subtitle written to {output_path}")
        if translated_srt is not None and translated_output_path is not None:
            console.success(f"Translated subtitle written to {translated_output_path}")
        if transcript_text is not None:
            console.success(f"Transcription written to {transcript_path}")
        if cleaned_transcript_text is not None:
            console.success(f"Cleanup written to {cleanup_path}")
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


def ensure_runtime_dependencies() -> None:
    """Validate that required external binaries are available on `PATH`.

    :raises SubmasterError: If `ffmpeg` or `ffprobe` cannot be located.
    """
    # Fail early with a user-facing error before any media or model work starts.
    missing = [
        command for command in ("ffmpeg", "ffprobe") if shutil.which(command) is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise SubmasterError(f"Missing required external tool(s): {joined}.")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI workflow from argument parsing through subtitle generation.

    :param argv: Optional argument list used instead of `sys.argv[1:]`.
    :type argv: list[str] | None
    :returns: Process exit status code for the CLI invocation.
    :rtype: int
    """
    # Parse CLI options once at the top so the rest of the function works
    # from validated arguments.
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        console.banner(">>> SubMaster <<<")
        # Verify external tools and input/output paths before doing any
        # expensive setup.
        ensure_runtime_dependencies()
        clip_range = resolve_clip_range(args.range)
        models_dir = Path(args.models_dir).expanduser().resolve()

        if args.batch and args.chapters:
            raise SubmasterError("--chapters cannot be used together with --batch.")

        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            raise SubmasterError(f"Input file does not exist: {input_path}")

        if args.batch:
            if not input_path.is_dir():
                raise SubmasterError(
                    f"Batch mode requires a folder input, but got: {input_path}"
                )

            output_dir = resolve_batch_output_dir(args.output)
            if output_dir is not None:
                output_dir = output_dir.resolve()

            console.info(f"Scanning batch folder: {input_path}")
            batch_inputs, skipped_count = discover_batch_inputs(input_path)
            if not batch_inputs:
                raise SubmasterError(
                    f"No video files were found in batch folder: {input_path}"
                )

            jobs = build_batch_jobs(batch_inputs, output_dir)
            skipped_suffix = (
                f"; skipped {skipped_count} non-video or unreadable file(s)"
                if skipped_count
                else ""
            )
            console.info(
                f"Found {len(jobs)} video file(s) to process{skipped_suffix}."
            )
            console.banner("----------------------------------")

            resources = prepare_processing_resources(args, models_dir, console)
            failures = 0

            for index, (media_path, output_path) in enumerate(jobs, start=1):
                try:
                    process_media_file(
                        media_path,
                        output_path,
                        args,
                        console,
                        resources,
                        clip_range,
                        job_index=index,
                        job_total=len(jobs),
                        skip_video_validation=True,
                    )
                except SubmasterError as exc:
                    failures += 1
                    console.dismiss_progress()
                    console.error(f"[{index}/{len(jobs)}] {media_path.name}: {exc}")

            console.banner("----------------------------------")
            if failures:
                console.warn(
                    f"Batch finished with {len(jobs) - failures} success(es) and {failures} failure(s)."
                )
                return 1

            console.success(
                f"Batch finished successfully: {len(jobs)} subtitle file(s) written."
            )
            return 0

        if input_path.is_dir():
            raise SubmasterError(
                "Input path is a directory. Use --batch to process a folder."
            )

        output_path = resolve_output_path(input_path, args.output).resolve()
        if not has_video_stream(input_path):
            raise SubmasterError("Input must contain a video stream.")
        resources = prepare_processing_resources(args, models_dir, console)
        process_media_file(
            input_path,
            output_path,
            args,
            console,
            resources,
            clip_range,
            skip_video_validation=True,
        )

        if args.chapters:
            chapters_path = Path(args.chapters).expanduser().resolve()
            if not chapters_path.exists():
                raise SubmasterError(f"Chapter file does not exist: {chapters_path}")
            # e.g. video.mp4 → video_chapters.mp4, written next to the original
            chapters_output = input_path.with_stem(input_path.stem + "_chapters")
            embed_chapters(input_path, chapters_path, chapters_output, console)

        return 0
    except KeyboardInterrupt:
        # Map user cancellation to the conventional shell exit code.
        console.dismiss_progress()
        console.error("Operation cancelled by user.")
        return 130
    except SubmasterError as exc:
        # Convert expected CLI failures into a clean user message and a
        # non-zero exit status.
        console.dismiss_progress()
        console.error(str(exc))
        return 1
