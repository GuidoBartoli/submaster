from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .config import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    DEFAULT_TRANSLATION_MODEL,
    MODEL_SPECS,
    TRANSLATION_MODEL_SPECS,
)
from .console import Console
from .errors import SubmasterError
from .llama_cpp import LlamaCppRunner
from .media import create_work_dir, extract_audio, has_video_stream
from .models import ensure_model_available, ensure_translation_model_available
from .srt import normalize_srt
from .translation import SubtitleTranslator
from .whisper_cpp import WhisperCppRunner


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
            "Create synchronized .srt subtitles from a video file using whisper.cpp, "
            "with optional offline translation via llama.cpp."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core input/output options define the source media file and final
    # subtitle location.
    parser.add_argument("input", help="Path to a video file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .srt file path or directory. Defaults to the input stem next to the source file.",
    )

    # Whisper-specific options control transcription quality, language
    # handling, and runtime selection.
    parser.add_argument(
        "-m",
        "--model",
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
        "--whisper-cli",
        help="Path to a whisper.cpp executable such as whisper-cli.",
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
        "--show-timings",
        action="store_true",
        help="Display the final whisper.cpp timing summary.",
    )
    parser.add_argument(
        "--show-model-info",
        action="store_true",
        help="Display whisper.cpp model-load metadata.",
    )

    # Translation options stay optional and are only used when the caller
    # asks for a second-language subtitle file.
    parser.add_argument(
        "--translate-to",
        help="Translate the generated subtitles into the given target language code or name.",
    )
    parser.add_argument(
        "--translation-model",
        default=DEFAULT_TRANSLATION_MODEL,
        choices=tuple(TRANSLATION_MODEL_SPECS),
        help="Tencent HY-MT translation model size to use when --translate-to is set.",
    )
    parser.add_argument(
        "--llama-cli",
        help="Path to a llama.cpp executable such as llama-cli.",
    )
    return parser


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
    work_dir: Path | None = None

    try:
        # Verify external tools and input/output paths before doing any
        # expensive setup.
        ensure_runtime_dependencies()

        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise SubmasterError(f"Input file does not exist: {input_path}")

        output_path = resolve_output_path(input_path, args.output).resolve()
        if output_path.exists() and not args.overwrite:
            raise SubmasterError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Surface a short execution summary before starting media inspection
        # and model loading.
        console.banner("Submaster", f"{input_path.name} -> {output_path.name}")
        if not has_video_stream(input_path):
            raise SubmasterError(
                "Input must contain a video stream."
            )
        summary = f"Whisper: {args.model} | Language: {args.language} | Device: {args.device}"
        if args.translate_to:
            summary += f" | Translate to: {args.translate_to} ({args.translation_model})"
        console.info(summary)

        # Prepare the transcription runner and ensure the requested whisper
        # model is ready locally.
        runner = WhisperCppRunner(
            console=console,
            cli_path=Path(args.whisper_cli).expanduser() if args.whisper_cli else None,
        )
        model_path = ensure_model_available(
            args.model, Path(args.models_dir).expanduser().resolve(), console
        )

        # Extract normalized mono audio into a temporary workspace before
        # invoking whisper.cpp.
        work_dir = create_work_dir()
        audio_path = work_dir / f"{input_path.stem}.wav"
        console.note("Extracting audio track from video.")
        extract_audio(input_path, audio_path, console)

        # Run transcription into the temp directory, then normalize the
        # emitted SRT content.
        output_base = work_dir / output_path.stem
        raw_srt_path = runner.run(
            audio_path=audio_path,
            model_path=model_path,
            output_base=output_base,
            language=args.language,
            requested_device=args.device,
            threads=args.threads,
            show_timings=args.show_timings,
            show_model_info=args.show_model_info,
        )

        normalized_srt = normalize_srt(raw_srt_path.read_text(encoding="utf-8"))
        if args.translate_to:
            # Translation reuses the normalized SRT text so the
            # target-language output keeps clean cue formatting.
            translation_model_path = ensure_translation_model_available(
                args.translation_model,
                Path(args.models_dir).expanduser().resolve(),
                console,
            )
            translation_runner = LlamaCppRunner(
                console=console,
                cli_path=Path(args.llama_cli).expanduser() if args.llama_cli else None,
            )
            translator = SubtitleTranslator(
                console=console,
                runner=translation_runner,
                model_path=translation_model_path,
                target_language=args.translate_to,
                source_language=args.language,
                requested_device=args.device,
                threads=args.threads,
            )
            normalized_srt = translator.translate_srt(normalized_srt)

        # Persist the final subtitle file and optionally keep the extracted
        # WAV for debugging or reuse.
        output_path.write_text(normalized_srt, encoding="utf-8", newline="")

        if args.keep_audio:
            kept_audio = output_path.with_name(f"{output_path.stem}.normalized.wav")
            shutil.copy2(audio_path, kept_audio)
            console.success(f"Kept normalized audio at {kept_audio}")

        console.success(f"Subtitle written to {output_path}")
        return 0
    except KeyboardInterrupt:
        # Map user cancellation to the conventional shell exit code.
        console.error("Operation cancelled by user.")
        return 130
    except SubmasterError as exc:
        # Convert expected CLI failures into a clean user message and a
        # non-zero exit status.
        console.error(str(exc))
        return 1
    finally:
        # Temporary media artifacts should never survive beyond the current CLI run.
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
