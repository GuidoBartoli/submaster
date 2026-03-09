from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .config import DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_THREADS, MODEL_SPECS
from .console import Console
from .errors import SubmasterError
from .media import create_work_dir, extract_audio, has_video_stream
from .models import ensure_model_available
from .srt import normalize_srt
from .whisper_cpp import WhisperCppRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submaster",
        description="Create synchronized .srt subtitles from a video file using whisper.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to a video file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .srt file path or directory. Defaults to the input stem next to the source file.",
    )
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
    return parser


def resolve_output_path(source_path: Path, requested_output: str | None) -> Path:
    if not requested_output:
        return source_path.with_suffix(".srt")

    output_path = Path(requested_output).expanduser()
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{source_path.stem}.srt"
    if output_path.suffix.lower() != ".srt":
        return (
            output_path / f"{source_path.stem}.srt"
            if requested_output.endswith(os.sep)
            else output_path.with_suffix(".srt")
        )
    return output_path


def ensure_runtime_dependencies() -> None:
    missing = [
        command for command in ("ffmpeg", "ffprobe") if shutil.which(command) is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise SubmasterError(f"Missing required external tool(s): {joined}.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()
    work_dir: Path | None = None

    try:
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

        console.banner("Submaster", f"{input_path.name} -> {output_path.name}")
        if not has_video_stream(input_path):
            raise SubmasterError(
                "Input must contain a video stream."
            )
        console.info(
            f"Model: {args.model} | Language: {args.language} | Device: {args.device}"
        )

        runner = WhisperCppRunner(
            console=console,
            cli_path=Path(args.whisper_cli).expanduser() if args.whisper_cli else None,
        )
        model_path = ensure_model_available(
            args.model, Path(args.models_dir).expanduser().resolve(), console
        )

        work_dir = create_work_dir()
        audio_path = work_dir / f"{input_path.stem}.wav"
        console.note("Extracting audio track from video.")
        extract_audio(input_path, audio_path, console)

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
        output_path.write_text(normalized_srt, encoding="utf-8", newline="")

        if args.keep_audio:
            kept_audio = output_path.with_name(f"{output_path.stem}.normalized.wav")
            shutil.copy2(audio_path, kept_audio)
            console.success(f"Kept normalized audio at {kept_audio}")

        console.success(f"Subtitle written to {output_path}")
        return 0
    except KeyboardInterrupt:
        console.error("Operation cancelled by user.")
        return 130
    except SubmasterError as exc:
        console.error(str(exc))
        return 1
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
