import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.cli import (
    build_parser,
    main,
    parse_time_value,
    resolve_clip_range,
    resolve_output_path,
    resolve_vad_model_path,
)
from submaster.errors import SubmasterError


class CliTests(unittest.TestCase):
    """Exercise CLI parsing and high-level workflow orchestration."""

    def test_parser_accepts_optional_whisper_summary_flags(self) -> None:
        """Verify that whisper summary flags parse as simple booleans."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.mp4",
                "--show-timings",
                "--show-model-info",
            ]
        )

        self.assertTrue(args.show_timings)
        self.assertTrue(args.show_model_info)

    def test_parser_accepts_translation_flags(self) -> None:
        """Verify that translation-related CLI options are accepted together."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.mp4",
                "--translate-to",
                "it",
                "--translation-model",
                "large",
                "--llama-cli",
                "./llama.cpp/build/bin/llama-cli",
            ]
        )

        self.assertEqual(args.translate_to, "it")
        self.assertEqual(args.translation_model, "large")
        self.assertEqual(args.llama_cli, "./llama.cpp/build/bin/llama-cli")

    def test_parser_accepts_range_context_and_vad_flags(self) -> None:
        """Verify that clip-limited whisper options are parsed together."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.mp4",
                "--range",
                "00:10:00",
                "00:11:30.500",
                "--max-context",
                "0",
                "--vad-model",
                "silero-v6.2.0",
            ]
        )

        self.assertEqual(args.range, ["00:10:00", "00:11:30.500"])
        self.assertEqual(args.max_context, 0)
        self.assertEqual(args.vad_model, "silero-v6.2.0")

    def test_parse_time_value_accepts_flexible_timestamp_formats(self) -> None:
        """Verify that user-facing clip timestamps normalize to milliseconds."""
        self.assertEqual(parse_time_value("75.5"), 75_500)
        self.assertEqual(parse_time_value("01:15"), 75_000)
        self.assertEqual(parse_time_value("01:02:03,250"), 3_723_250)

    def test_resolve_clip_range_rejects_empty_or_reverse_ranges(self) -> None:
        """Verify that clip range bounds must advance forward."""
        with self.assertRaisesRegex(SubmasterError, "greater than"):
            resolve_clip_range(["00:01:00", "00:01:00"])

    def test_resolve_vad_model_path_accepts_existing_file(self) -> None:
        """Verify that explicit VAD file paths bypass named-model downloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "ggml-vad.bin"
            model_path.write_bytes(b"vad")

            resolved = resolve_vad_model_path(
                str(model_path),
                Path(tmpdir) / "models",
                SimpleNamespace(),
            )

        self.assertEqual(resolved, model_path.resolve())

    def test_main_rejects_audio_only_input(self) -> None:
        """Verify that the CLI rejects media files without a video stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            input_path.write_bytes(b"fake")

            # Stub the external dependency and probe checks so the test can focus
            # on the video-stream validation branch.
            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", return_value=False):
                    exit_code = main([str(input_path)])

        self.assertEqual(exit_code, 1)

    def test_resolve_output_path_accepts_windows_style_directory_separator(self) -> None:
        """Verify that Windows-style trailing separators are treated as directories."""
        source_path = Path("/tmp/input.mp4")

        resolved = resolve_output_path(source_path, "subs\\")

        self.assertEqual(resolved, Path("subs") / "input.srt")

    def test_main_invokes_translation_when_requested(self) -> None:
        """Verify that the translation pipeline runs when `--translate-to` is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            raw_srt_path = Path(tmpdir) / "generated.srt"
            output_path = Path(tmpdir) / "output.srt"
            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()
            input_path.write_bytes(b"fake")
            raw_srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                encoding="utf-8",
            )

            translator_instance = unittest.mock.Mock()
            translator_instance.translate_srt.return_value = (
                "1\r\n00:00:00,000 --> 00:00:01,000\r\nciao\r\n"
            )

            # Patch all external work so the test exercises orchestration only.
            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", return_value=True):
                    with patch("submaster.cli.create_work_dir", return_value=work_dir):
                        with patch("submaster.cli.extract_audio", return_value=work_dir / "audio.wav"):
                            with patch("submaster.cli.ensure_model_available", return_value=Path(tmpdir) / "whisper.bin"):
                                with patch("submaster.cli.ensure_translation_model_available", return_value=Path(tmpdir) / "HY-MT1.5-1.8B-Q4_K_M.gguf"):
                                    with patch("submaster.cli.WhisperCppRunner") as whisper_runner_cls:
                                        whisper_runner = whisper_runner_cls.return_value
                                        whisper_runner.run.return_value = raw_srt_path
                                        with patch("submaster.cli.LlamaCppRunner"):
                                            with patch("submaster.cli.SubtitleTranslator", return_value=translator_instance):
                                                exit_code = main(
                                                    [
                                                        str(input_path),
                                                        "--output",
                                                        str(output_path),
                                                        "--translate-to",
                                                        "it",
                                                        "--overwrite",
                                                    ]
                                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "1\n00:00:00,000 --> 00:00:01,000\nciao\n",
            )
            translator_instance.translate_srt.assert_called_once()

    def test_main_applies_range_offset_and_passes_whisper_context_options(self) -> None:
        """Verify that ranged transcription keeps source timestamps and runner options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            raw_srt_path = Path(tmpdir) / "generated.srt"
            output_path = Path(tmpdir) / "output.srt"
            work_dir = Path(tmpdir) / "work"
            models_dir = Path(tmpdir) / "models"
            vad_model_path = models_dir / "ggml-silero-v6.2.0.bin"
            work_dir.mkdir()
            input_path.write_bytes(b"fake")
            raw_srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                encoding="utf-8",
            )

            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", return_value=True):
                    with patch("submaster.cli.create_work_dir", return_value=work_dir):
                        with patch("submaster.cli.extract_audio", return_value=work_dir / "audio.wav") as extract_audio_mock:
                            with patch("submaster.cli.ensure_model_available", return_value=Path(tmpdir) / "whisper.bin"):
                                with patch("submaster.cli.ensure_vad_model_available", return_value=vad_model_path) as ensure_vad_model_mock:
                                    with patch("submaster.cli.WhisperCppRunner") as whisper_runner_cls:
                                        whisper_runner = whisper_runner_cls.return_value
                                        whisper_runner.run.return_value = raw_srt_path
                                        exit_code = main(
                                            [
                                                str(input_path),
                                                "--output",
                                                str(output_path),
                                                "--overwrite",
                                                "--range",
                                                "00:10:00",
                                                "00:10:05",
                                                "--max-context",
                                                "0",
                                                "--models-dir",
                                                str(models_dir),
                                                "--vad-model",
                                                "silero-v6.2.0",
                                            ]
                                        )

            self.assertEqual(exit_code, 0)
            ensure_vad_model_mock.assert_called_once_with(
                "silero-v6.2.0",
                models_dir.resolve(),
                unittest.mock.ANY,
            )
            extract_audio_mock.assert_called_once_with(
                input_path.resolve(),
                work_dir / "input.wav",
                unittest.mock.ANY,
                clip_start_ms=600_000,
                clip_duration_ms=5_000,
            )
            whisper_runner.run.assert_called_once_with(
                audio_path=work_dir / "input.wav",
                model_path=Path(tmpdir) / "whisper.bin",
                output_base=work_dir / "output",
                language="auto",
                requested_device="auto",
                threads=unittest.mock.ANY,
                max_context=0,
                vad_model_path=vad_model_path.resolve(),
                show_timings=False,
                show_model_info=False,
            )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "1\n00:10:00,000 --> 00:10:01,000\nhello\n",
            )

    def test_main_dismisses_active_progress_before_keyboard_interrupt_error(self) -> None:
        """Verify cancelled runs clear the progress bar before printing the error."""
        events: list[str] = []
        console = SimpleNamespace(
            dismiss_progress=lambda: events.append("dismiss"),
            error=lambda message: events.append(f"error:{message}"),
        )

        with patch("submaster.cli.Console", return_value=console):
            with patch("submaster.cli.ensure_runtime_dependencies", side_effect=KeyboardInterrupt):
                exit_code = main(["input.mp4"])

        self.assertEqual(exit_code, 130)
        self.assertEqual(events, ["dismiss", "error:Operation cancelled by user."])


if __name__ == "__main__":
    unittest.main()
