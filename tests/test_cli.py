import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.cli import build_parser, main, resolve_output_path


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
