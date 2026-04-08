import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.cli import (
    build_batch_jobs,
    build_parser,
    main,
    parse_time_value,
    resolve_batch_output_dir,
    resolve_clip_range,
    resolve_input_path,
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

    def test_parser_accepts_transcript_flag(self) -> None:
        """Verify that plain-text transcript output can be enabled from the CLI."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.mp4",
                "--transcript",
            ]
        )

        self.assertTrue(args.transcript)

    def test_parser_accepts_batch_flag(self) -> None:
        """Verify that folder processing can be enabled from the CLI."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "videos",
                "--batch",
            ]
        )

        self.assertTrue(args.batch)

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

    def test_resolve_batch_output_dir_rejects_single_srt_path(self) -> None:
        """Verify that batch mode only accepts directory outputs."""
        with self.assertRaisesRegex(SubmasterError, "directory path"):
            resolve_batch_output_dir("subs/output.srt")

    def test_resolve_input_path_maps_smb_url_to_gvfs_mount(self) -> None:
        """Verify that SMB URLs resolve to the expected GVFS-mounted local path."""
        smb_url = "smb://jupiter.local/magia/Istruzioni/Outside%20The%20Box/Instructions"
        expected_path = Path(
            "/run/user/1000/gvfs/smb-share:server=jupiter.local,share=magia"
        ) / "Istruzioni" / "Outside The Box" / "Instructions"

        with patch("submaster.cli.os.getuid", return_value=1000):
            with patch("pathlib.Path.exists", return_value=True):
                resolved = resolve_input_path(smb_url)

        self.assertEqual(resolved, expected_path.resolve())

    def test_resolve_input_path_rejects_malformed_smb_url(self) -> None:
        """Verify malformed SMB URLs fail with a user-facing validation error."""
        with self.assertRaisesRegex(SubmasterError, "Malformed SMB URL"):
            resolve_input_path("smb://jupiter.local")

    def test_build_batch_jobs_rejects_duplicate_output_stems(self) -> None:
        """Verify that two batch inputs cannot target the same output subtitle path."""
        input_paths = [
            Path("/tmp/movie.mp4"),
            Path("/tmp/movie.mkv"),
        ]

        with self.assertRaisesRegex(SubmasterError, "same subtitle path"):
            build_batch_jobs(input_paths, None)

    def test_main_rejects_directory_without_batch(self) -> None:
        """Verify that directory inputs require explicit batch mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "videos"
            input_dir.mkdir()

            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                exit_code = main([str(input_dir)])

        self.assertEqual(exit_code, 1)

    def test_main_batch_processes_every_detected_video_file(self) -> None:
        """Verify that batch mode processes each direct child video file once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "videos"
            output_dir = Path(tmpdir) / "subs"
            input_dir.mkdir()
            alpha_path = input_dir / "alpha.mp4"
            bravo_path = input_dir / "bravo.customext"
            notes_path = input_dir / "notes.txt"
            alpha_path.write_bytes(b"fake")
            bravo_path.write_bytes(b"fake")
            notes_path.write_text("ignore me", encoding="utf-8")

            processed_jobs: list[tuple[Path, Path, int, int, bool]] = []

            def fake_has_video_stream(candidate: Path) -> bool:
                return candidate.name != "notes.txt"

            def fake_process_media_file(
                input_path,
                output_path,
                args,
                console,
                resources,
                clip_range,
                *,
                job_index=None,
                job_total=None,
                skip_video_validation=False,
            ) -> None:
                processed_jobs.append(
                    (
                        input_path,
                        output_path,
                        job_index,
                        job_total,
                        skip_video_validation,
                    )
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", side_effect=fake_has_video_stream):
                    with patch("submaster.cli.prepare_processing_resources", return_value=SimpleNamespace()):
                        with patch("submaster.cli.process_media_file", side_effect=fake_process_media_file):
                            exit_code = main(
                                [
                                    str(input_dir),
                                    "--batch",
                                    "--output",
                                    str(output_dir),
                                ]
                            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                processed_jobs,
                [
                    (alpha_path.resolve(), (output_dir / "alpha.srt").resolve(), 1, 2, True),
                    (bravo_path.resolve(), (output_dir / "bravo.srt").resolve(), 2, 2, True),
                ],
            )
            self.assertTrue((output_dir / "alpha.srt").exists())
            self.assertTrue((output_dir / "bravo.srt").exists())

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
                output_base=work_dir / "transcript",
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

    def test_main_writes_plain_text_transcript_when_requested(self) -> None:
        """Verify that `--transcript` writes a companion `.txt` dialogue file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            raw_srt_path = Path(tmpdir) / "generated.srt"
            output_path = Path(tmpdir) / "output.srt"
            transcript_path = output_path.with_suffix(".txt")
            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()
            input_path.write_bytes(b"fake")
            raw_srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nhow are\nyou?\n",
                encoding="utf-8",
            )

            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", return_value=True):
                    with patch("submaster.cli.create_work_dir", return_value=work_dir):
                        with patch("submaster.cli.extract_audio", return_value=work_dir / "audio.wav"):
                            with patch("submaster.cli.ensure_model_available", return_value=Path(tmpdir) / "whisper.bin"):
                                with patch("submaster.cli.WhisperCppRunner") as whisper_runner_cls:
                                    whisper_runner = whisper_runner_cls.return_value
                                    whisper_runner.run.return_value = raw_srt_path
                                    exit_code = main(
                                        [
                                            str(input_path),
                                            "--output",
                                            str(output_path),
                                            "--transcript",
                                            "--overwrite",
                                        ]
                                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                transcript_path.read_text(encoding="utf-8"),
                "hello how are you?\n",
            )

    def test_main_dismisses_active_progress_before_keyboard_interrupt_error(self) -> None:
        """Verify cancelled runs clear the progress bar before printing the error."""
        events: list[str] = []
        console = SimpleNamespace(
            banner=lambda _message: None,
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
