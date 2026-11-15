import tempfile
from contextlib import ExitStack
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from submaster.cli import main
from submaster.errors import SubmasterError


class AudioInputTests(unittest.TestCase):
    """Verify audio output rules independently of installed models and runtimes."""

    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.input_path = self.root / "recording.mp3"
        self.input_path.write_bytes(b"original audio")
        self.work_dir = self.root / "work"
        self.work_dir.mkdir()
        self.raw_srt_path = self.work_dir / "transcript.srt"
        self.raw_srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello there.\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nHow are you?\n",
            encoding="utf-8",
        )
        self.console = Mock()
        self.resources = SimpleNamespace(
            whisper_runner=Mock(),
            whisper_model_path=self.root / "whisper.bin",
            vad_model_path=self.root / "vad.bin",
            translator=None,
            transcript_cleaner=None,
        )
        self.resources.whisper_runner.run.return_value = self.raw_srt_path
        self.stack.enter_context(patch("submaster.cli.Console", return_value=self.console))
        self.stack.enter_context(patch("submaster.cli.ensure_runtime_dependencies"))
        self.stack.enter_context(patch("submaster.cli.has_video_stream", return_value=False))
        self.stack.enter_context(patch("submaster.cli.has_audio_stream", return_value=True))
        self.prepare = self.stack.enter_context(patch(
            "submaster.cli.prepare_processing_resources", return_value=self.resources,
        ))
        self.create_work = self.stack.enter_context(patch(
            "submaster.cli.create_work_dir", return_value=self.work_dir,
        ))
        self.extract = self.stack.enter_context(patch(
            "submaster.cli.extract_audio", side_effect=self._extract_audio,
        ))
        self.embed = self.stack.enter_context(patch("submaster.cli.embed_chapters"))

    @staticmethod
    def _extract_audio(source, destination, console, **kwargs) -> Path:
        destination.write_bytes(b"normalized audio")
        return destination

    def test_audio_without_transcribe_creates_no_output_or_model_work(self) -> None:
        """Even optional output flags must stay inactive without --transcribe."""
        output = self.root / "ignored" / "output.srt"
        exit_code = main([
            str(self.input_path), "-o", str(output), "--cleanup", "--summarize",
            "--translate", "it", "--keep-audio", "--overwrite",
        ])

        self.assertEqual(exit_code, 0)
        self.prepare.assert_not_called()
        self.create_work.assert_not_called()
        self.extract.assert_not_called()
        self.embed.assert_not_called()
        self.assertFalse(output.parent.exists())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["recording.mp3", "work"])

    def test_audio_transcript_ignores_output_and_existing_subtitles_and_chapters(self) -> None:
        """Only the transcript destination matters; video sidecars remain untouched."""
        output = self.root / "ignored" / "output.srt"
        old_subtitle = self.input_path.with_suffix(".srt")
        old_subtitle.write_text("existing subtitle", encoding="utf-8")
        self.input_path.with_suffix(".chp").write_text("invalid chapters", encoding="utf-8")

        exit_code = main([str(self.input_path), "--transcribe", "-o", str(output)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            (self.root / "recording_transcript.txt").read_text(encoding="utf-8"),
            "Hello there.\nHow are you?\n",
        )
        self.assertEqual(self.input_path.read_bytes(), b"original audio")
        self.assertEqual(old_subtitle.read_text(encoding="utf-8"), "existing subtitle")
        self.assertFalse(output.parent.exists())
        self.assertFalse(self.work_dir.exists())
        self.assertFalse((self.root / "recording_cleanup.txt").exists())
        self.embed.assert_not_called()

    def test_audio_translation_cleanup_summary_and_range_remain_available(self) -> None:
        """Audio runs retain both languages, cleaned prose, and summaries as text."""
        translator = Mock()
        translator.target_language.code = "it"
        translator.translate_srt.return_value = (
            "1\n00:01:00,000 --> 00:01:01,000\nCiao.\n\n"
            "2\n00:01:01,000 --> 00:01:02,000\nCome stai?\n"
        )
        cleaner = Mock()
        cleaner.clean_text.return_value = "Hello there. How are you?\n"
        cleaner.summarize_text.return_value = "A greeting.\n"
        self.resources.translator = translator
        self.resources.transcript_cleaner = cleaner

        exit_code = main([
            str(self.input_path), "--transcribe", "--cleanup", "--summarize",
            "--translate", "italian", "--range", "60", "65", "--keep-audio",
        ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            (self.root / "recording_transcript_it.txt").read_text(encoding="utf-8"),
            "Ciao.\nCome stai?\n",
        )
        self.assertEqual(
            (self.root / "recording_cleanup.txt").read_text(encoding="utf-8"),
            cleaner.clean_text.return_value,
        )
        self.assertEqual(
            (self.root / "recording_summary.txt").read_text(encoding="utf-8"),
            "A greeting.\n",
        )
        cleaner.clean_text.assert_called_once_with("Hello there.\nHow are you?\n")
        cleaner.summarize_text.assert_called_once_with("Hello there. How are you?\n")
        translator.translate_srt.assert_called_once()
        self.assertIn("00:01:00,000", translator.translate_srt.call_args.args[0])
        self.assertEqual(self.extract.call_args.kwargs["clip_start_ms"], 60_000)
        self.assertEqual(self.extract.call_args.kwargs["clip_duration_ms"], 5_000)
        self.assertEqual((self.root / "recording.normalized.wav").read_bytes(), b"normalized audio")
        self.assertEqual(self.input_path.read_bytes(), b"original audio")
        self.assertFalse(list(self.root.glob("*.srt")))
        self.assertFalse(self.work_dir.exists())

    def test_existing_audio_outputs_require_overwrite(self) -> None:
        """Each enabled transcript output is protected before audio extraction."""
        self.resources.translator = Mock()
        self.resources.translator.target_language.code = "it"
        self.resources.transcript_cleaner = Mock()
        for filename in (
            "recording_transcript.txt", "recording_transcript_it.txt",
            "recording_cleanup.txt", "recording_summary.txt",
        ):
            with self.subTest(filename=filename):
                destination = self.root / filename
                destination.write_text("existing", encoding="utf-8")

                exit_code = main([
                    str(self.input_path), "--transcribe", "--cleanup", "--summarize",
                    "--translate", "it",
                ])

                self.assertEqual(exit_code, 1)
                self.assertEqual(destination.read_text(encoding="utf-8"), "existing")
                self.create_work.assert_not_called()
                destination.unlink()

    def test_overwrite_replaces_transcript_and_preserves_source(self) -> None:
        """--overwrite must replace text outputs without writing over the audio."""
        transcript = self.root / "recording_transcript.txt"
        transcript.write_text("existing", encoding="utf-8")

        self.assertEqual(main([str(self.input_path), "--transcribe", "--overwrite"]), 0)

        self.assertEqual(transcript.read_text(encoding="utf-8"), "Hello there.\nHow are you?\n")
        self.assertEqual(self.input_path.read_bytes(), b"original audio")
        self.assertFalse(list(self.root.glob("*.srt")))

    def test_failed_audio_processing_removes_intermediates_without_outputs(self) -> None:
        """Native failures retain the source and remove the temporary WAV and SRT."""
        self.resources.whisper_runner.run.side_effect = SubmasterError("transcription failed")

        self.assertEqual(main([str(self.input_path), "--transcribe"]), 1)

        self.assertFalse(self.work_dir.exists())
        self.assertEqual(list(self.root.iterdir()), [self.input_path])
        self.assertEqual(self.input_path.read_bytes(), b"original audio")


if __name__ == "__main__":
    unittest.main()
