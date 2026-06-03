import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.errors import SubmasterError
from submaster.media import _build_chapter_metadata, extract_audio, has_video_stream, parse_chapters, probe_duration_seconds


class ParseChaptersTests(unittest.TestCase):
    """Exercise the plain-text chapter file parser."""

    def _write_chapters(self, tmpdir: str, content: str) -> Path:
        path = Path(tmpdir) / "chapters.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def test_parse_chapters_returns_correct_timestamps(self) -> None:
        """Verify that valid chapter lines produce the expected millisecond start values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(
                tmpdir,
                "00:00:00 Introduction\n"
                "00:23:20 Start\n"
                "01:04:44 Second Performance\n",
            )
            chapters = parse_chapters(path)

        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0], {"title": "Introduction", "start": 0})
        self.assertEqual(chapters[1], {"title": "Start", "start": 1_400_000})
        self.assertEqual(chapters[2], {"title": "Second Performance", "start": 3_884_000})

    def test_parse_chapters_skips_blank_lines(self) -> None:
        """Verify that empty and whitespace-only lines are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(
                tmpdir,
                "\n00:00:00 Intro\n\n   \n00:01:30 Part Two\n",
            )
            chapters = parse_chapters(path)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Intro")
        self.assertEqual(chapters[1]["title"], "Part Two")

    def test_parse_chapters_rejects_malformed_timestamp(self) -> None:
        """Verify that lines not matching HH:MM:SS raise SubmasterError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(tmpdir, "0:00 Bad format\n")
            with self.assertRaisesRegex(SubmasterError, "Wrong chapter format"):
                parse_chapters(path)

    def test_parse_chapters_rejects_out_of_range_time_fields(self) -> None:
        """Verify that minute and second values above 59 are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for raw_line in ("00:60:00 Invalid\n", "00:00:99 Invalid\n"):
                path = self._write_chapters(tmpdir, raw_line)
                with self.assertRaisesRegex(SubmasterError, "Wrong chapter format"):
                    parse_chapters(path)

    def test_parse_chapters_rejects_empty_file(self) -> None:
        """Verify that a chapter file with no valid entries raises SubmasterError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(tmpdir, "\n\n\n")
            with self.assertRaisesRegex(SubmasterError, "No chapters found"):
                parse_chapters(path)

    def test_parse_chapters_preserves_title_with_spaces(self) -> None:
        """Verify that chapter titles containing spaces are captured in full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(tmpdir, "00:40:30 First Performance Live\n")
            chapters = parse_chapters(path)

        self.assertEqual(chapters[0]["title"], "First Performance Live")

    def test_has_video_stream_reads_ffprobe_json(self) -> None:
        """Verify that video stream detection is based on ffprobe stream metadata."""
        result = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"codec_type":"audio"},{"codec_type":"video"}]}',
            stderr="",
        )

        with patch("submaster.media.subprocess.run", return_value=result) as run_mock:
            has_video = has_video_stream(Path("/tmp/input.mkv"))

        self.assertTrue(has_video)
        self.assertIn("-show_streams", run_mock.call_args.args[0])

    def test_probe_duration_seconds_returns_none_for_missing_or_invalid_duration(self) -> None:
        """Verify that optional duration metadata degrades to None when unusable."""
        for stdout in ('{"format":{}}', '{"format":{"duration":"unknown"}}'):
            result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            with patch("submaster.media.subprocess.run", return_value=result):
                self.assertIsNone(probe_duration_seconds(Path("/tmp/input.mp4")))

    def test_probe_helpers_raise_submaster_error_when_ffprobe_fails(self) -> None:
        """Verify that ffprobe failures surface stderr as a user-facing error."""
        result = SimpleNamespace(returncode=1, stdout="", stderr="bad media")

        with patch("submaster.media.subprocess.run", return_value=result):
            with self.assertRaisesRegex(SubmasterError, "bad media"):
                has_video_stream(Path("/tmp/input.mp4"))

    def test_extract_audio_builds_ranged_ffmpeg_command_and_reports_output(self) -> None:
        """Verify that clip range options are passed to ffmpeg and output is validated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "audio.wav"
            destination.write_bytes(b"wav")
            stdout_lines = iter(["out_time_ms=500000\n", "progress=end\n", ""])

            class DummyProcess:
                stdout = SimpleNamespace(readline=lambda: next(stdout_lines))
                stderr = SimpleNamespace(read=lambda: "")

                def poll(self) -> int | None:
                    return 0

                def wait(self) -> int:
                    return 0

            progress_events: list[tuple[str, float | None]] = []
            console = SimpleNamespace(
                info=lambda _message: None,
                progress=lambda label, total, unit="": SimpleNamespace(
                    update=lambda completed, extra="": progress_events.append(("update", completed)),
                    finish=lambda completed=None, extra="": progress_events.append(("finish", completed)),
                ),
            )

            with patch("submaster.media.subprocess.Popen", return_value=DummyProcess()) as popen_mock:
                resolved = extract_audio(
                    Path("/tmp/input.mp4"),
                    destination,
                    console,
                    clip_start_ms=1_500,
                    clip_duration_ms=2_000,
                )

        command = popen_mock.call_args.args[0]
        self.assertEqual(resolved, destination)
        self.assertIn("-ss", command)
        self.assertIn("1.500", command)
        self.assertIn("-t", command)
        self.assertIn("2.000", command)
        self.assertIn(("finish", 2.0), progress_events)

    def test_extract_audio_raises_when_ffmpeg_fails(self) -> None:
        """Verify that ffmpeg stderr is preserved on extraction failure."""
        stdout_lines = iter(["progress=end\n", ""])

        class DummyProcess:
            stdout = SimpleNamespace(readline=lambda: next(stdout_lines))
            stderr = SimpleNamespace(read=lambda: "no audio stream")

            def poll(self) -> int | None:
                return 1

            def wait(self) -> int:
                return 1

        console = SimpleNamespace(
            progress=lambda label, total, unit="": SimpleNamespace(
                update=lambda completed, extra="": None,
                finish=lambda completed=None, extra="": None,
            ),
        )

        with patch("submaster.media.probe_duration_seconds", return_value=None):
            with patch("submaster.media.subprocess.Popen", return_value=DummyProcess()):
                with self.assertRaisesRegex(SubmasterError, "no audio stream"):
                    extract_audio(Path("/tmp/input.mp4"), Path("/tmp/out.wav"), console)

    def test_build_chapter_metadata_sets_end_to_next_start_or_duration(self) -> None:
        """Verify that ffmetadata chapter end times align with the next chapter."""
        metadata = _build_chapter_metadata(
            [
                {"title": "Intro", "start": 0},
                {"title": "Middle", "start": 10_000},
            ],
            duration_ms=20_000,
        )

        self.assertIn("START=0\nEND=9999\ntitle=Intro", metadata)
        self.assertIn("START=10000\nEND=20000\ntitle=Middle", metadata)


if __name__ == "__main__":
    unittest.main()
