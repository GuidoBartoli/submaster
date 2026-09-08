import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.errors import SubmasterError
from submaster.media import (
    _build_chapter_metadata,
    _write_chapter_video,
    build_chapter_output_path,
    calculate_legacy_video_bitrate,
    extract_audio,
    has_video_stream,
    parse_chapters,
    probe_bitrate_bps,
    probe_duration_seconds,
)


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

    def test_probe_bitrate_bps_returns_valid_container_bitrate(self) -> None:
        """Verify the source bitrate can be used to size legacy conversions."""
        result = SimpleNamespace(
            returncode=0,
            stdout='{"format":{"bit_rate":"624556"}}',
            stderr="",
        )

        with patch("submaster.media.subprocess.run", return_value=result):
            bitrate = probe_bitrate_bps(Path("/tmp/input.rmvb"))

        self.assertEqual(bitrate, 624_556)

    def test_calculate_legacy_video_bitrate_reserves_audio_and_overhead(self) -> None:
        """Verify converted output targets approximately the source file size."""
        with patch("submaster.media.probe_bitrate_bps", return_value=624_556):
            bitrate = calculate_legacy_video_bitrate(Path("/tmp/input.rmvb"))

        self.assertEqual(bitrate, 520_556)

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

    def test_chapter_output_uses_supported_container_for_common_inputs(self) -> None:
        """Verify legacy containers use MKV while MP4 retains its extension."""
        expected_paths = {
            "movie.rmvb": "movie_chapters.mkv",
            "movie.avi": "movie_chapters.mkv",
            "movie.mp4": "movie_chapters.mp4",
            "movie.mpg": "movie_chapters.mkv",
        }

        for input_name, expected_name in expected_paths.items():
            with self.subTest(input_name=input_name):
                self.assertEqual(
                    build_chapter_output_path(Path("/tmp") / input_name),
                    Path("/tmp") / expected_name,
                )

    def test_write_chapter_video_stream_copies_mp4(self) -> None:
        """Verify a compatible MP4 is written without re-encoding its streams."""
        result = SimpleNamespace(returncode=0, stderr="")

        with patch("submaster.media.subprocess.run", return_value=result) as run_mock:
            encoding_mode = _write_chapter_video(
                Path("/tmp/movie.mp4"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/movie_chapters.mp4"),
            )

        self.assertEqual(encoding_mode, "copy")
        command = run_mock.call_args.args[0]
        self.assertIn("-map_chapters", command)
        self.assertIn("-codec", command)
        self.assertNotIn("aac", command)

    def test_write_chapter_video_retries_mkv_with_aac_audio(self) -> None:
        """Verify legacy audio is converted when Matroska rejects stream-copy."""
        results = [
            SimpleNamespace(returncode=1, stderr="unsupported codec"),
            SimpleNamespace(returncode=0, stderr=""),
        ]

        with patch("submaster.media.subprocess.run", side_effect=results) as run_mock:
            encoding_mode = _write_chapter_video(
                Path("/tmp/movie.avi"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/movie_chapters.mkv"),
            )

        self.assertEqual(encoding_mode, "audio")
        self.assertEqual(run_mock.call_count, 2)
        retry_command = run_mock.call_args.args[0]
        self.assertIn("-c:v", retry_command)
        self.assertIn("-c:a", retry_command)
        self.assertIn("aac", retry_command)

    def test_write_chapter_video_transcodes_realmedia_video_and_audio(self) -> None:
        """Verify RMVB avoids the unplayable RV40 stream-copy path."""
        result = SimpleNamespace(returncode=0, stderr="")

        with patch("submaster.media.subprocess.run", return_value=result) as run_mock:
            encoding_mode = _write_chapter_video(
                Path("/tmp/movie.rmvb"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/movie_chapters.mkv"),
                target_video_bitrate=520_556,
            )

        self.assertEqual(encoding_mode, "full")
        self.assertEqual(run_mock.call_count, 1)
        command = run_mock.call_args.args[0]
        self.assertIn("libx264", command)
        self.assertIn("aac", command)
        self.assertIn("-b:v", command)
        self.assertIn("520556", command)

    def test_write_chapter_video_uses_full_conversion_as_final_fallback(self) -> None:
        """Verify incompatible legacy video also gets a playable fallback."""
        results = [
            SimpleNamespace(returncode=1, stderr="copy failed"),
            SimpleNamespace(returncode=1, stderr="audio-only failed"),
            SimpleNamespace(returncode=0, stderr=""),
        ]

        with patch("submaster.media.subprocess.run", side_effect=results) as run_mock:
            encoding_mode = _write_chapter_video(
                Path("/tmp/movie.mpg"),
                Path("/tmp/metadata.txt"),
                Path("/tmp/movie_chapters.mkv"),
            )

        self.assertEqual(encoding_mode, "full")
        self.assertEqual(run_mock.call_count, 3)
        self.assertIn("libx264", run_mock.call_args.args[0])

    def test_write_chapter_video_surfaces_ffmpeg_error(self) -> None:
        """Verify final FFmpeg diagnostics are included in the CLI failure."""
        result = SimpleNamespace(returncode=1, stderr="muxer rejected codec")

        with patch("submaster.media.subprocess.run", return_value=result):
            with self.assertRaisesRegex(SubmasterError, "muxer rejected codec"):
                _write_chapter_video(
                    Path("/tmp/movie.avi"),
                    Path("/tmp/metadata.txt"),
                    Path("/tmp/movie_chapters.mkv"),
                )


if __name__ == "__main__":
    unittest.main()
