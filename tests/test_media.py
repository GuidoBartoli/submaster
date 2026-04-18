import tempfile
import unittest
from pathlib import Path

from submaster.errors import SubmasterError
from submaster.media import parse_chapters


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

    def test_parse_chapters_rejects_out_of_range_minutes(self) -> None:
        """Verify that minute values above 59 are rejected even though the regex matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(tmpdir, "00:60:00 Invalid\n")
            with self.assertRaisesRegex(SubmasterError, "Wrong chapter format"):
                parse_chapters(path)

    def test_parse_chapters_rejects_out_of_range_seconds(self) -> None:
        """Verify that second values above 59 are rejected even though the regex matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_chapters(tmpdir, "00:00:99 Invalid\n")
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


if __name__ == "__main__":
    unittest.main()
