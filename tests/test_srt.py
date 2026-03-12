import unittest

from submaster.errors import SubmasterError
from submaster.srt import format_timestamp, normalize_srt, parse_srt


class SrtTests(unittest.TestCase):
    """Exercise SRT parsing and rendering helpers."""

    def test_format_timestamp(self) -> None:
        """Verify that millisecond timestamps render in canonical SRT format."""
        self.assertEqual(format_timestamp(3_723_456), "01:02:03,456")

    def test_normalize_srt_reindexes_and_normalizes_newlines(self) -> None:
        """Verify that normalization fixes numbering and line-ending style."""
        raw = (
            "7\n"
            "00:00:00.000 --> 00:00:01.500\n"
            "hello world\n\n"
            "9\n"
            "00:00:01,700 --> 00:00:03,000\n"
            "second line\n"
        )
        normalized = normalize_srt(raw)
        expected = (
            "1\r\n"
            "00:00:00,000 --> 00:00:01,500\r\n"
            "hello world\r\n\r\n"
            "2\r\n"
            "00:00:01,700 --> 00:00:03,000\r\n"
            "second line\r\n"
        )
        self.assertEqual(normalized, expected)

    def test_normalize_srt_applies_optional_timestamp_offset(self) -> None:
        """Verify that normalization can shift cue timings onto the source timeline."""
        raw = "1\n00:00:00,500 --> 00:00:01,500\nhello\n"

        normalized = normalize_srt(raw, offset_ms=10_000)

        self.assertEqual(
            normalized,
            "1\r\n00:00:10,500 --> 00:00:11,500\r\nhello\r\n",
        )

    def test_parse_srt_rejects_reverse_ranges(self) -> None:
        """Verify that cues ending before they start are rejected."""
        with self.assertRaises(SubmasterError):
            parse_srt("1\n00:00:02,000 --> 00:00:01,000\nbad\n")


if __name__ == "__main__":
    unittest.main()
