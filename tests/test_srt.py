import unittest

from submaster.errors import SubmasterError
from submaster.srt import format_timestamp, normalize_srt, parse_srt


class SrtTests(unittest.TestCase):
    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(3_723_456), "01:02:03,456")

    def test_normalize_srt_reindexes_and_normalizes_newlines(self) -> None:
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

    def test_parse_srt_rejects_reverse_ranges(self) -> None:
        with self.assertRaises(SubmasterError):
            parse_srt("1\n00:00:02,000 --> 00:00:01,000\nbad\n")


if __name__ == "__main__":
    unittest.main()
