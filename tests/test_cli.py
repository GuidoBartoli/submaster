import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submaster.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_accepts_optional_whisper_summary_flags(self) -> None:
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

    def test_main_rejects_audio_only_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            input_path.write_bytes(b"fake")

            with patch("submaster.cli.ensure_runtime_dependencies", return_value=None):
                with patch("submaster.cli.has_video_stream", return_value=False):
                    exit_code = main([str(input_path)])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
