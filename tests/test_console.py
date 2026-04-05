import io
import unittest
from unittest.mock import patch

from submaster.console import BOLD, Console, ORANGE, RESET


class ConsoleTests(unittest.TestCase):
    """Exercise console formatting behavior around progress and log lines."""

    def test_info_rerenders_active_progress_on_separate_line(self) -> None:
        """Verify that info messages clear and restore an active progress bar."""
        console = Console()
        console.color = False
        console.stream = io.StringIO()

        with patch("submaster.console.shutil.get_terminal_size", return_value=__import__("os").terminal_size((100, 20))):
            progress = console.progress("llama", total=10, unit=" cues")
            progress.update(4)
            console.info("GPU mode requested via llama-completion (cuda).")

        output = console.stream.getvalue()

        self.assertIn("[INFO] GPU mode requested via llama-completion (cuda).\n", output)
        self.assertGreaterEqual(output.count("[llama]"), 2)
        self.assertFalse(output.endswith("[INFO] GPU mode requested via llama-completion (cuda).\n"))

    def test_selected_progress_labels_use_orange(self) -> None:
        """Verify ffmpeg/whisper/llama progress prefixes render in orange."""
        console = Console()
        console.color = True
        console.stream = io.StringIO()

        with patch("submaster.console.shutil.get_terminal_size", return_value=__import__("os").terminal_size((100, 20))):
            progress = console.progress("ffmpeg", total=10, unit="s")
            progress.update(1)

        output = console.stream.getvalue()

        self.assertIn(f"{ORANGE}[ffmpeg]{RESET}", output)

    def test_progress_can_hide_redundant_value_suffix(self) -> None:
        """Verify determinate progress can render only percent, elapsed time, and ETA."""
        console = Console()
        console.color = False
        console.stream = io.StringIO()

        with patch("submaster.console.shutil.get_terminal_size", return_value=__import__("os").terminal_size((100, 20))):
            with patch("submaster.console.time.monotonic", side_effect=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]):
                progress = console.progress("whisper", total=100, unit="%", show_value=False)
                progress.update(15)

        output = console.stream.getvalue()

        self.assertIn("[whisper]", output)
        self.assertIn("15.00% 00:01 ETA 00:05", output)
        self.assertNotIn("15.0/100.0%", output)

    def test_emphasis_renders_bold_when_color_is_enabled(self) -> None:
        """Verify emphasized lines use ANSI bold styling when available."""
        console = Console()
        console.color = True
        console.stream = io.StringIO()

        console.emphasis("[1/8] 'clip.mp4' -> 'clip.srt'")

        output = console.stream.getvalue()

        self.assertEqual(
            output,
            f"{BOLD}[1/8] 'clip.mp4' -> 'clip.srt'{RESET}\n",
        )


if __name__ == "__main__":
    unittest.main()
