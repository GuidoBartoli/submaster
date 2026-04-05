from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass


RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
CYAN = "\033[38;5;44m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[38;5;42m"
YELLOW = "\033[38;5;220m"
RED = "\033[38;5;196m"


def _enable_windows_virtual_terminal() -> bool:
    """Enable ANSI escape handling on supported Windows consoles.

    :returns: `True` when ANSI color output should work on Windows.
    :rtype: bool
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        # Attempt to enable VT processing on both stdout and stderr handles.
        kernel32 = ctypes.windll.kernel32
        enable_vt = 0x0004
        enabled = False
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            if handle in (0, -1):
                continue
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
                continue
            if kernel32.SetConsoleMode(handle, mode.value | enable_vt) == 0:
                continue
            enabled = True
        if enabled:
            return True
        # Fall back to common environment markers used by terminals that
        # already provide ANSI support.
        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("TERM_PROGRAM")
        )
    except Exception:
        # If ctypes setup fails, still detect known ANSI-capable terminals.
        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("TERM_PROGRAM")
        )


def _supports_color() -> bool:
    """Detect whether both output streams support ANSI color rendering.

    :returns: `True` when styled output should be emitted.
    :rtype: bool
    """
    # Only emit color when both streams are interactive to avoid polluting redirected output.
    if not (sys.stderr.isatty() and sys.stdout.isatty()):
        return False
    if os.name == "nt":
        return _enable_windows_virtual_terminal()
    term = os.environ.get("TERM", "")
    return bool(term and term.lower() != "dumb")


def _style(enabled: bool, color: str, text: str) -> str:
    """Wrap text in ANSI color codes when styling is enabled.

    :param enabled: Whether ANSI styling should be applied.
    :type enabled: bool
    :param color: ANSI prefix to prepend to the text.
    :type color: str
    :param text: Raw message content to style.
    :type text: str
    :returns: Styled or plain text depending on `enabled`.
    :rtype: str
    """
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def format_bytes(num_bytes: float) -> str:
    """Format a byte count using human-readable binary units.

    :param num_bytes: Byte count to format.
    :type num_bytes: float
    :returns: Human-readable byte string.
    :rtype: str
    """
    # Scale the value until it fits the current unit or the largest unit is reached.
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes:.1f} B"


def format_seconds(seconds: float) -> str:
    """Format seconds as `MM:SS` or `H:MM:SS`.

    :param seconds: Elapsed seconds to format.
    :type seconds: float
    :returns: Compact clock-style duration string.
    :rtype: str
    """
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class Console:
    """Render user-facing CLI output with optional ANSI styling."""

    def __init__(self) -> None:
        """Initialize the console writer and color capability state."""
        self.color = _supports_color()
        self.stream = sys.stderr
        self._lock = threading.Lock()
        self._active_progress: ProgressBar | None = None

    def _write(self, message: str) -> None:
        """Write raw text to the console stream in a thread-safe way.

        :param message: Text to write to the configured stream.
        :type message: str
        """
        # Progress bars and spinners update from multiple code paths, so writes are serialized.
        with self._lock:
            self.stream.write(message)
            self.stream.flush()

    def line(self, message: str = "") -> None:
        """Write a single newline-terminated message.

        :param message: Message content to print.
        :type message: str
        """
        active_progress = self._active_progress
        if active_progress is not None and not active_progress._closed:
            active_progress._clear_current_line()
            self._write(f"{message}\n")
            active_progress._render_current_state()
            return
        self._write(f"{message}\n")

    def banner(self, title: str, detail: str | None = None) -> None:
        """Print a styled banner for a major CLI stage.

        :param title: Primary banner title.
        :type title: str
        :param detail: Optional secondary line shown below the title.
        :type detail: str | None
        """
        heading = _style(self.color, BOLD + BLUE, title)
        self.line(heading)
        if detail:
            self.line(_style(self.color, DIM, detail))

    def info(self, message: str) -> None:
        """Print an informational message.

        :param message: Message content to print.
        :type message: str
        """
        self.line(f"{_style(self.color, BLUE, '[INFO]')} {message}")

    def note(self, message: str) -> None:
        """Print a step-oriented status message.

        :param message: Message content to print.
        :type message: str
        """
        self.line(f"{_style(self.color, CYAN, '[STEP]')} {message}")

    def emphasis(self, message: str) -> None:
        """Print an emphasized line for major per-item progress updates.

        :param message: Message content to print.
        :type message: str
        """
        self.line(_style(self.color, BOLD, message))

    def success(self, message: str) -> None:
        """Print a success message.

        :param message: Message content to print.
        :type message: str
        """
        self.line(f"{_style(self.color, GREEN, '[ OK ]')} {message}")

    def warn(self, message: str) -> None:
        """Print a warning message.

        :param message: Message content to print.
        :type message: str
        """
        self.line(f"{_style(self.color, YELLOW, '[WARN]')} {message}")

    def error(self, message: str) -> None:
        """Print an error message.

        :param message: Message content to print.
        :type message: str
        """
        self.line(f"{_style(self.color, RED, '[ERR!]')} {message}")

    def dismiss_progress(self) -> None:
        """Clear any active progress line without redrawing it."""
        active_progress = self._active_progress
        if active_progress is None:
            return
        if not active_progress._closed:
            active_progress._clear_current_line()
            active_progress._closed = True
        self._active_progress = None

    def progress(
        self,
        label: str,
        total: float | None,
        unit: str = "",
        show_value: bool = True,
    ) -> "ProgressBar":
        """Create a progress bar bound to this console.

        :param label: Short progress label shown in the prefix.
        :type label: str
        :param total: Total amount of work, or `None` for indeterminate progress.
        :type total: float | None
        :param unit: Optional unit suffix shown next to numeric values.
        :type unit: str
        :param show_value: Whether to show the raw completed/total value suffix.
        :type show_value: bool
        :returns: Progress bar instance connected to this console.
        :rtype: ProgressBar
        """
        return ProgressBar(
            self, label=label, total=total, unit=unit, show_value=show_value
        )

    def spinner(self, label: str) -> "Spinner":
        """Create a spinner context manager bound to this console.

        :param label: Status label displayed next to the spinner.
        :type label: str
        :returns: Spinner instance connected to this console.
        :rtype: Spinner
        """
        return Spinner(self, label=label)


@dataclass
class ProgressBar:
    """Render an updating progress bar on a single console line.

    :param console: Console used to write progress output.
    :type console: Console
    :param label: Prefix label shown before the bar.
    :type label: str
    :param total: Total amount of work, or `None` for indeterminate progress.
    :type total: float | None
    :param unit: Optional unit suffix shown next to numeric values.
    :type unit: str
    """

    console: Console
    label: str
    total: float | None
    unit: str = ""
    show_value: bool = True

    def __post_init__(self) -> None:
        """Initialize internal state and render the initial progress line."""
        # Size the bar against the current terminal width while keeping it readable in narrow terminals.
        self._start = time.monotonic()
        self._last_render = 0.0
        self._closed = False
        self._completed = 0.0
        self._extra = ""
        self._last_line_length = 0
        self._width = max(20, min(40, shutil.get_terminal_size((100, 20)).columns - 48))
        self.console._active_progress = self
        self._render(0.0, "")

    def update(self, completed: float, extra: str = "") -> None:
        """Refresh the progress display with the latest completion value.

        :param completed: Current amount of completed work.
        :type completed: float
        :param extra: Optional suffix appended to the rendered line.
        :type extra: str
        """
        if self._closed:
            return
        self._completed = completed
        self._extra = extra
        now = time.monotonic()
        # Throttle intermediate updates so fast producers do not flood the terminal.
        if now - self._last_render < 0.05 and self.total and completed < self.total:
            return
        self._render(completed, extra)

    def finish(self, completed: float | None = None, extra: str = "") -> None:
        """Mark the progress bar complete and move to the next terminal line.

        :param completed: Final completion value, or the configured total when omitted.
        :type completed: float | None
        :param extra: Optional suffix appended to the final line.
        :type extra: str
        """
        if self._closed:
            return
        final_value = self.total if completed is None and self.total is not None else completed or 0.0
        self._completed = final_value
        self._extra = extra
        self._render(final_value, extra, final=True)
        self._closed = True
        if self.console._active_progress is self:
            self.console._active_progress = None
        self.console._write("\n")

    def _render(self, completed: float, extra: str, final: bool = False) -> None:
        """Render the current progress state onto the terminal line.

        :param completed: Current amount of completed work.
        :type completed: float
        :param extra: Optional suffix appended to the rendered line.
        :type extra: str
        :param final: Whether this render represents the terminal state.
        :type final: bool
        """
        total = self.total
        elapsed_seconds = time.monotonic() - self._start
        elapsed = format_seconds(elapsed_seconds)
        if total and total > 0:
            # Determinate progress renders a proportional bar and percentage.
            ratio = max(0.0, min(1.0, completed / total))
            filled = int(ratio * self._width)
            bar = "#" * filled + "-" * (self._width - filled)
            percent = f"{ratio * 100:6.2f}%"
            current_label = (
                f"{completed:,.1f}/{total:,.1f}{self.unit}" if self.show_value else ""
            )
            eta = ""
            if 0 < completed < total and elapsed_seconds > 0:
                rate = completed / elapsed_seconds
                if rate > 0:
                    eta = f" ETA {format_seconds((total - completed) / rate)}"
        else:
            # Indeterminate progress renders a moving pulse instead of a percentage.
            pulse = int((time.monotonic() * 8) % self._width)
            bar_chars = ["-"] * self._width
            bar_chars[pulse] = "#"
            bar = "".join(bar_chars)
            percent = "   -- "
            current_label = f"{completed:,.1f}{self.unit}" if self.show_value else ""
            eta = ""

        prefix = _style(
            self.console.color, self._label_color(), f"[{self.label}]"
        )
        suffix_parts = [part for part in (current_label, elapsed) if part]
        if eta:
            suffix_parts.append(eta.strip())
        if extra:
            suffix_parts.append(extra)
        suffix = " ".join(suffix_parts)
        line = f"\r{prefix} |{bar}| {percent} {suffix}"
        if final:
            line += " done"
        visible_length = max(0, len(line) - 1)
        if visible_length < self._last_line_length:
            line += " " * (self._last_line_length - visible_length)
        self.console._write(line)
        self._last_line_length = visible_length
        self._last_render = time.monotonic()

    def _label_color(self) -> str:
        """Return the ANSI color used for the progress prefix label."""
        if self.label.lower() in {"ffmpeg", "whisper", "llama"}:
            return ORANGE
        return CYAN

    def _clear_current_line(self) -> None:
        """Erase the currently rendered progress line from the terminal."""
        self.console._write("\r" + (" " * self._last_line_length) + "\r")

    def _render_current_state(self) -> None:
        """Render the latest saved progress state after an out-of-band log line."""
        if self._closed:
            return
        self._render(self._completed, self._extra)


class Spinner:
    """Render a background spinner while a blocking task runs."""

    def __init__(self, console: Console, label: str) -> None:
        """Initialize a spinner bound to a console.

        :param console: Console used to write spinner output.
        :type console: Console
        :param label: Status label displayed next to the spinner.
        :type label: str
        """
        self.console = console
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0

    def __enter__(self) -> "Spinner":
        """Start the spinner thread when entering the context manager.

        :returns: The running spinner instance.
        :rtype: Spinner
        """
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        """Stop the spinner and render the final status line.

        :param exc_type: Exception type raised inside the context, if any.
        :type exc_type: type | None
        :param exc: Exception instance raised inside the context, if any.
        :type exc: BaseException | None
        :param _tb: Traceback object for the active exception, if any.
        :type _tb: object
        """
        # Emit a final OK or ERR marker once the wrapped task completes.
        self._stop.set()
        self._thread.join()
        elapsed = format_seconds(time.monotonic() - self._start)
        status = "[ OK ]" if exc is None else "[ERR!]"
        color = GREEN if exc is None else RED
        label = _style(self.console.color, color, status)
        self.console._write(f"\r{label} {self.label} ({elapsed}){' ' * 20}\n")

    def _run(self) -> None:
        """Animate the spinner until the enclosing context is finished."""
        frames = "|/-\\"
        index = 0
        while not self._stop.is_set():
            # Rewrite the same line repeatedly to give visual feedback during long subprocess calls.
            elapsed = format_seconds(time.monotonic() - self._start)
            prefix = _style(self.console.color, CYAN, f"[{frames[index % len(frames)]}]")
            self.console._write(f"\r{prefix} {self.label} ({elapsed})")
            time.sleep(0.1)
            index += 1
