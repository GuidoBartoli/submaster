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
GREEN = "\033[38;5;42m"
YELLOW = "\033[38;5;220m"
RED = "\033[38;5;196m"


def _enable_windows_virtual_terminal() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

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
        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("TERM_PROGRAM")
        )
    except Exception:
        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("TERM_PROGRAM")
        )


def _supports_color() -> bool:
    if not (sys.stderr.isatty() and sys.stdout.isatty()):
        return False
    if os.name == "nt":
        return _enable_windows_virtual_terminal()
    term = os.environ.get("TERM", "")
    return bool(term and term.lower() != "dumb")


def _style(enabled: bool, color: str, text: str) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes:.1f} B"


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class Console:
    def __init__(self) -> None:
        self.color = _supports_color()
        self.stream = sys.stderr
        self._lock = threading.Lock()

    def _write(self, message: str) -> None:
        with self._lock:
            self.stream.write(message)
            self.stream.flush()

    def line(self, message: str = "") -> None:
        self._write(f"{message}\n")

    def banner(self, title: str, detail: str | None = None) -> None:
        heading = _style(self.color, BOLD + BLUE, title)
        self.line(heading)
        if detail:
            self.line(_style(self.color, DIM, detail))

    def info(self, message: str) -> None:
        self.line(f"{_style(self.color, BLUE, '[INFO]')} {message}")

    def note(self, message: str) -> None:
        self.line(f"{_style(self.color, CYAN, '[STEP]')} {message}")

    def success(self, message: str) -> None:
        self.line(f"{_style(self.color, GREEN, '[ OK ]')} {message}")

    def warn(self, message: str) -> None:
        self.line(f"{_style(self.color, YELLOW, '[WARN]')} {message}")

    def error(self, message: str) -> None:
        self.line(f"{_style(self.color, RED, '[ERR!]')} {message}")

    def progress(self, label: str, total: float | None, unit: str = "") -> "ProgressBar":
        return ProgressBar(self, label=label, total=total, unit=unit)

    def spinner(self, label: str) -> "Spinner":
        return Spinner(self, label=label)


@dataclass
class ProgressBar:
    console: Console
    label: str
    total: float | None
    unit: str = ""

    def __post_init__(self) -> None:
        self._start = time.monotonic()
        self._last_render = 0.0
        self._closed = False
        self._last_line_length = 0
        self._width = max(20, min(40, shutil.get_terminal_size((100, 20)).columns - 48))
        self._render(0.0, "")

    def update(self, completed: float, extra: str = "") -> None:
        if self._closed:
            return
        now = time.monotonic()
        if now - self._last_render < 0.05 and self.total and completed < self.total:
            return
        self._render(completed, extra)

    def finish(self, completed: float | None = None, extra: str = "") -> None:
        if self._closed:
            return
        final_value = self.total if completed is None and self.total is not None else completed or 0.0
        self._render(final_value, extra, final=True)
        self._closed = True
        self.console._write("\n")

    def _render(self, completed: float, extra: str, final: bool = False) -> None:
        total = self.total
        elapsed = format_seconds(time.monotonic() - self._start)
        if total and total > 0:
            ratio = max(0.0, min(1.0, completed / total))
            filled = int(ratio * self._width)
            bar = "#" * filled + "-" * (self._width - filled)
            percent = f"{ratio * 100:6.2f}%"
            current_label = f"{completed:,.1f}/{total:,.1f}{self.unit}"
        else:
            pulse = int((time.monotonic() * 8) % self._width)
            bar_chars = ["-"] * self._width
            bar_chars[pulse] = "#"
            bar = "".join(bar_chars)
            percent = "   -- "
            current_label = f"{completed:,.1f}{self.unit}"

        prefix = _style(self.console.color, CYAN, f"[{self.label}]")
        suffix = f"{current_label} {elapsed}"
        if extra:
            suffix = f"{suffix} {extra}"
        line = f"\r{prefix} |{bar}| {percent} {suffix}"
        if final:
            line += " done"
        visible_length = max(0, len(line) - 1)
        if visible_length < self._last_line_length:
            line += " " * (self._last_line_length - visible_length)
        self.console._write(line)
        self._last_line_length = visible_length
        self._last_render = time.monotonic()


class Spinner:
    def __init__(self, console: Console, label: str) -> None:
        self.console = console
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        self._stop.set()
        self._thread.join()
        elapsed = format_seconds(time.monotonic() - self._start)
        status = "[OK]" if exc is None else "[ERR]"
        color = GREEN if exc is None else RED
        label = _style(self.console.color, color, status)
        self.console._write(f"\r{label} {self.label} ({elapsed}){' ' * 20}\n")

    def _run(self) -> None:
        frames = "|/-\\"
        index = 0
        while not self._stop.is_set():
            elapsed = format_seconds(time.monotonic() - self._start)
            prefix = _style(self.console.color, CYAN, f"[{frames[index % len(frames)]}]")
            self.console._write(f"\r{prefix} {self.label} ({elapsed})")
            time.sleep(0.1)
            index += 1
