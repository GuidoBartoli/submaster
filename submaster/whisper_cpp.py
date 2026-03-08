from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .console import Console
from .errors import SubmasterError


class WhisperCppRunner:
    def __init__(self, console: Console, cli_path: Path | None = None) -> None:
        self.console = console
        self.cli_path = self._resolve_cli_path(cli_path)
        self.supports_no_gpu_flag = self._supports_flag("-ng")

    def _resolve_cli_path(self, cli_path: Path | None) -> Path:
        candidates: list[Path] = []
        if cli_path:
            candidates.append(cli_path)

        env_candidate = os.environ.get("WHISPER_CPP_CLI")
        if env_candidate:
            candidates.append(Path(env_candidate))

        for command_name in ("whisper-cli", "main"):
            resolved = shutil.which(command_name)
            if resolved:
                candidates.append(Path(resolved))

        cwd = Path.cwd()
        # Check the common local build layouts before giving up and asking for --whisper-cli.
        candidates.extend(
            [
                cwd / "whisper.cpp" / "build" / "bin" / "whisper-cli",
                cwd / "whisper.cpp" / "build" / "bin" / "main",
                cwd / "whisper-cli",
                cwd / "main",
                cwd / "build" / "bin" / "whisper-cli",
                cwd / "build" / "bin" / "main",
            ]
        )

        for candidate in candidates:
            if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()

        searched = "\n".join(f"  - {item}" for item in candidates if item)
        raise SubmasterError(
            "Unable to find a whisper.cpp executable.\n"
            "Install or build whisper.cpp, then put 'whisper-cli' on PATH or pass --whisper-cli.\n"
            f"Searched:\n{searched}"
        )

    def _supports_flag(self, flag: str) -> bool:
        for help_flag in ("--help", "-h"):
            result = subprocess.run(
                [str(self.cli_path), help_flag],
                capture_output=True,
                text=True,
                check=False,
            )
            help_text = (result.stdout or "") + "\n" + (result.stderr or "")
            if flag in help_text:
                return True
        return False

    def resolve_device(self, requested: str) -> str:
        normalized = requested.lower()
        if normalized not in {"auto", "cpu", "gpu"}:
            raise SubmasterError("Device must be one of: auto, cpu, gpu.")
        if normalized == "cpu":
            return "cpu"
        if normalized == "gpu":
            return "gpu"
        if platform.system() == "Darwin":
            return "gpu"
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            probe = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                return "gpu"
        return "cpu"

    def run(
        self,
        audio_path: Path,
        model_path: Path,
        output_base: Path,
        language: str,
        requested_device: str,
        threads: int,
    ) -> Path:
        device = self.resolve_device(requested_device)
        command = [
            str(self.cli_path),
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-of",
            str(output_base),
            "-osrt",
            "-t",
            str(max(1, threads)),
        ]

        if language.lower() != "auto":
            command.extend(["-l", language])

        if device == "cpu" and self.supports_no_gpu_flag:
            command.append("-ng")
        elif device == "cpu" and not self.supports_no_gpu_flag:
            self.console.warn("This whisper.cpp build does not advertise '-ng'; CPU forcing may be ignored.")

        if device == "gpu":
            self.console.info(f"GPU mode requested via {self.cli_path.name}.")
        else:
            self.console.info(f"CPU mode requested via {self.cli_path.name}.")

        with self.console.spinner("Transcribing with whisper.cpp"):
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            detail = "\n".join(line for line in (result.stderr or result.stdout).splitlines() if line.strip())
            raise SubmasterError(detail or "whisper.cpp transcription failed.")

        srt_path = output_base.with_suffix(".srt")
        if not srt_path.exists():
            raise SubmasterError("whisper.cpp finished without creating an SRT file.")

        return srt_path
