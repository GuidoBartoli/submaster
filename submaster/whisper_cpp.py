from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from .console import Console, format_seconds
from .errors import SubmasterError


class WhisperCppRunner:
    PROGRESS_RE = re.compile(r"progress\s*=\s*(?P<percent>\d+)%")

    def __init__(self, console: Console, cli_path: Path | None = None) -> None:
        self.console = console
        self.cli_path = self._resolve_cli_path(cli_path)
        self.supports_no_gpu_flag = self._supports_flag("-ng")
        self.gpu_backends = self._detect_gpu_backends_for(self.cli_path)

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _resolve_cli_path(self, cli_path: Path | None) -> Path:
        explicit_candidates: list[Path] = []
        discovered_candidates: list[Path] = []
        bundled_candidates: list[Path] = []

        if cli_path:
            explicit_candidates.append(cli_path)

        env_candidate = os.environ.get("WHISPER_CPP_CLI")
        if env_candidate:
            explicit_candidates.append(Path(env_candidate))

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            prefix_path = Path(conda_prefix)
            discovered_candidates.extend(
                [
                    prefix_path / "bin" / "whisper-cli",
                    prefix_path / "bin" / "main",
                    prefix_path / "Scripts" / "whisper-cli.exe",
                    prefix_path / "Scripts" / "main.exe",
                ]
            )

        for command_name in ("whisper-cli", "main"):
            resolved = shutil.which(command_name)
            if resolved:
                discovered_candidates.append(Path(resolved))

        cwd = Path.cwd()
        project_root = self._project_root()

        # Check the common local build layouts before falling back to the bundled binary.
        discovered_candidates.extend(
            [
                cwd / "whisper.cpp" / "build" / "bin" / "whisper-cli",
                cwd / "whisper.cpp" / "build" / "bin" / "main",
                cwd / "whisper-cli",
                cwd / "main",
                cwd / "build" / "bin" / "whisper-cli",
                cwd / "build" / "bin" / "main",
            ]
        )

        bundled_candidates.extend(
            [
                project_root / "whisper" / "whisper-cli-gpu",
                project_root / "whisper" / "whisper-cli-cpu",
                project_root / "whisper" / "whisper-cli",
            ]
        )

        searched = explicit_candidates + discovered_candidates + bundled_candidates

        for candidate in self._existing_candidates(explicit_candidates):
            if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()

        ranked_discovered = self._rank_candidates_by_gpu(discovered_candidates)
        if ranked_discovered:
            return ranked_discovered[0].resolve()

        ranked_bundled = self._rank_candidates_by_gpu(bundled_candidates)
        if ranked_bundled:
            return ranked_bundled[0].resolve()

        searched_text = "\n".join(f"  - {item}" for item in searched if item)
        raise SubmasterError(
            "Unable to find a whisper.cpp executable.\n"
            "Install or build whisper.cpp, put 'whisper-cli' on PATH, or use the bundled fallback at './whisper/whisper-cli'.\n"
            "If you are using Conda, install 'conda-forge::whisper.cpp'.\n"
            "Note: the PyPI package named 'whisper-cli' is not the whisper.cpp binary used by this app.\n"
            f"Searched:\n{searched_text}"
        )

    def _existing_candidates(self, candidates: list[Path]) -> list[Path]:
        existing: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if not candidate:
                continue
            candidate = candidate.expanduser()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                existing.append(candidate)
        return existing

    def _rank_candidates_by_gpu(self, candidates: list[Path]) -> list[Path]:
        existing = self._existing_candidates(candidates)
        if not existing:
            return []

        gpu_candidates: list[Path] = []
        cpu_candidates: list[Path] = []
        for candidate in existing:
            if self._detect_gpu_backends_for(candidate):
                gpu_candidates.append(candidate)
            else:
                cpu_candidates.append(candidate)
        return gpu_candidates + cpu_candidates

    def _supports_flag(self, flag: str) -> bool:
        for help_flag in ("--help", "-h"):
            try:
                result = subprocess.run(
                    [str(self.cli_path), help_flag],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                continue
            help_text = (result.stdout or "") + "\n" + (result.stderr or "")
            if flag in help_text:
                return True
        return False

    def _detect_gpu_backends_for(self, cli_path: Path) -> set[str]:
        patterns = {
            "cuda": ("libggml-cuda*", "ggml-cuda*.dll", "ggml-cuda*.dylib"),
            "vulkan": ("libggml-vulkan*", "ggml-vulkan*.dll", "ggml-vulkan*.dylib"),
            "opencl": ("libggml-opencl*", "ggml-opencl*.dll", "ggml-opencl*.dylib"),
            "metal": ("libggml-metal*", "ggml-metal*.dll", "ggml-metal*.dylib"),
            "hip": ("libggml-hip*", "ggml-hip*.dll", "ggml-hip*.dylib"),
            "sycl": ("libggml-sycl*", "ggml-sycl*.dll", "ggml-sycl*.dylib"),
        }

        root_candidates = {
            cli_path.parent,
            cli_path.parent.parent,
            cli_path.parent.parent / "lib",
            cli_path.parent.parent / "src",
            cli_path.parent.parent / "bin",
        }

        detected: set[str] = set()
        for root in root_candidates:
            if not root.exists():
                continue
            for backend, backend_patterns in patterns.items():
                if backend in detected:
                    continue
                for pattern in backend_patterns:
                    try:
                        if any(root.rglob(pattern)):
                            detected.add(backend)
                            break
                    except OSError:
                        continue
        return detected

    def resolve_device(self, requested: str) -> str:
        normalized = requested.lower()
        if normalized not in {"auto", "cpu", "gpu"}:
            raise SubmasterError("Device must be one of: auto, cpu, gpu.")
        if normalized == "cpu":
            return "cpu"
        if normalized == "gpu" and not self.gpu_backends and platform.system() != "Darwin":
            return "cpu"
        if normalized == "gpu":
            return "gpu"
        if platform.system() == "Darwin":
            return "gpu"
        if not self.gpu_backends:
            return "cpu"
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
        requested = requested_device.lower()
        device = self.resolve_device(requested_device)
        backend_label = ", ".join(sorted(self.gpu_backends)) if self.gpu_backends else "cpu-only build"
        command = [
            str(self.cli_path),
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-of",
            str(output_base),
            "-osrt",
            "-np",
            "-pp",
            "-t",
            str(max(1, threads)),
        ]

        if language.lower() != "auto":
            command.extend(["-l", language])

        if requested in {"auto", "gpu"} and device == "cpu" and not self.gpu_backends and platform.system() != "Darwin":
            self.console.warn(
                "GPU was requested or auto-selected, but this whisper.cpp executable appears CPU-only. "
                "Build whisper.cpp with CUDA/Vulkan/OpenCL support to use the GPU."
            )
        elif requested == "gpu" and device == "cpu":
            self.console.warn("GPU was requested, but no usable GPU backend was detected. Falling back to CPU.")

        if device == "cpu" and self.supports_no_gpu_flag:
            command.append("-ng")
        elif device == "cpu" and not self.supports_no_gpu_flag:
            self.console.warn("This whisper.cpp build does not advertise '-ng'; CPU forcing may be ignored.")

        if device == "gpu":
            self.console.info(f"GPU mode requested via {self.cli_path.name} ({backend_label}).")
        else:
            self.console.info(f"CPU mode requested via {self.cli_path.name} ({backend_label}).")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress = self.console.progress("whisper", total=100, unit="%")
        output_lines: list[str] = []
        last_percent = 0
        started_at = time.monotonic()

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_lines.append(line)
            match = self.PROGRESS_RE.search(line)
            if not match:
                continue
            percent = int(match.group("percent"))
            last_percent = max(last_percent, min(percent, 100))
            elapsed = time.monotonic() - started_at
            if last_percent > 0:
                remaining = elapsed * (100 - last_percent) / last_percent
                extra = f"ETA {format_seconds(remaining)}"
            else:
                extra = "ETA --:--"
            progress.update(last_percent, extra=extra)

        return_code = process.wait()
        finish_extra = ""
        if last_percent > 0 and last_percent < 100:
            elapsed = time.monotonic() - started_at
            finish_extra = f"stopped at {last_percent}% after {format_seconds(elapsed)}"
        progress.finish(100 if return_code == 0 else last_percent, extra=finish_extra)

        if return_code != 0:
            detail = "\n".join(output_lines)
            raise SubmasterError(detail or "whisper.cpp transcription failed.")

        srt_path = output_base.with_suffix(".srt")
        if not srt_path.exists():
            raise SubmasterError("whisper.cpp finished without creating an SRT file.")

        return srt_path
