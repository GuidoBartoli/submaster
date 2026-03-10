from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import (
    LLAMA_CONTEXT_SIZE,
    LLAMA_MAX_BATCH_CHARS,
    LLAMA_N_GPU_LAYERS_ALL,
    LLAMA_N_GPU_LAYERS_CPU,
    LLAMA_REPEAT_PENALTY,
    LLAMA_TEMPERATURE,
    LLAMA_TOP_K,
    LLAMA_TOP_P,
)
from .console import Console
from .errors import SubmasterError


class LlamaCppRunner:
    def __init__(self, console: Console, cli_path: Path | None = None) -> None:
        self.console = console
        self.cli_path = self._resolve_cli_path(cli_path)
        self.supports_ngl_flag = self._supports_any_flag("-ngl", "--n-gpu-layers")
        self.supports_simple_io = self._supports_any_flag("--simple-io")
        self.supports_no_display_prompt = self._supports_any_flag("--no-display-prompt")
        self.supports_no_warmup = self._supports_any_flag("--no-warmup")
        self.gpu_backends = self._detect_gpu_backends_for(self.cli_path)
        self._announced_modes: set[str] = set()

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _windows_pathexts(self) -> tuple[str, ...]:
        raw_value = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        return tuple(ext.lower() for ext in raw_value.split(";") if ext.strip())

    def _is_executable_file(self, candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        if platform.system() == "Windows":
            return candidate.suffix.lower() in self._windows_pathexts()
        return os.access(candidate, os.X_OK)

    def _expand_explicit_candidate(self, candidate: Path) -> list[Path]:
        expanded = [candidate.expanduser()]
        if platform.system() != "Windows" or candidate.suffix:
            return expanded

        for suffix in self._windows_pathexts():
            expanded.append(candidate.with_suffix(suffix))
        return expanded

    def _path_command_names(self) -> tuple[str, ...]:
        return ("llama-cli", "main")

    def _local_command_names(self) -> tuple[str, ...]:
        base_names = self._path_command_names()
        if platform.system() != "Windows":
            return base_names
        exe_names = tuple(f"{name}.exe" for name in base_names)
        return exe_names + base_names

    def _common_build_directories(self, root: Path) -> list[Path]:
        directories = [
            root / "llama.cpp" / "build" / "bin",
            root / "llama.cpp" / "build",
            root / "build" / "bin",
            root / "build",
            root,
        ]
        config_names = ("Release", "RelWithDebInfo", "Debug")
        expanded: list[Path] = []
        seen: set[Path] = set()
        for directory in directories:
            for candidate in (directory, *(directory / config for config in config_names)):
                if candidate in seen:
                    continue
                seen.add(candidate)
                expanded.append(candidate)
        return expanded

    def _candidate_paths(self, directories: list[Path], names: tuple[str, ...]) -> list[Path]:
        return [directory / name for directory in directories for name in names]

    def _resolve_cli_path(self, cli_path: Path | None) -> Path:
        explicit_candidates: list[Path] = []
        discovered_candidates: list[Path] = []

        if cli_path:
            explicit_candidates.extend(self._expand_explicit_candidate(cli_path))

        env_candidate = os.environ.get("LLAMA_CPP_CLI")
        if env_candidate:
            explicit_candidates.extend(self._expand_explicit_candidate(Path(env_candidate)))

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            prefix_path = Path(conda_prefix)
            discovered_candidates.extend(
                self._candidate_paths([prefix_path / "bin"], self._local_command_names())
            )
            discovered_candidates.extend(
                self._candidate_paths(
                    [
                        prefix_path / "Scripts",
                        prefix_path / "Library" / "bin",
                    ],
                    self._local_command_names(),
                )
            )

        for command_name in self._path_command_names():
            resolved = shutil.which(command_name)
            if resolved:
                discovered_candidates.append(Path(resolved))

        cwd = Path.cwd()
        project_root = self._project_root()
        discovered_candidates.extend(
            self._candidate_paths(self._common_build_directories(cwd), self._local_command_names())
        )
        discovered_candidates.extend(
            self._candidate_paths(self._common_build_directories(project_root), self._local_command_names())
        )

        searched = explicit_candidates + discovered_candidates

        for candidate in self._existing_candidates(explicit_candidates):
            if candidate and self._is_executable_file(candidate):
                return candidate.resolve()

        ranked_discovered = self._rank_candidates_by_gpu(discovered_candidates)
        if ranked_discovered:
            return ranked_discovered[0].resolve()

        searched_text = "\n".join(f"  - {item}" for item in searched if item)
        raise SubmasterError(
            "Unable to find a llama.cpp executable.\n"
            "Install or build llama.cpp, put 'llama-cli' on PATH, or place an executable path "
            "in --llama-cli / LLAMA_CPP_CLI.\n"
            "If you are using Conda, install 'conda-forge::llama.cpp'.\n"
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
            if self._is_executable_file(candidate):
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

    def _supports_any_flag(self, *flags: str) -> bool:
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
            if any(flag in help_text for flag in flags):
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
            cli_path.parent / "lib",
            cli_path.parent.parent,
            cli_path.parent.parent / "lib",
            cli_path.parent.parent / "Library" / "bin",
            cli_path.parent.parent / "Library" / "lib",
            cli_path.parent.parent / "DLLs",
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

    def _inspect_linked_ggml_libraries(self) -> tuple[set[str], list[str]]:
        if platform.system() != "Linux":
            return set(), []

        try:
            result = subprocess.run(
                ["ldd", str(self.cli_path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return set(), []

        if result.returncode != 0:
            return set(), []

        lines = [line.strip() for line in result.stdout.splitlines() if "ggml" in line]
        linked_backends: set[str] = set()
        for line in lines:
            lowered = line.lower()
            if "libggml-cuda" in lowered:
                linked_backends.add("cuda")
            if "libggml-vulkan" in lowered:
                linked_backends.add("vulkan")
            if "libggml-opencl" in lowered:
                linked_backends.add("opencl")
            if "libggml-metal" in lowered:
                linked_backends.add("metal")
            if "libggml-hip" in lowered:
                linked_backends.add("hip")
            if "libggml-sycl" in lowered:
                linked_backends.add("sycl")
        return linked_backends, lines

    def _verify_gpu_runtime_linkage(self, requested: str, device: str) -> None:
        if requested != "gpu" or device != "gpu":
            return
        if platform.system() != "Linux":
            return

        linked_backends, linked_lines = self._inspect_linked_ggml_libraries()
        if not linked_lines or linked_backends:
            return

        detail = "\n".join(linked_lines)
        raise SubmasterError(
            "GPU was requested, but the selected llama.cpp executable is dynamically linked "
            "against CPU-only ggml libraries at runtime.\n"
            "This usually happens when a Conda environment provides CPU-only ggml libraries "
            "that shadow your local GPU-enabled llama.cpp build.\n"
            "Rebuild llama.cpp outside Conda with the desired GPU backend enabled, or make "
            "sure ldd resolves a GPU ggml library.\n"
            f"ldd output:\n{detail}"
        )

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
        return "gpu"

    def _estimated_max_tokens(self, prompt: str) -> int:
        prompt_chars = max(1, len(prompt))
        return max(256, min(2_048, int(prompt_chars / 2)))

    def _normalize_output(self, output: str) -> str:
        normalized = output.strip()
        if normalized.startswith("```"):
            parts = normalized.split("```")
            normalized = "\n".join(part for part in parts if part.strip() and not part.strip().isidentifier())
        return normalized.strip()

    def _announce_mode_once(self, device: str) -> None:
        if device in self._announced_modes:
            return

        backend_label = ", ".join(sorted(self.gpu_backends)) if self.gpu_backends else "cpu-only build"
        if device == "gpu":
            self.console.info(f"GPU mode requested via {self.cli_path.name} ({backend_label}).")
        else:
            self.console.info(f"CPU mode requested via {self.cli_path.name} ({backend_label}).")
        self._announced_modes.add(device)

    def run_prompt(
        self,
        model_path: Path,
        prompt: str,
        requested_device: str,
        threads: int,
    ) -> str:
        requested = requested_device.lower()
        device = self.resolve_device(requested_device)

        if requested in {"auto", "gpu"} and device == "cpu" and not self.gpu_backends and platform.system() != "Darwin":
            self.console.warn(
                "GPU was requested or auto-selected, but this llama.cpp executable appears CPU-only. "
                "Build llama.cpp with CUDA, Vulkan, Metal, HIP, OpenCL, or SYCL support to use the GPU."
            )
        elif requested == "gpu" and device == "cpu":
            self.console.warn("GPU was requested, but no usable GPU backend was detected. Falling back to CPU.")

        command = [
            str(self.cli_path),
            "-m",
            str(model_path),
            "-p",
            prompt,
            "-n",
            str(self._estimated_max_tokens(prompt)),
            "-c",
            str(LLAMA_CONTEXT_SIZE),
            "-t",
            str(max(1, threads)),
            "--temp",
            str(LLAMA_TEMPERATURE),
            "--top-k",
            str(LLAMA_TOP_K),
            "--top-p",
            str(LLAMA_TOP_P),
            "--repeat-penalty",
            str(LLAMA_REPEAT_PENALTY),
        ]

        if self.supports_simple_io:
            command.append("--simple-io")
        if self.supports_no_display_prompt:
            command.append("--no-display-prompt")
        if self.supports_no_warmup:
            command.append("--no-warmup")

        if self.supports_ngl_flag:
            gpu_layers = LLAMA_N_GPU_LAYERS_ALL if device == "gpu" else LLAMA_N_GPU_LAYERS_CPU
            command.extend(["-ngl", str(gpu_layers)])
        elif device == "cpu":
            self.console.warn("This llama.cpp build does not advertise '-ngl'; CPU forcing may be ignored.")

        self._announce_mode_once(device)
        self._verify_gpu_runtime_linkage(requested, device)

        with self.console.spinner("Running subtitle translation batch.") as _spinner:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            detail = "\n".join(
                part.strip()
                for part in (result.stdout or "", result.stderr or "")
                if part and part.strip()
            )
            raise SubmasterError(detail or "llama.cpp translation failed.")

        normalized_output = self._normalize_output(result.stdout or "")
        if not normalized_output:
            raise SubmasterError("llama.cpp finished without producing translated text.")
        return normalized_output
