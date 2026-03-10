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
    """Discover and invoke a local `whisper.cpp` command-line executable."""

    PROGRESS_RE = re.compile(r"progress\s*=\s*(?P<percent>\d+)%")
    TIMING_RE = re.compile(r"whisper_print_timings:\s*(?P<detail>.+)")
    MODEL_LOAD_RE = re.compile(r"whisper_model_load:\s*(?P<detail>.+)")

    def __init__(self, console: Console, cli_path: Path | None = None) -> None:
        """Initialize the runner and inspect the selected executable.

        :param console: Console used for status, warnings, and progress output.
        :type console: Console
        :param cli_path: Optional explicit path to a `whisper.cpp` executable.
        :type cli_path: pathlib.Path | None
        :raises SubmasterError: If no usable executable can be found.
        """
        self.console = console
        self.cli_path = self._resolve_cli_path(cli_path)
        # Cache feature probes once so transcription calls can assemble the
        # right command line without repeated subprocess invocations.
        self.supports_no_gpu_flag = self._supports_flag("-ng")
        self.gpu_backends = self._detect_gpu_backends_for(self.cli_path)

    def _project_root(self) -> Path:
        """Return the repository root used for local binary discovery.

        :returns: Repository root path.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parent.parent

    def _windows_pathexts(self) -> tuple[str, ...]:
        """Return executable suffixes recognized on Windows.

        :returns: Normalized `PATHEXT` suffix list.
        :rtype: tuple[str, ...]
        """
        raw_value = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        return tuple(
            ext.lower()
            for ext in raw_value.split(";")
            if ext.strip()
        )

    def _is_executable_file(self, candidate: Path) -> bool:
        """Check whether a candidate path is an executable file.

        :param candidate: Candidate filesystem path to inspect.
        :type candidate: pathlib.Path
        :returns: `True` when the path points to an executable file.
        :rtype: bool
        """
        if not candidate.is_file():
            return False
        if platform.system() == "Windows":
            return candidate.suffix.lower() in self._windows_pathexts()
        return os.access(candidate, os.X_OK)

    def _expand_explicit_candidate(self, candidate: Path) -> list[Path]:
        """Expand an explicit executable path into platform-specific variants.

        :param candidate: User-provided executable path.
        :type candidate: pathlib.Path
        :returns: Candidate paths to try in order.
        :rtype: list[pathlib.Path]
        """
        expanded = [candidate.expanduser()]
        if platform.system() != "Windows" or candidate.suffix:
            return expanded

        # Bare Windows executable names may rely on PATHEXT rather than an explicit suffix.
        for suffix in self._windows_pathexts():
            expanded.append(candidate.with_suffix(suffix))
        return expanded

    def _path_command_names(self) -> tuple[str, ...]:
        """Return executable names to search on `PATH`.

        :returns: Command names accepted for PATH discovery.
        :rtype: tuple[str, ...]
        """
        return ("whisper-cli", "main")

    def _local_command_names(self) -> tuple[str, ...]:
        """Return executable names to search in local build directories.

        :returns: Platform-appropriate local command names.
        :rtype: tuple[str, ...]
        """
        base_names = self._path_command_names()
        if platform.system() != "Windows":
            return base_names
        exe_names = tuple(f"{name}.exe" for name in base_names)
        return exe_names + base_names

    def _common_build_directories(self, root: Path) -> list[Path]:
        """List common `whisper.cpp` build output directories under a root.

        :param root: Root directory to inspect.
        :type root: pathlib.Path
        :returns: Deduplicated build directories to search.
        :rtype: list[pathlib.Path]
        """
        # Cover both in-repo builds and standard CMake configuration subdirectories.
        directories = [
            root / "whisper.cpp" / "build" / "bin",
            root / "whisper.cpp" / "build" / "src",
            root / "whisper.cpp" / "build",
            root / "build" / "bin",
            root / "build" / "src",
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
        """Build full candidate paths from directories and executable names.

        :param directories: Directories to search.
        :type directories: list[pathlib.Path]
        :param names: Executable basenames to combine with each directory.
        :type names: tuple[str, ...]
        :returns: Candidate executable paths.
        :rtype: list[pathlib.Path]
        """
        return [directory / name for directory in directories for name in names]

    def _bundled_candidate_paths(self, project_root: Path) -> list[Path]:
        """Return bundled fallback binary locations for supported platforms.

        :param project_root: Repository root containing the bundled binary folder.
        :type project_root: pathlib.Path
        :returns: Candidate bundled executable paths.
        :rtype: list[pathlib.Path]
        """
        if platform.system() == "Linux":
            names = ("whisper-cli-gpu", "whisper-cli-cpu", "whisper-cli")
        elif platform.system() == "Windows":
            names = ("whisper-cli-gpu.exe", "whisper-cli-cpu.exe", "whisper-cli.exe")
        else:
            names = ()
        return [project_root / "whisper" / name for name in names]

    def _resolve_cli_path(self, cli_path: Path | None) -> Path:
        """Resolve the best available `whisper.cpp` executable path.

        :param cli_path: Optional explicit executable path from the caller.
        :type cli_path: pathlib.Path | None
        :returns: Resolved executable path.
        :rtype: pathlib.Path
        :raises SubmasterError: If no usable executable can be found.
        """
        explicit_candidates: list[Path] = []
        discovered_candidates: list[Path] = []

        # Explicit CLI arguments and environment variables have highest priority.
        if cli_path:
            explicit_candidates.extend(self._expand_explicit_candidate(cli_path))

        env_candidate = os.environ.get("WHISPER_CPP_CLI")
        if env_candidate:
            explicit_candidates.extend(self._expand_explicit_candidate(Path(env_candidate)))

        # Conda installs often place the executable and shared libraries outside the normal PATH.
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            prefix_path = Path(conda_prefix)
            discovered_candidates.extend(self._candidate_paths([prefix_path / "bin"], self._local_command_names()))
            discovered_candidates.extend(
                self._candidate_paths(
                    [
                        prefix_path / "Scripts",
                        prefix_path / "Library" / "bin",
                    ],
                    self._local_command_names(),
                )
            )

        # Search PATH next so system installs outrank ad-hoc local builds.
        for command_name in self._path_command_names():
            resolved = shutil.which(command_name)
            if resolved:
                discovered_candidates.append(Path(resolved))

        cwd = Path.cwd()
        project_root = self._project_root()

        # Check the common local build layouts before falling back to the bundled binary.
        discovered_candidates.extend(
            self._candidate_paths(self._common_build_directories(cwd), self._local_command_names())
        )
        bundled_candidates = self._bundled_candidate_paths(project_root)

        searched = explicit_candidates + discovered_candidates + bundled_candidates

        # Respect explicit paths as long as they resolve to a runnable executable.
        for candidate in self._existing_candidates(explicit_candidates):
            if candidate and self._is_executable_file(candidate):
                return candidate.resolve()

        # Prefer GPU-capable builds when discovery finds multiple executables.
        ranked_discovered = self._rank_candidates_by_gpu(discovered_candidates)
        if ranked_discovered:
            return ranked_discovered[0].resolve()

        ranked_bundled = self._rank_candidates_by_gpu(bundled_candidates)
        if ranked_bundled:
            return ranked_bundled[0].resolve()

        searched_text = "\n".join(f"  - {item}" for item in searched if item)
        if platform.system() == "Linux":
            bundled_hint = "or use the bundled fallback at './whisper/whisper-cli'."
        else:
            bundled_hint = "A repo-local bundled fallback is only shipped for Linux x86_64."
        raise SubmasterError(
            "Unable to find a whisper.cpp executable.\n"
            f"Install or build whisper.cpp, put 'whisper-cli' on PATH, or place an executable path in --whisper-cli / WHISPER_CPP_CLI. {bundled_hint}\n"
            "If you are using Conda, install 'conda-forge::whisper.cpp'.\n"
            "Note: the PyPI package named 'whisper-cli' is not the whisper.cpp binary used by this app.\n"
            f"Searched:\n{searched_text}"
        )

    def _existing_candidates(self, candidates: list[Path]) -> list[Path]:
        """Filter a candidate list down to unique executable files.

        :param candidates: Raw candidate paths to inspect.
        :type candidates: list[pathlib.Path]
        :returns: Existing executable candidates.
        :rtype: list[pathlib.Path]
        """
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
        """Rank executable candidates so GPU-capable builds come first.

        :param candidates: Raw candidate paths to rank.
        :type candidates: list[pathlib.Path]
        :returns: Existing candidates ordered by GPU support preference.
        :rtype: list[pathlib.Path]
        """
        existing = self._existing_candidates(candidates)
        if not existing:
            return []

        # Keep CPU-only builds as a fallback when no GPU-enabled binary exists.
        gpu_candidates: list[Path] = []
        cpu_candidates: list[Path] = []
        for candidate in existing:
            if self._detect_gpu_backends_for(candidate):
                gpu_candidates.append(candidate)
            else:
                cpu_candidates.append(candidate)
        return gpu_candidates + cpu_candidates

    def _supports_flag(self, flag: str) -> bool:
        """Check whether the selected executable advertises a CLI flag.

        :param flag: Flag to search for in the help output.
        :type flag: str
        :returns: `True` when the flag is mentioned by the executable.
        :rtype: bool
        """
        # Probe both common help switches because downstream packages do not always support both.
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
        """Detect available GPU backends near a `whisper.cpp` executable.

        :param cli_path: Executable path whose sibling libraries should be inspected.
        :type cli_path: pathlib.Path
        :returns: Detected backend names such as `cuda` or `vulkan`.
        :rtype: set[str]
        """
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

        # Scan nearby library locations because GPU builds often ship backend libraries next to the binary.
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
        """Inspect Linux dynamic linkage for ggml backend libraries.

        :returns: Detected linked backends and the raw ggml-related `ldd` lines.
        :rtype: tuple[set[str], list[str]]
        """
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
        """Validate that GPU mode is backed by GPU-enabled ggml libraries on Linux.

        :param requested: Raw device mode requested by the caller.
        :type requested: str
        :param device: Effective device mode chosen by the runner.
        :type device: str
        :raises SubmasterError: If GPU mode resolves to CPU-only shared libraries.
        """
        # Only Linux shared-library builds need this extra runtime validation.
        if requested != "gpu" or device != "gpu":
            return
        if platform.system() != "Linux":
            return

        linked_backends, linked_lines = self._inspect_linked_ggml_libraries()
        if not linked_lines:
            return
        if linked_backends:
            return

        detail = "\n".join(linked_lines)
        raise SubmasterError(
            "GPU was requested, but the selected whisper.cpp executable is dynamically linked "
            "against CPU-only ggml libraries at runtime.\n"
            "This usually happens when a Conda environment provides libggml-cpu and shadows "
            "your local CUDA-enabled whisper.cpp build.\n"
            "Rebuild whisper.cpp outside Conda with -DGGML_CUDA=1 -DBUILD_SHARED_LIBS=OFF, "
            "or make sure libggml-cuda is the library that ldd resolves.\n"
            f"ldd output:\n{detail}"
        )

    def _extract_timing_lines(self, lines: list[str]) -> list[str]:
        """Extract `whisper.cpp` timing lines from combined process output.

        :param lines: Raw output lines emitted by the executable.
        :type lines: list[str]
        :returns: Timing detail lines stripped of their log prefix.
        :rtype: list[str]
        """
        timings: list[str] = []
        for line in lines:
            match = self.TIMING_RE.search(line)
            if not match:
                continue
            timings.append(match.group("detail").strip())
        return timings

    def _extract_model_load_lines(self, lines: list[str]) -> list[str]:
        """Extract model-load metadata lines from combined process output.

        :param lines: Raw output lines emitted by the executable.
        :type lines: list[str]
        :returns: Model-load detail lines stripped of their log prefix.
        :rtype: list[str]
        """
        details: list[str] = []
        for line in lines:
            match = self.MODEL_LOAD_RE.search(line)
            if not match:
                continue
            details.append(match.group("detail").strip())
        return details

    def _print_model_summary(self, model_load_lines: list[str]) -> None:
        """Print model-load metadata collected from `whisper.cpp`.

        :param model_load_lines: Parsed model-load detail lines.
        :type model_load_lines: list[str]
        """
        if not model_load_lines:
            return
        self.console.info("whisper.cpp model:")
        for detail in model_load_lines:
            self.console.line(f"       {detail}")

    def _print_timing_summary(self, elapsed_seconds: float, timing_lines: list[str]) -> None:
        """Print wall-clock and internal timing information.

        :param elapsed_seconds: Measured wall-clock runtime for the transcription.
        :type elapsed_seconds: float
        :param timing_lines: Parsed timing detail lines from `whisper.cpp`.
        :type timing_lines: list[str]
        """
        self.console.info(f"Transcription wall time: {format_seconds(elapsed_seconds)}")
        if not timing_lines:
            return
        self.console.info("whisper.cpp timings:")
        for timing_line in timing_lines:
            self.console.line(f"       {timing_line}")

    def resolve_device(self, requested: str) -> str:
        """Resolve the effective runtime device for transcription.

        :param requested: Requested device mode: `auto`, `cpu`, or `gpu`.
        :type requested: str
        :returns: Effective device mode after capability checks.
        :rtype: str
        :raises SubmasterError: If the requested mode is invalid.
        """
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
        if "cuda" not in self.gpu_backends:
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
        return "gpu"

    def run(
        self,
        audio_path: Path,
        model_path: Path,
        output_base: Path,
        language: str,
        requested_device: str,
        threads: int,
        show_timings: bool,
        show_model_info: bool,
    ) -> Path:
        """Run `whisper.cpp` and return the generated subtitle file path.

        :param audio_path: Normalized WAV file to transcribe.
        :type audio_path: pathlib.Path
        :param model_path: Whisper model file to load.
        :type model_path: pathlib.Path
        :param output_base: Output basename used by `whisper.cpp`.
        :type output_base: pathlib.Path
        :param language: Language code or `auto` for detection.
        :type language: str
        :param requested_device: Requested device mode: `auto`, `cpu`, or `gpu`.
        :type requested_device: str
        :param threads: Number of CPU threads to use.
        :type threads: int
        :param show_timings: Whether to print extracted timing information.
        :type show_timings: bool
        :param show_model_info: Whether to print extracted model-load metadata.
        :type show_model_info: bool
        :returns: Path to the generated SRT file.
        :rtype: pathlib.Path
        :raises SubmasterError: If execution fails or no SRT file is produced.
        """
        requested = requested_device.lower()
        device = self.resolve_device(requested_device)
        backend_label = ", ".join(sorted(self.gpu_backends)) if self.gpu_backends else "cpu-only build"

        # Build the base command first, then layer language and device-specific flags on top.
        command = [
            str(self.cli_path),
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-of",
            str(output_base),
            "-osrt",
            "-pp",
            "-t",
            str(max(1, threads)),
        ]

        if language.lower() != "auto":
            command.extend(["-l", language])

        # Warn when the requested mode cannot be honored because the discovered build is CPU-only.
        if requested in {"auto", "gpu"} and device == "cpu" and not self.gpu_backends and platform.system() != "Darwin":
            self.console.warn(
                "GPU was requested or auto-selected, but this whisper.cpp executable appears CPU-only. "
                "Build whisper.cpp with CUDA/Vulkan/OpenCL support to use the GPU."
            )
        elif requested == "gpu" and device == "cpu":
            self.console.warn("GPU was requested, but no usable GPU backend was detected. Falling back to CPU.")

        # Force CPU mode only when the selected binary advertises a dedicated no-GPU switch.
        if device == "cpu" and self.supports_no_gpu_flag:
            command.append("-ng")
        elif device == "cpu" and not self.supports_no_gpu_flag:
            self.console.warn("This whisper.cpp build does not advertise '-ng'; CPU forcing may be ignored.")

        if device == "gpu":
            self.console.info(f"GPU mode requested via {self.cli_path.name} ({backend_label}).")
        else:
            self.console.info(f"CPU mode requested via {self.cli_path.name} ({backend_label}).")

        self._verify_gpu_runtime_linkage(requested, device)

        # Stream stdout so progress lines can update the terminal in real time.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress = self.console.progress("whisper", total=100, unit="%", show_value=False)
        output_lines: list[str] = []
        last_percent = 0
        started_at = time.monotonic()

        assert process.stdout is not None

        # Parse progress lines while retaining the full output for error reporting and summaries.
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
            progress.update(last_percent)

        return_code = process.wait()
        total_elapsed = time.monotonic() - started_at
        finish_extra = ""

        # Preserve partial completion information when the process stops early.
        if last_percent > 0 and last_percent < 100:
            finish_extra = f"stopped at {last_percent}% after {format_seconds(total_elapsed)}"
        progress.finish(100 if return_code == 0 else last_percent, extra=finish_extra)

        if return_code != 0:
            detail = "\n".join(output_lines)
            raise SubmasterError(detail or "whisper.cpp transcription failed.")

        # Print optional summaries only after a successful run so the output remains readable.
        if show_model_info:
            self._print_model_summary(self._extract_model_load_lines(output_lines))
        if show_timings:
            self._print_timing_summary(total_elapsed, self._extract_timing_lines(output_lines))

        # `whisper.cpp` writes the SRT next to the requested output basename.
        srt_path = output_base.with_suffix(".srt")
        if not srt_path.exists():
            raise SubmasterError("whisper.cpp finished without creating an SRT file.")

        return srt_path
