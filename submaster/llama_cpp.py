from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import json
from contextlib import nullcontext
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

_PROMPT_CUE_RE = re.compile(
    r"\[\[\[cue:(?P<id>\d+)\]\]\]\s*(?P<text>.*?)\s*\[\[\[/cue\]\]\]",
    re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


class LlamaCppRunner:
    """Discover and invoke a local `llama.cpp` command-line executable."""

    def __init__(self, console: Console, cli_path: Path | None = None) -> None:
        """Initialize the runner and inspect the selected executable.

        :param console: Console used for status, warnings, and spinner output.
        :type console: Console
        :param cli_path: Optional explicit path to a `llama.cpp` executable.
        :type cli_path: pathlib.Path | None
        :raises SubmasterError: If no usable executable can be found.
        """
        self.console = console
        self.cli_path = self._resolve_cli_path(cli_path)
        # Cache supported flags and GPU backends once so translation calls can stay fast.
        self.supports_ngl_flag = self._supports_any_flag("-ngl", "--n-gpu-layers")
        self.supports_simple_io = self._supports_any_flag("--simple-io")
        self.supports_no_display_prompt = self._supports_any_flag("--no-display-prompt")
        self.supports_conversation = self._supports_any_flag("-cnv", "--conversation")
        self.supports_no_conversation = self._supports_any_flag("-no-cnv", "--no-conversation")
        self.supports_no_warmup = self._supports_any_flag("--no-warmup")
        self.supports_single_turn = self._supports_any_flag("--single-turn")
        self.supports_chat_template_kwargs = self._supports_any_flag("--chat-template-kwargs")
        self.supports_reasoning_format = self._supports_any_flag("--reasoning-format")
        self.gpu_backends = self._detect_gpu_backends_for(self.cli_path)
        self._announced_modes: set[str] = set()

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
        return tuple(ext.lower() for ext in raw_value.split(";") if ext.strip())

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
        return ("llama-completion", "llama-cli", "main")

    def _completion_sibling_for(self, candidate: Path) -> Path | None:
        """Return a sibling `llama-completion` executable when present.

        :param candidate: Candidate executable path selected for translation.
        :type candidate: pathlib.Path
        :returns: Sibling completion executable, or `None` when unavailable.
        :rtype: pathlib.Path | None
        """
        normalized_name = candidate.name.lower()
        if "llama-completion" in normalized_name:
            return candidate
        if normalized_name not in {"llama-cli", "llama-cli.exe", "main", "main.exe"}:
            return None

        sibling_name = "llama-completion.exe" if candidate.suffix.lower() == ".exe" else "llama-completion"
        sibling = candidate.with_name(sibling_name)
        if self._is_executable_file(sibling):
            return sibling
        return None

    def _prefer_completion_binary(self, candidate: Path) -> Path:
        """Prefer the dedicated completion binary for prompt-based translation.

        :param candidate: Candidate executable path.
        :type candidate: pathlib.Path
        :returns: Completion-capable executable path.
        :rtype: pathlib.Path
        """
        completion_sibling = self._completion_sibling_for(candidate)
        return completion_sibling or candidate

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
        """List common `llama.cpp` build output directories under a root.

        :param root: Root directory to inspect.
        :type root: pathlib.Path
        :returns: Deduplicated build directories to search.
        :rtype: list[pathlib.Path]
        """
        # Cover both in-repo builds and standard CMake configuration subdirectories.
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
        """Build full candidate paths from directories and executable names.

        :param directories: Directories to search.
        :type directories: list[pathlib.Path]
        :param names: Executable basenames to combine with each directory.
        :type names: tuple[str, ...]
        :returns: Candidate executable paths.
        :rtype: list[pathlib.Path]
        """
        return [directory / name for directory in directories for name in names]

    def _resolve_cli_path(self, cli_path: Path | None) -> Path:
        """Resolve the best available `llama.cpp` executable path.

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

        env_candidate = os.environ.get("LLAMA_CPP_CLI")
        if env_candidate:
            explicit_candidates.extend(self._expand_explicit_candidate(Path(env_candidate)))

        # Conda installs often place the executable and shared libraries outside the normal PATH.
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

        # Search PATH next so system installs outrank ad-hoc local builds.
        for command_name in self._path_command_names():
            resolved = shutil.which(command_name)
            if resolved:
                discovered_candidates.append(Path(resolved))

        # Finally search common local build layouts in the current working tree and repo root.
        cwd = Path.cwd()
        project_root = self._project_root()
        discovered_candidates.extend(
            self._candidate_paths(self._common_build_directories(cwd), self._local_command_names())
        )
        discovered_candidates.extend(
            self._candidate_paths(self._common_build_directories(project_root), self._local_command_names())
        )

        searched = explicit_candidates + discovered_candidates

        # Respect explicit paths as long as they resolve to a runnable executable.
        for candidate in self._existing_candidates(explicit_candidates):
            if candidate and self._is_executable_file(candidate):
                return self._prefer_completion_binary(candidate).resolve()

        # Prefer GPU-capable builds when discovery finds multiple executables.
        ranked_discovered = self._rank_candidates_by_gpu(discovered_candidates)
        if ranked_discovered:
            return self._prefer_completion_binary(ranked_discovered[0]).resolve()

        searched_text = "\n".join(f"  - {item}" for item in searched if item)
        raise SubmasterError(
            "Unable to find a llama.cpp executable.\n"
            "Install or build llama.cpp, put 'llama-cli' on PATH, or place an executable path "
            "in --llama-cli / LLAMA_CPP_CLI.\n"
            "If you are using Conda, install 'conda-forge::llama.cpp'.\n"
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

    def _supports_any_flag(self, *flags: str) -> bool:
        """Check whether the selected executable advertises any given CLI flag.

        :param flags: One or more flags to search for in the help output.
        :type flags: str
        :returns: `True` when at least one flag is mentioned by the executable.
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
            if any(flag in help_text for flag in flags):
                return True
        return False

    def _detect_gpu_backends_for(self, cli_path: Path) -> set[str]:
        """Detect available GPU backends near a `llama.cpp` executable.

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
        """Resolve the effective runtime device for translation.

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
        return "gpu"

    def _estimated_max_tokens(self, prompt: str) -> int:
        """Estimate a completion token budget from prompt size.

        :param prompt: Prompt text that will be sent to `llama.cpp`.
        :type prompt: str
        :returns: Conservative maximum token count for the response.
        :rtype: int
        """
        matches = list(_PROMPT_CUE_RE.finditer(prompt))
        if matches:
            source_chars = sum(len(match.group("text").strip()) for match in matches)
            # Translation output often matches source length and must repeat cue markers.
            token_budget = source_chars + (len(matches) * 32)
            return max(256, min(1_536, token_budget))

        prompt_chars = max(1, len(prompt))
        return max(256, min(768, int(prompt_chars / 2)))

    def _normalize_output(self, output: str) -> str:
        """Normalize raw model output for downstream cue parsing.

        :param output: Raw stdout emitted by `llama.cpp`.
        :type output: str
        :returns: Cleaned translation text with wrapper fences removed.
        :rtype: str
        """
        normalized = output.strip()
        normalized = normalized.replace("[end of text]", "").strip()
        if normalized.startswith("```"):
            # Some models wrap responses in fenced blocks; remove the fence lines only.
            parts = normalized.split("```")
            normalized = "\n".join(part for part in parts if part.strip() and not part.strip().isidentifier())
        return normalized.strip()

    def _strip_reasoning_output(self, output: str) -> str:
        """Remove common Qwen-style reasoning wrappers from model output.

        :param output: Normalized model output.
        :type output: str
        :returns: Output with leading `<think>` sections removed when present.
        :rtype: str
        """
        normalized = output.strip()
        closing_tag = re.search(r"</think>", normalized, flags=re.IGNORECASE)
        if closing_tag is not None:
            normalized = normalized[closing_tag.end():].strip()
        normalized = _THINK_TAG_RE.sub("", normalized).strip()
        return normalized

    def _announce_mode_once(self, device: str) -> None:
        """Emit the selected runtime mode at most once per device.

        :param device: Effective device mode being used.
        :type device: str
        """
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
        system_prompt: str | None = None,
        show_spinner: bool = True,
        context_size: int = LLAMA_CONTEXT_SIZE,
        temperature: float = LLAMA_TEMPERATURE,
        top_k: int | None = LLAMA_TOP_K,
        top_p: float = LLAMA_TOP_P,
        repeat_penalty: float = LLAMA_REPEAT_PENALTY,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        spinner_label: str = "Running subtitle translation batch.",
    ) -> str:
        """Run a prompt through `llama.cpp`.

        :param model_path: Model file to load.
        :type model_path: pathlib.Path
        :param prompt: Prompt text to send to the model.
        :type prompt: str
        :param requested_device: Requested device mode: `auto`, `cpu`, or `gpu`.
        :type requested_device: str
        :param threads: Number of CPU threads to use.
        :type threads: int
        :returns: Normalized text output from the model.
        :rtype: str
        :raises SubmasterError: If execution fails or produces no usable output.
        """
        requested = requested_device.lower()
        device = self.resolve_device(requested_device)

        # Warn when the requested mode cannot be honored because the discovered build is CPU-only.
        if requested in {"auto", "gpu"} and device == "cpu" and not self.gpu_backends and platform.system() != "Darwin":
            self.console.warn(
                "GPU was requested or auto-selected, but this llama.cpp executable appears CPU-only. "
                "Build llama.cpp with CUDA, Vulkan, Metal, HIP, OpenCL, or SYCL support to use the GPU."
            )
        elif requested == "gpu" and device == "cpu":
            self.console.warn("GPU was requested, but no usable GPU backend was detected. Falling back to CPU.")

        # Build a deterministic command line so batch translation output stays machine-parseable.
        command = [
            str(self.cli_path),
            "-m",
            str(model_path),
            "-n",
            str(max_tokens if max_tokens is not None else self._estimated_max_tokens(prompt)),
            "-c",
            str(max(512, context_size)),
            "-t",
            str(max(1, threads)),
            "--temp",
            str(temperature),
            "--top-p",
            str(top_p),
            "--repeat-penalty",
            str(repeat_penalty),
        ]
        if top_k is not None:
            command.extend(["--top-k", str(top_k)])

        use_chat_turn = bool(system_prompt and self.supports_conversation and self.supports_single_turn)
        if use_chat_turn:
            command.extend(
                [
                    "--conversation",
                    "--single-turn",
                    "--system-prompt",
                    system_prompt,
                    "-p",
                    prompt,
                ]
            )
            if disable_thinking and self.supports_chat_template_kwargs:
                command.extend(
                    [
                        "--chat-template-kwargs",
                        json.dumps({"enable_thinking": False}, separators=(",", ":")),
                    ]
                )
        else:
            command.extend(["-p", prompt])

        # Enable optional quality-of-life flags only when the selected binary advertises them.
        if self.supports_simple_io:
            command.append("--simple-io")
        if self.supports_no_display_prompt:
            command.append("--no-display-prompt")
        if not use_chat_turn and self.supports_no_conversation:
            command.append("--no-conversation")
        if self.supports_no_warmup:
            command.append("--no-warmup")
        if not use_chat_turn and not self.supports_no_conversation and self.supports_single_turn:
            command.append("--single-turn")
        if disable_thinking and not use_chat_turn and self.supports_reasoning_format:
            command.extend(["--reasoning-format", "none"])

        if self.supports_ngl_flag:
            gpu_layers = LLAMA_N_GPU_LAYERS_ALL if device == "gpu" else LLAMA_N_GPU_LAYERS_CPU
            command.extend(["-ngl", str(gpu_layers)])
        elif device == "cpu":
            self.console.warn("This llama.cpp build does not advertise '-ngl'; CPU forcing may be ignored.")

        self._announce_mode_once(device)
        self._verify_gpu_runtime_linkage(requested, device)

        spinner_context = self.console.spinner(spinner_label) if show_spinner else nullcontext()
        with spinner_context:
            with tempfile.SpooledTemporaryFile(max_size=1_048_576) as stdout_file:
                with tempfile.SpooledTemporaryFile(max_size=1_048_576) as stderr_file:
                    result = subprocess.run(
                        command,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        stdin=subprocess.DEVNULL,
                        check=False,
                    )
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    stdout_text = stdout_file.read().decode("utf-8", errors="replace")
                    stderr_text = stderr_file.read().decode("utf-8", errors="replace")

        # Surface combined stdout and stderr when the subprocess fails.
        if result.returncode != 0:
            detail = "\n".join(
                part.strip()
                for part in (stdout_text, stderr_text)
                if part and part.strip()
            )
            raise SubmasterError(detail or "llama.cpp translation failed.")

        # Normalize the model output before cue parsing so fenced wrappers do not break the translator.
        normalized_output = self._normalize_output(stdout_text)
        if disable_thinking:
            normalized_output = self._strip_reasoning_output(normalized_output)
        if not normalized_output:
            raise SubmasterError("llama.cpp finished without producing translated text.")
        return normalized_output
