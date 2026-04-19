#!/usr/bin/env python3
"""Clone and build whisper.cpp and llama.cpp with GPU support auto-detected from the current hardware."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


_WHISPER_URL = "https://github.com/ggml-org/whisper.cpp.git"
_LLAMA_URL = "https://github.com/ggml-org/llama.cpp.git"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the runtime setup utility."""
    parser = argparse.ArgumentParser(
        prog="setup_runtimes.py",
        description=(
            "Clone whisper.cpp and llama.cpp, detect your GPU, and build both "
            "executables so submaster can find them automatically."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "cuda", "vulkan", "metal", "cpu"),
        help="GPU backend to build with. 'auto' probes the current hardware.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip 'git pull' when the repository is already cloned.",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip cmake configure and build when the build directory already exists.",
    )
    return parser


def _compiler_available() -> bool:
    """Return True if at least one C/C++ compiler is reachable on PATH."""
    candidates = ("gcc", "g++", "clang", "clang++", "cc", "c++", "cl")
    return any(shutil.which(c) is not None for c in candidates)


def _vulkan_sdk_available() -> bool:
    """Return True if Vulkan development headers appear to be installed."""
    if os.environ.get("VULKAN_SDK"):
        return True
    # pkg-config is the canonical way to check on Linux
    if shutil.which("pkg-config"):
        result = subprocess.run(
            ["pkg-config", "--exists", "vulkan"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return True
    # Fall back to checking for the system header directly
    header_paths = (
        Path("/usr/include/vulkan/vulkan.h"),
        Path("/usr/local/include/vulkan/vulkan.h"),
    )
    return any(p.exists() for p in header_paths)


def check_prerequisites(backend: str) -> str:
    """Verify required build tools are present and return the cmake generator to use.

    Performs hard checks (abort on failure) for ``git`` and ``cmake``, soft checks
    (warn only) for the C/C++ compiler and backend-specific libraries.

    :param backend: GPU backend name; used to trigger backend-specific checks.
    :returns: cmake generator string, e.g. ``"Ninja"`` or ``"Unix Makefiles"``.
    :raises SystemExit: If ``git`` or ``cmake`` are missing.
    """
    system = platform.system()

    def _install_hint(pkg: str) -> str:
        if system == "Darwin":
            return f"  macOS:   brew install {pkg}"
        if system == "Windows":
            return f"  Windows: winget install {pkg}  or  choco install {pkg}"
        return f"  Linux:   sudo apt install {pkg}  or  sudo dnf install {pkg}"

    # --- Hard requirements: git and cmake ---
    missing = [t for t in ("git", "cmake") if shutil.which(t) is None]
    if missing:
        for tool in missing:
            print(f"! '{tool}' not found.")
            print(_install_hint(tool))
        raise SystemExit("! Missing required tools. Aborting.")

    # --- C/C++ compiler (soft check — cmake will diagnose this more precisely) ---
    if not _compiler_available():
        print("! No C/C++ compiler found on PATH.")
        if system == "Darwin":
            print("  macOS:   xcode-select --install")
        elif system == "Windows":
            print("  Windows: install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/")
        else:
            print("  Linux:   sudo apt install build-essential  or  sudo dnf install gcc-c++")

    # --- cmake generator: prefer Ninja, fall back to platform make ---
    if shutil.which("ninja") or shutil.which("ninja-build"):
        generator = "Ninja"
    else:
        print("! ninja not found; falling back to make. Install ninja-build for faster builds.")
        if system == "Windows":
            generator = "NMake Makefiles"
        else:
            generator = "Unix Makefiles"

    # --- Backend-specific checks (soft) ---
    if backend == "cuda":
        if shutil.which("nvcc") is None:
            print("! 'nvcc' not found. Install the CUDA toolkit to enable CUDA builds.")
            if system == "Windows":
                print("  Windows: https://developer.nvidia.com/cuda-downloads")
            else:
                print("  Linux:   sudo apt install nvidia-cuda-toolkit  or  https://developer.nvidia.com/cuda-downloads")

    if backend == "vulkan":
        if not _vulkan_sdk_available():
            print("! Vulkan development headers not found.")
            if system == "Windows":
                print("  Windows: install the LunarG Vulkan SDK from https://vulkan.lunarg.com/")
            else:
                print("  Linux:   sudo apt install libvulkan-dev vulkan-tools")

    return generator


def detect_gpu_backend() -> str:
    """Probe the current hardware and return the best GPU backend name.

    :returns: One of ``"cuda"``, ``"vulkan"``, ``"metal"``, or ``"cpu"``.
    """
    system = platform.system()

    # Metal is the only GPU backend on macOS — always available
    if system == "Darwin":
        print("> GPU backend detected: metal")
        return "metal"

    # CUDA: nvidia-smi -L lists GPUs; non-empty output means at least one is present
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                print("> GPU backend detected: cuda")
                return "cuda"
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Vulkan: vulkaninfo --summary works on Linux and Windows
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        try:
            result = subprocess.run(
                [vulkaninfo, "--summary"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                print("> GPU backend detected: vulkan")
                return "vulkan"
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Windows secondary Vulkan signal: VULKAN_SDK environment variable
    if system == "Windows" and os.environ.get("VULKAN_SDK"):
        print("> GPU backend detected: vulkan (via VULKAN_SDK env var)")
        return "vulkan"

    print("> GPU backend detected: cpu (no supported GPU found)")
    return "cpu"


def clone_or_update(name: str, url: str, dest: Path, *, no_update: bool) -> None:
    """Clone the repository into ``dest``, or pull the latest changes if it already exists.

    :param name: Human-readable project name used in log output.
    :param url: Git remote URL.
    :param dest: Local destination path.
    :param no_update: When ``True``, skip ``git pull`` for existing repos.
    :raises SystemExit: If the git command fails.
    """
    try:
        if not dest.exists():
            print(f"- Cloning {name}...")
            # --depth 1 fetches only the latest commit — sufficient for building
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
            )
        elif no_update:
            print(f"- {name}: repo exists, skipping update (--no-update).")
        else:
            print(f"- {name}: repo exists, pulling latest changes...")
            subprocess.run(["git", "-C", str(dest), "pull"], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"! git command failed with exit code {exc.returncode}.") from exc


def cmake_configure(source: Path, build: Path, backend: str, generator: str) -> None:
    """Run cmake configure for the given source tree.

    :param source: Root of the cloned repository.
    :param build: Build output directory (created by cmake if absent).
    :param backend: GPU backend name — controls which ``-DGGML_*`` flag is added.
    :param generator: cmake generator string, e.g. ``"Ninja"``.
    :raises SystemExit: If cmake exits with a non-zero status.
    """
    cmd = ["cmake", "-S", str(source), "-B", str(build), "-G", generator]
    if backend == "cuda":
        cmd.append("-DGGML_CUDA=ON")
    elif backend == "vulkan":
        cmd.append("-DGGML_VULKAN=ON")
    # metal and cpu need no extra flags — ggml detects Metal on Darwin automatically

    # Note: -DBUILD_SHARED_LIBS=OFF is intentionally omitted. Shared GPU libraries
    # (e.g. libggml-cuda.so) must be built alongside the binary so the submaster
    # runtime auto-detection can locate them via rglob().
    print(f"- cmake configure: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"! cmake configure failed with exit code {exc.returncode}.") from exc


def cmake_build(build: Path) -> None:
    """Run cmake build in the given build directory.

    :param build: Build directory previously configured by cmake.
    :raises SystemExit: If cmake exits with a non-zero status.
    """
    cmd = ["cmake", "--build", str(build), "--config", "Release", "-j"]
    print(f"- cmake build: {' '.join(cmd)}")
    try:
        # No capture_output — build takes minutes and streaming output keeps the user informed
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"! cmake build failed with exit code {exc.returncode}.") from exc


def verify_executable(build_bin: Path, name: str, exe_name: str) -> Path:
    """Confirm the expected executable exists after a successful build.

    Checks both the flat Ninja layout (``bin/exe``) and the MSVC layout
    (``bin/Release/exe``) with and without ``.exe`` suffix.

    :param build_bin: The ``bin/`` directory inside the build tree.
    :param name: Human-readable project name for error messages.
    :param exe_name: Base executable name without extension.
    :returns: Path to the found executable.
    :raises SystemExit: If no candidate is found.
    """
    candidates = [
        build_bin / exe_name,
        build_bin / "Release" / exe_name,
        build_bin / (exe_name + ".exe"),
        build_bin / "Release" / (exe_name + ".exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            print(f"> {name} executable: {candidate}")
            return candidate
    raise SystemExit(
        f"! {name} build appeared to succeed but '{exe_name}' was not found in {build_bin}."
    )


def setup_project(
    name: str,
    url: str,
    root: Path,
    backend: str,
    generator: str,
    *,
    no_update: bool,
    no_rebuild: bool,
) -> None:
    """Run the full clone → configure → build → verify pipeline for one project.

    :param name: Repository subdirectory name, e.g. ``"whisper.cpp"``.
    :param url: Git remote URL.
    :param root: Project root from which repos are cloned.
    :param backend: GPU backend, e.g. ``"cuda"``.
    :param generator: cmake generator string.
    :param no_update: Skip ``git pull`` for existing repos.
    :param no_rebuild: Skip cmake steps when build directory already exists.
    :raises SystemExit: If any step fails.
    """
    source = root / name
    build = source / "build"
    bin_dir = build / "bin"
    exe_name = "whisper-cli" if name == "whisper.cpp" else "llama-cli"

    clone_or_update(name, url, source, no_update=no_update)

    if no_rebuild and build.exists():
        print(f"- {name}: build dir exists, skipping rebuild (--no-rebuild).")
    else:
        cmake_configure(source, build, backend, generator)
        cmake_build(build)

    verify_executable(bin_dir, name, exe_name)


def main(argv: list[str] | None = None) -> int:
    """Run the full runtime setup workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    print("> SubMaster runtime setup")

    backend = args.backend
    if backend == "auto":
        backend = detect_gpu_backend()
    else:
        print(f"> GPU backend: {backend} (user override)")

    generator = check_prerequisites(backend)

    root = Path.cwd()
    print(f"- Working directory: {root}")

    # Warn if the script is not being run from the submaster project root
    if not (root / "pyproject.toml").exists():
        print("! Warning: pyproject.toml not found in the current directory.")
        print("!          Run this script from the submaster project root.")

    errors = 0

    print("\n> Setting up whisper.cpp")
    try:
        setup_project(
            "whisper.cpp",
            _WHISPER_URL,
            root,
            backend,
            generator,
            no_update=args.no_update,
            no_rebuild=args.no_rebuild,
        )
    except SystemExit as exc:
        print(exc)
        errors += 1

    print("\n> Setting up llama.cpp")
    try:
        setup_project(
            "llama.cpp",
            _LLAMA_URL,
            root,
            backend,
            generator,
            no_update=args.no_update,
            no_rebuild=args.no_rebuild,
        )
    except SystemExit as exc:
        print(exc)
        errors += 1

    print()
    if errors == 0:
        print("> Setup complete. Run 'submaster --help' to get started.")
    else:
        print(f"! Setup finished with {errors} error(s). Check the output above.")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
