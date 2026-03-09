<p align="center">
  <img src="submaster.png" width="600px" alt="Submaster" />
  <br><b>An open source video transcription utility</b>
</p>

`submaster` is a command-line Python application that transcribes spoken dialogues from a video file into a synchronized `.srt` subtitle file using `ffmpeg` and `whisper.cpp` backend.

## Features

- Accepts common video formats such as MP4, MKV, MOV, AVI, and RMVB
- Extracts and normalizes mono WAV audio automatically before transcription
- Supports `tiny`, `base`, `small`, `medium`, `large`, and `turbo` Whisper models
- Downloads missing GGML model files on demand
- Works on Linux, macOS, and Windows when `ffmpeg` and `whisper.cpp` are available
- Prefers GPU execution in `--device auto` when the selected `whisper.cpp` build appears GPU-capable
- Produces normalized `.srt` output with clean cue numbering and timestamps

## Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| Linux | Supported | Includes a bundled Linux `x86_64` CPU fallback at [`whisper/whisper-cli`](whisper/whisper-cli) |
| macOS | Supported | Requires an installed or locally built `whisper.cpp` executable |
| Windows | Supported | Requires an installed or locally built `whisper.cpp` executable; common `.exe` build outputs are auto-detected |

The bundled fallback is Linux-only. On macOS and Windows, install `whisper.cpp` with Conda or build it locally and point `submaster` at it with `--whisper-cli` if needed.

## Project Layout

```text
main.py                  Program entry point
submaster/               CLI package
whisper/                 Repo-local whisper.cpp fallback metadata and Linux binary
models/                  Auto-downloaded GGML model files
tests/                   Unit tests
```

## Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`
- a working `whisper.cpp` executable (`whisper-cli` recommended)

`submaster` has no third-party Python runtime dependencies. The package metadata in [`pyproject.toml`](pyproject.toml) allows Python `>=3.10`. The provided Conda environment in [`environment.yml`](environment.yml) uses Python 3.12 as the default tested setup.

## Quick Start With Conda

The simplest cross-platform setup is the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate submaster
```

It installs Python 3.12, `ffmpeg`, `ffprobe`, `whisper.cpp`, build tools, and the current project in editable mode.

Verify the toolchain:

```bash
python --version
ffmpeg -version
ffprobe -version
whisper-cli -h
submaster --help
```

If `ffmpeg`, `ffprobe`, or `whisper-cli` is still missing:

```bash
conda install -n submaster -c conda-forge ffmpeg
conda install -n submaster -c conda-forge whisper.cpp
```

## How `whisper.cpp` Is Located

Lookup order:

- `--whisper-cli`
- `WHISPER_CPP_CLI`
- Conda, `PATH`, and common local build outputs
- bundled repo fallback on Linux `x86_64` only

Common local build outputs include:
- Linux and macOS: `./whisper.cpp/build/bin/whisper-cli`, `./build/bin/whisper-cli`
- Windows: `.\whisper.cpp\build\bin\Release\whisper-cli.exe`, `.\build\bin\Release\whisper-cli.exe`, plus non-`Release` `.exe` variants

Do not use:

```bash
pip install whisper-cli
```

That PyPI package is unrelated to the native `whisper.cpp` executable expected by this project.

## Installing Or Building `whisper.cpp`

If your environment was created before `whisper.cpp` was added to [`environment.yml`](environment.yml), install it explicitly:

```bash
conda install -n submaster -c conda-forge whisper.cpp
```

### Build From Source

If you are not using Conda, build `whisper.cpp` locally from the upstream source tree:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -G Ninja
cmake --build whisper.cpp/build -j --config Release
```

Typical outputs are `./whisper.cpp/build/bin/whisper-cli` on Linux/macOS and `.\whisper.cpp\build\bin\Release\whisper-cli.exe` on Windows. Add that directory to `PATH` or pass it explicitly:

```bash
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
```

Windows PowerShell equivalent:

```powershell
submaster input.mp4 --whisper-cli .\whisper.cpp\build\bin\Release\whisper-cli.exe
```

If you choose not to use Conda, install `ffmpeg`, `git`, `cmake`, and `ninja` first:

- Ubuntu or Debian: `sudo apt install ffmpeg git cmake ninja-build build-essential`
- macOS with Homebrew: `brew install ffmpeg git cmake ninja`
- Windows: install the same tools with `winget`, Chocolatey, Scoop, or another local package manager, then build `whisper.cpp` as shown above

## GPU Notes

`--device auto` prefers GPU execution on macOS and when the selected `whisper.cpp` build exposes a GPU backend such as CUDA, Vulkan, OpenCL, Metal, HIP, or SYCL. Actual behavior still depends on the selected binary, drivers, and local hardware.

### NVIDIA CUDA Builds

If you want GPU execution on NVIDIA hardware, install the CUDA toolkit first and build `whisper.cpp` with CUDA enabled:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -G Ninja -DGGML_CUDA=1 -DBUILD_SHARED_LIBS=OFF
cmake --build whisper.cpp/build -j --config Release
```

Before building, confirm that `nvcc` is available:

```bash
nvcc --version
```

If you are using Conda, build the CUDA-enabled binary outside the Conda environment or force a static build with `-DBUILD_SHARED_LIBS=OFF`, otherwise `whisper-cli` may load Conda's CPU-only `ggml` libraries at runtime.

### Linux Runtime Linkage Check

When `--device gpu` is requested on Linux, `submaster` performs an extra runtime check with `ldd`. If the selected `whisper-cli` is dynamically linked against CPU-only `ggml` libraries, `submaster` stops with an explicit error instead of silently running on the CPU.

Useful Linux checks:

```bash
ldd "$(which whisper-cli)" | grep ggml
whisper-cli -m ./models/ggml-base.bin -f sample.wav -pp -np
```

If startup logs contain `whisper_backend_init_gpu: no GPU found`, the current executable is not using your GPU.

On macOS, `whisper.cpp` typically uses Metal when the selected build includes it. On Windows, `submaster` auto-detects common `whisper-cli.exe` build locations and nearby GPU backend DLLs. If you keep multiple builds around, pass `--whisper-cli` explicitly.

## Installing The CLI Without Conda

If you are not using the Conda environment, install the package manually:

```bash
python -m pip install -e .
```

On Windows, `py -3 -m pip install -e .` works as well.

Or run it without installation:

```bash
python main.py --help
```

## Usage

Basic:

```bash
submaster input.mp4
```

Common patterns:

```bash
submaster input.mp4 --model turbo --device gpu
submaster input.mp4 --language en --model small
submaster input.mkv --output ./subs/
submaster input.mov --keep-audio
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
submaster input.mp4 --show-timings
submaster input.mp4 --show-model-info
```

On Windows, `--output .\subs\` and `--whisper-cli .\whisper.cpp\build\bin\Release\whisper-cli.exe` work as expected.

## Command Summary

```text
submaster INPUT
  [-o OUTPUT]
  [-m {tiny,base,small,medium,large,turbo}]
  [-l LANGUAGE]
  [--device {auto,cpu,gpu}]
  [--models-dir MODELS_DIR]
  [--whisper-cli PATH]
  [--threads N]
  [--overwrite]
  [--keep-audio]
  [--show-timings]
  [--show-model-info]
```

## Notes

- `large` maps to `ggml-large-v3.bin`; `turbo` maps to `ggml-large-v3-turbo.bin`.
- Audio-only inputs are intentionally rejected. Pass a video file and let `submaster` extract the audio track.
- `--show-timings` prints the final `whisper_print_timings` block, and `--show-model-info` prints the `whisper_model_load` block.
- Model files are intentionally ignored by git through `.gitignore`.

## Validation

Run the test suite:

```bash
python -m unittest
```

Check the CLI wiring:

```bash
python main.py --help
```

## Upstream References

- OpenAI Whisper: <https://github.com/openai/whisper>
- whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
- whisper.cpp model files: <https://huggingface.co/ggerganov/whisper.cpp>
