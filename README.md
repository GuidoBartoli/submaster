<p align="center">
  <img src="submaster.png" width="600px" alt="Submaster" />
  <br><b>An open source tool for video subtitle generation and translation</b>
</p>

**SubMaster** is a command-line Python application that takes a video as input and transcribes spoken dialogues into a synchronized subtitle file with optional translation into another language.

<p align="center">
<img width="721" height="166" alt="image" src="https://github.com/user-attachments/assets/f3e6be8f-f59b-4ac2-a940-d78cd33d9040" />
</p>

## Features

- Accepts common video formats such as MP4, MKV, MOV, AVI, and RMVB
- Batch-processes every direct child video file in a folder without relying on a fixed extension allowlist
- Extracts and normalizes mono WAV audio using `ffmpeg` before transcription
- Supports `tiny`, `base`, `small`, `medium`, `large`, and `turbo` Whisper models
- Optionally translates subtitles into another language with **Tencent HY-MT 1.5** models through `llama.cpp`
- Optionally polishes `--transcribe` output with a local **Qwen3.5-9B** cleanup pass through `llama.cpp`
- Optionally embeds chapter markers from a plain-text file into a copy of the input video with `--chapters`
- Downloads missing Whisper, HY-MT, and transcript-cleanup model files on demand into the local `models/` cache
- Works on Linux, macOS, and Windows when `ffmpeg`, `whisper.cpp`, and `llama.cpp` are available
- Prefers GPU execution in `--device auto` when the selected native runtime appears GPU-capable
- Disables rolling Whisper text context by default to reduce long-form repetition loops
- Produces normalized `.srt` output with clean cue numbering and timestamps

## Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| Linux | Supported | Includes a bundled Linux `x86_64` `whisper-cli` CPU fallback at [`whisper/whisper-cli`](whisper/whisper-cli) |
| macOS | Supported | Requires installed or locally built `whisper.cpp` and `llama.cpp` executables |
| Windows | Supported | Requires installed or locally built `whisper.cpp` and `llama.cpp` executables; common `.exe` build outputs are auto-detected |

The bundled fallback is only for `whisper.cpp` on Linux `x86_64`. Translation and transcript cleanup always require a working `llama-cli`.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe`
- `whisper-cli` and `llama-cli` executables — provided by `setup_runtimes.py` (see Quick Setup below)

`submaster` has no third-party Python runtime dependencies. The package metadata in [`pyproject.toml`](pyproject.toml) allows Python `>=3.10`. Python 3.12 is the default tested setup in this repository.

## Quick Setup

Run the included setup script from the project root to clone, build, and install both native runtimes automatically:

```bash
python -m pip install -e .
python setup_runtimes.py
```

The editable install provides the `submaster` command for your current Python environment. The runtime setup script:

1. Detects your GPU (CUDA, Vulkan, Metal, or CPU fallback)
2. Checks for required build tools (`git`, `cmake`, C/C++ compiler, `ninja`) and prints per-platform install hints for anything missing
3. Clones `whisper.cpp` and `llama.cpp` into the project root and builds them with the correct flags
4. Verifies the resulting executables — `submaster` picks them up automatically from those locations

Install build prerequisites first:

- Ubuntu or Debian: `sudo apt install git cmake ninja-build build-essential`
- macOS with Homebrew: `brew install git cmake ninja`
- Windows: install [Git](https://git-scm.com/), [CMake](https://cmake.org/), and [Ninja](https://ninja-build.org/) via `winget`, Chocolatey, or Scoop

For CUDA builds, the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is also required. For Vulkan builds, install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) (Windows) or `libvulkan-dev` (Linux).

Common options:

```bash
python setup_runtimes.py --backend cpu      # force a CPU-only build
python setup_runtimes.py --backend cuda     # force CUDA regardless of detection
python setup_runtimes.py --no-update        # offline: skip git pull
python setup_runtimes.py --no-rebuild       # skip build, re-verify executables only
```

On Windows, Ninja must be installed (via `winget install Ninja-build.Ninja`, Chocolatey, or Scoop) so cmake can run from any terminal. Without Ninja, cmake falls back to NMake Makefiles, which requires a Visual Studio Developer Command Prompt.

## Usage

- Basic transcription: `submaster input.mp4`
- Batch-process all video files in a folder: `submaster ./videos`
- Translate the generated subtitles into Italian: `submaster input.mp4 --translate it`
- Generate a plain-text transcription and clean it locally: `submaster input.mp4 --transcribe --cleanup`

### Options

If `--translate` is set, the written `.srt` file keeps the original-language subtitles and an additional `<output-stem>_<language_code>.srt` file is written with the translated subtitles.

`--transcribe` writes a companion plain-text `<video-stem>_transcript.txt` file next to the subtitle output. Add `--cleanup` to also write `<video-stem>_cleanup.txt` with the cleaned-up transcription.

`--chapters FILE` reads chapter timestamps from a plain-text file and produces a chapter-embedded copy of the input video named `<input-stem>_chapters.<ext>` next to the original. Each non-empty line must follow the format `HH:MM:SS Title`. Cannot be combined with folder input.

Example chapter file:

```text
00:00:00 Introduction
00:23:20 Start
00:40:30 First Performance
00:40:56 Break
01:04:44 Second Performance
01:24:45 Crowd Shots
01:27:45 Credits
```

`--range START END` limits extraction, transcription, and optional translation to the selected source clip while keeping subtitle timestamps aligned to the original video timeline. Accepted formats are `SS`, `MM:SS`, or `HH:MM:SS` with optional `.mmm` or `,mmm` milliseconds.

When the positional input is a folder, SubMaster probes each direct child file with `ffprobe`. Files that do not expose a video stream are skipped. Batch outputs default to `<source-stem>.srt` next to each source file, or into a shared directory when `--output DIR` is provided.

`--max-context` controls how much previously decoded text `whisper.cpp` feeds back into later decode windows. Submaster defaults this to `0` to reduce repetition loops on long recordings. Pass `--max-context -1` to restore the upstream `whisper.cpp` behavior.

## Translation and Transcription

### Translation Backend

Subtitle translation uses **Tencent HY-MT 1.5 GGUF** models through `llama.cpp`.

Available translation sizes:

- `small`: `HY-MT1.5-1.8B-Q4_K_M.gguf`
- `large`: `HY-MT1.5-7B-Q4_K_M.gguf`

The project intentionally downloads the `Q4_K_M` variants instead of full-precision or FP8 checkpoints. For this CLI, the int4 GGUF models are the most suitable choice because they keep downloads, RAM usage, and VRAM usage reasonable while still fitting the local/offline execution goal on consumer hardware.

The models are downloaded automatically into `models/` the first time translation is requested.

### Transcript Cleanup Backend

Plain-text transcription cleanup uses **Qwen3.5-9B Q4_K_M GGUF** through `llama.cpp`.

- Enabled with `--transcribe --cleanup`
- Uses a chunked cleanup pipeline with a `16K` llama.cpp context window
- Keeps subtitle generation unchanged and writes a cleaned companion `.txt` file
- Has no effect when `--transcribe` is not enabled

The cleanup model is downloaded automatically into `models/` the first time transcription polishing is requested.

## How Native Runtimes Are Located

### `whisper.cpp`

Lookup order:

- `WHISPER_CPP_CLI` environment variable
- Conda, `PATH`, and common local build outputs
- bundled repo fallback on Linux `x86_64` only

Common local build outputs (produced by `setup_runtimes.py`):

- Linux and macOS: `./whisper.cpp/build/bin/whisper-cli`
- Windows: `.\whisper.cpp\build\bin\Release\whisper-cli.exe`

### `llama.cpp`

Lookup order:

- `LLAMA_CPP_CLI` environment variable
- Conda, `PATH`, and common local build outputs

Common local build outputs (produced by `setup_runtimes.py`):

- Linux and macOS: `./llama.cpp/build/bin/llama-cli`
- Windows: `.\llama.cpp\build\bin\Release\llama-cli.exe`

## Validation

Run the test suite:

```bash
python -m unittest
```

## Upstream References

- OpenAI Whisper: <https://github.com/openai/whisper>
- whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
- whisper.cpp model files: <https://huggingface.co/ggerganov/whisper.cpp>
- whisper.cpp VAD model files: <https://huggingface.co/ggml-org/whisper-vad/tree/main>
- Tencent HY-MT1.5-1.8B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF>
- Tencent HY-MT1.5-7B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-7B-GGUF>
- Qwen3.5-9B GGUF: <https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
