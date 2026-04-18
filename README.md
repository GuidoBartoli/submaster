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
- `ffmpeg`
- `ffprobe`
- a working `whisper.cpp` executable (`whisper-cli` recommended)
- a working `llama.cpp` executable (`llama-cli` recommended) if subtitle translation or transcript cleanup is enabled

`submaster` has no third-party Python runtime dependencies. The package metadata in [`pyproject.toml`](pyproject.toml) allows Python `>=3.10`. Python 3.12 is the default tested setup in this repository.

## Installation

### Python Package

Install the package manually:

```bash
python -m pip install -e .
```

On Windows, `py -3 -m pip install -e .` works as well.

Or run it without installation:

```bash
python main.py --help
```

Verify the toolchain:

```bash
python --version
ffmpeg -version
ffprobe -version
whisper-cli -h
llama-cli -h
submaster --help
```

## Translation Backend

Subtitle translation uses **Tencent HY-MT 1.5 GGUF** models through `llama.cpp`.

Available translation sizes:

- `small`: `HY-MT1.5-1.8B-Q4_K_M.gguf`
- `large`: `HY-MT1.5-7B-Q4_K_M.gguf`

The project intentionally downloads the `Q4_K_M` variants instead of full-precision or FP8 checkpoints. For this CLI, the int4 GGUF models are the most suitable choice because they keep downloads, RAM usage, and VRAM usage reasonable while still fitting the local/offline execution goal on consumer hardware.

The models are downloaded automatically into `models/` the first time translation is requested.

## Transcript Cleanup Backend

Plain-text transcription cleanup uses **Qwen3.5-9B Q4_K_M GGUF** through `llama.cpp`.

- Enabled with `--transcribe --cleanup`
- Uses a chunked cleanup pipeline with a `16K` llama.cpp context window
- Keeps subtitle generation unchanged and writes a cleaned companion `.txt` file
- Has no effect when `--transcribe` is not enabled

The cleanup model is downloaded automatically into `models/` the first time transcription polishing is requested.

## How Native Runtimes Are Located

### `whisper.cpp`

Lookup order:

- `--whisper-cli`
- `WHISPER_CPP_CLI`
- Conda, `PATH`, and common local build outputs
- bundled repo fallback on Linux `x86_64` only

Common local build outputs include:

- Linux and macOS: `./whisper.cpp/build/bin/whisper-cli`, `./build/bin/whisper-cli`
- Windows: `.\whisper.cpp\build\bin\Release\whisper-cli.exe`, `.\build\bin\Release\whisper-cli.exe`, plus non-`Release` `.exe` variants

### `llama.cpp`

Lookup order:

- `--llama-cli`
- `LLAMA_CPP_CLI`
- Conda, `PATH`, and common local build outputs

Common local build outputs include:

- Linux and macOS: `./llama.cpp/build/bin/llama-cli`, `./build/bin/llama-cli`
- Windows: `.\llama.cpp\build\bin\Release\llama-cli.exe`, `.\build\bin\Release\llama-cli.exe`, plus non-`Release` `.exe` variants

## Building `whisper.cpp` from source

### CPU builds

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

### GPU builds (CUDA example)

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -G Ninja -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build whisper.cpp/build -j --config Release
```

Before building, confirm that `nvcc` is available:

```bash
nvcc --version
```

## Building `llama.cpp` from source

### CPU builds

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -G Ninja
cmake --build llama.cpp/build -j --config Release
```

Typical outputs are `./llama.cpp/build/bin/llama-cli` on Linux/macOS and `.\llama.cpp\build\bin\Release\llama-cli.exe` on Windows. Add that directory to `PATH` or pass it explicitly:

```bash
submaster input.mp4 --translate-to it --llama-cli ./llama.cpp/build/bin/llama-cli
```

Windows PowerShell equivalent:

```powershell
submaster input.mp4 --translate-to it --llama-cli .\llama.cpp\build\bin\Release\llama-cli.exe
```

Install `ffmpeg`, `git`, `cmake`, and `ninja` first:

- Ubuntu or Debian: `sudo apt install ffmpeg git cmake ninja-build build-essential`
- macOS with Homebrew: `brew install ffmpeg git cmake ninja`
- Windows: install the same tools with `winget`, Chocolatey, Scoop, or another local package manager, then build `whisper.cpp` and `llama.cpp` as shown above

### GPU builds

#### CUDA 

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build -j --config Release
```

#### Vulkan

```bash
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build -j --config Release
```

On macOS, `whisper.cpp` and `llama.cpp` typically use Metal when the selected build includes it. On Windows, `submaster` auto-detects common local `.exe` build outputs and nearby GPU backend DLLs. If you keep multiple builds around, pass `--whisper-cli` and `--llama-cli` explicitly.

## Usage

Basic transcription:

```bash
submaster input.mp4
```

Batch-process all video files in a folder:

```bash
submaster ./videos --batch
```

Translate the generated subtitles into Italian:

```bash
submaster input.mp4 --translate-to it
```

Generate a plain-text transcription and clean it locally:

```bash
submaster input.mp4 --transcribe --cleanup
```

Common patterns:

```bash
submaster input.mp4 --model turbo --device gpu
submaster input.mp4 --language en --model small
submaster input.mp4 --range 00:10:00 00:12:30
submaster input.mp4 --range 600 750 --translate-to it
submaster input.mp4 --vad-model silero-v6.2.0
submaster input.mp4 --vad-model ./models/ggml-silero-v6.2.0.bin
submaster input.mp4 --max-context -1
submaster input.mp4 --translate-to it --translation-model large
submaster input.mp4 --translate-to ja --device gpu --llama-cli ./llama.cpp/build/bin/llama-cli
submaster input.mp4 --transcribe --cleanup
submaster input.mkv --output ./subs/
submaster ./videos --batch --output ./subs/
submaster input.mov --keep-audio
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
submaster input.mp4 --show-timings
submaster input.mp4 --show-model-info
```

On Windows, `--output .\subs\`, `--whisper-cli .\whisper.cpp\build\bin\Release\whisper-cli.exe`, and `--llama-cli .\llama.cpp\build\bin\Release\llama-cli.exe` work as expected.

If `--translate-to` is set, the written `.srt` file keeps the original-language subtitles and an additional `<output-stem>_<language_code>.srt` file is written with the translated subtitles.

`--transcribe` writes a companion plain-text `<video-stem>_transcript.txt` file next to the subtitle output. Add `--cleanup` to also write `<video-stem>_cleanup.txt` with the cleaned-up transcription.

`--range START END` limits extraction, transcription, and optional translation to the selected source clip while keeping subtitle timestamps aligned to the original video timeline. Accepted formats are `SS`, `MM:SS`, or `HH:MM:SS` with optional `.mmm` or `,mmm` milliseconds.

`--batch` treats the positional input as a folder and probes each direct child file with `ffprobe`. Files that do not expose a video stream are skipped. Batch outputs default to `<source-stem>.srt` next to each source file, or into a shared directory when `--output DIR` is provided.

`--max-context` controls how much previously decoded text `whisper.cpp` feeds back into later decode windows. Submaster defaults this to `0` to reduce repetition loops on long recordings. Pass `--max-context -1` to restore the upstream `whisper.cpp` behavior.

`--vad-model VALUE` enables `whisper.cpp` voice activity detection. `VALUE` can be either a local file path or one of the built-in names `silero-v5.1.2` and `silero-v6.2.0`. Named VAD models are downloaded automatically into `models/` from the official `ggml-org/whisper-vad` repository on Hugging Face.

## Validation

Run the test suite:

```bash
python -m unittest
```

Check the CLI wiring:

```bash
python main.py --help
```

## Standalone Chapter Utility

This repository also includes a separate helper script at [`utils/add_chapters.py`](utils/add_chapters.py). It stays outside the main `submaster` CLI so it can be run independently when you only want to embed chapter markers from a text file into an existing video.

Usage:

```bash
python utils/add_chapters.py <input> <chapters> <output>
```

Arguments:

- `input`: Input video file in any format supported by `ffmpeg`
- `chapters`: Text file where each non-empty line contains `HH:MM:SS Title`
- `output`: Output video file to create; `.mp4` is added automatically if you omit a suffix

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

## Upstream References

- OpenAI Whisper: <https://github.com/openai/whisper>
- whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
- whisper.cpp model files: <https://huggingface.co/ggerganov/whisper.cpp>
- whisper.cpp VAD model files: <https://huggingface.co/ggml-org/whisper-vad/tree/main>
- Tencent HY-MT1.5-1.8B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF>
- Tencent HY-MT1.5-7B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-7B-GGUF>
- Qwen3.5-9B GGUF: <https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
