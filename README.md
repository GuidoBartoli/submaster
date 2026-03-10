<p align="center">
  <img src="submaster.png" width="600px" alt="Submaster" />
  <br><b>An open source tool for video subtitle generation and translation</b>
</p>

**SubMaster** is a command-line Python application that takes a video as input and transcribes spoken dialogues into a synchronized subtitle file with optional translation into another language.

## Features

- Accepts common video formats such as MP4, MKV, MOV, AVI, and RMVB
- Extracts and normalizes mono WAV audio using `ffmpeg` before transcription
- Supports `tiny`, `base`, `small`, `medium`, `large`, and `turbo` Whisper models
- Optionally translates subtitles into another language with **Tencent HY-MT 1.5** models through `llama.cpp`
- Downloads missing Whisper and HY-MT model files on demand into the local `models/` cache
- Works on Linux, macOS, and Windows when `ffmpeg`, `whisper.cpp`, and `llama.cpp` are available
- Prefers GPU execution in `--device auto` when the selected native runtime appears GPU-capable
- Produces normalized `.srt` output with clean cue numbering and timestamps

## Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| Linux | Supported | Includes a bundled Linux `x86_64` `whisper-cli` CPU fallback at [`whisper/whisper-cli`](whisper/whisper-cli) |
| macOS | Supported | Requires installed or locally built `whisper.cpp` and `llama.cpp` executables |
| Windows | Supported | Requires installed or locally built `whisper.cpp` and `llama.cpp` executables; common `.exe` build outputs are auto-detected |

The bundled fallback is only for `whisper.cpp` on Linux `x86_64`. Translation always requires a working `llama-cli`.

## Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`
- a working `whisper.cpp` executable (`whisper-cli` recommended)
- a working `llama.cpp` executable (`llama-cli` recommended) if subtitle translation is enabled

`submaster` has no third-party Python runtime dependencies. The package metadata in [`pyproject.toml`](pyproject.toml) allows Python `>=3.10`. The provided Conda environment in [`environment.yml`](environment.yml) uses Python 3.12 as the default tested setup.

## Installation

### Conda Environment (Recommended)

The simplest cross-platform setup is the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate submaster
```

It installs Python 3.12, `ffmpeg`, `ffprobe`, `whisper.cpp`, `llama.cpp`, build tools, and the current project in editable mode.

Verify the toolchain:

```bash
python --version
ffmpeg -version
ffprobe -version
whisper-cli -h
llama-cli -h
submaster --help
```

If `ffmpeg`, `ffprobe`, `whisper-cli`, or `llama-cli` is still missing:

```bash
conda install -n submaster -c conda-forge ffmpeg
conda install -n submaster -c conda-forge whisper.cpp
conda install -n submaster -c conda-forge llama.cpp
```

### Manual pip installation

If you are not using the Conda environment, install the package manually:

```bash
python -m pip install -e .
```

On Windows, `py -3 -m pip install -e .` works as well.

Or run it without installation:

```bash
python main.py --help
```

## Translation Backend

Subtitle translation uses **Tencent HY-MT 1.5 GGUF** models through `llama.cpp`.

Available translation sizes:

- `small`: `HY-MT1.5-1.8B-Q4_K_M.gguf`
- `large`: `HY-MT1.5-7B-Q4_K_M.gguf`

The project intentionally downloads the `Q4_K_M` variants instead of full-precision or FP8 checkpoints. For this CLI, the int4 GGUF models are the most suitable choice because they keep downloads, RAM usage, and VRAM usage reasonable while still fitting the local/offline execution goal on consumer hardware.

The models are downloaded automatically into `models/` the first time translation is requested.

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

If you choose not to use Conda, install `ffmpeg`, `git`, `cmake`, and `ninja` first:

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

Translate the generated subtitles into Italian:

```bash
submaster input.mp4 --translate-to it
```

Common patterns:

```bash
submaster input.mp4 --model turbo --device gpu
submaster input.mp4 --language en --model small
submaster input.mp4 --translate-to it --translation-model large
submaster input.mp4 --translate-to ja --device gpu --llama-cli ./llama.cpp/build/bin/llama-cli
submaster input.mkv --output ./subs/
submaster input.mov --keep-audio
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
submaster input.mp4 --show-timings
submaster input.mp4 --show-model-info
```

On Windows, `--output .\subs\`, `--whisper-cli .\whisper.cpp\build\bin\Release\whisper-cli.exe`, and `--llama-cli .\llama.cpp\build\bin\Release\llama-cli.exe` work as expected.

If `--translate-to` is set, the written `.srt` file contains the translated subtitles. If you want both source-language and translated subtitle files, run the command twice with different output paths.

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
- Tencent HY-MT1.5-1.8B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF>
- Tencent HY-MT1.5-7B-GGUF: <https://huggingface.co/tencent/HY-MT1.5-7B-GGUF>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
