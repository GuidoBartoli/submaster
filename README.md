# submaster

`submaster` is a command-line Python application that turns an audio or video file into a synchronized `.srt` subtitle file.

It uses:

- `ffmpeg` / `ffprobe` to normalize any input into a mono 16 kHz WAV file
- `whisper.cpp` to run local transcription with CPU or GPU when available
- automatic GGML model downloads into `./models`
- an SRT normalization pass so the final file is clean and VLC-friendly

## Features

- Accepts audio or video input
- Extracts the audio track first when the input is a video
- Supports `tiny`, `base`, `small`, `medium`, `large`, and `turbo`
- Downloads missing models on demand
- Prefers GPU in `--device auto` when a local GPU is detectable
- Produces compliant `.srt` output with normalized cue numbering and timestamps
- Shows colored terminal output, progress bars, and a transcription spinner

## Project Layout

```text
main.py                  Simple entry point
submaster/               CLI package
tests/                   Small unit test suite for SRT normalization
models/                  Auto-downloaded whisper.cpp model files
```

## Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`
- a working `whisper.cpp` executable (`whisper-cli` recommended)

`submaster` itself has no third-party Python runtime dependencies. The Conda environment below installs the project plus the toolchain needed to build `whisper.cpp`.

## Conda Setup

Create the environment from [environment.yml](/home/guido/Projects/submaster/environment.yml):

```bash
conda env create -f environment.yml
conda activate submaster
```

This environment installs:

- Python 3.12
- `ffmpeg` from `conda-forge`
- `ffprobe` as part of the `ffmpeg` package
- `git`, `cmake`, `ninja`, `c-compiler`, and `cxx-compiler` so you can build `whisper.cpp`
- the current project in editable mode via `pip -e .`

Verify the main tools after activation:

```bash
python --version
ffmpeg -version
ffprobe -version
submaster --help
```

If `ffmpeg` or `ffprobe` is still missing for any reason, reinstall it inside the environment:

```bash
conda install -n submaster -c conda-forge ffmpeg
```

## Install `whisper.cpp`

`submaster` calls a local `whisper.cpp` executable. The Conda environment gives you the build tools, but you still need to fetch and build `whisper.cpp` itself.

### CPU build

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -G Ninja
cmake --build whisper.cpp/build -j --config Release
```

The resulting executable is usually:

```bash
./whisper.cpp/build/bin/whisper-cli
```

You can either add that directory to `PATH` or pass it explicitly:

```bash
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
```

### NVIDIA GPU build

If you want GPU execution on NVIDIA hardware, `whisper.cpp`'s official build instructions require the CUDA toolkit to be installed on the system first. Then build with CUDA enabled:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -G Ninja -DGGML_CUDA=1
cmake --build whisper.cpp/build -j --config Release
```

For some newer NVIDIA GPUs, `whisper.cpp` also documents explicitly setting `CMAKE_CUDA_ARCHITECTURES`.

### If build tools are missing outside Conda

If you choose not to use the Conda environment, install the missing tools manually.

Ubuntu or Debian:

```bash
sudo apt install ffmpeg git cmake ninja-build build-essential
```

macOS with Homebrew:

```bash
brew install ffmpeg git cmake ninja
```

`whisper.cpp` can then be built with the same `cmake` commands shown above.

## Install the CLI Without Conda

If you are not using the Conda environment, install the package manually:

```bash
python3 -m pip install -e .
```

You can also run it without installation:

```bash
python3 main.py --help
```

## Usage

Basic:

```bash
submaster input.mp4
```

Choose model and force GPU:

```bash
submaster input.mp4 --model turbo --device gpu
```

Specify language for faster decoding:

```bash
submaster input.wav --language en --model small
```

Write subtitles into a custom folder:

```bash
submaster input.mkv --output ./subs/
```

Keep the normalized WAV file:

```bash
submaster input.mov --keep-audio
```

Use a specific `whisper.cpp` binary:

```bash
submaster input.mp4 --whisper-cli ./whisper.cpp/build/bin/whisper-cli
```

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
```

## Notes

- `large` maps to the `ggml-large.bin` whisper.cpp model.
- `turbo` maps to `ggml-large-v3-turbo.bin`.
- `--device auto` currently prefers GPU on macOS or when `nvidia-smi` reports an NVIDIA device. Actual GPU execution still depends on how `whisper.cpp` was built.
- Model files are intentionally ignored by git through `.gitignore`.

## Validation

Run the small test suite:

```bash
python3 -m unittest
```

Check the CLI wiring:

```bash
python3 main.py --help
```

## Upstream References

- OpenAI Whisper: <https://github.com/openai/whisper>
- whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
- whisper.cpp GGML models: <https://huggingface.co/ggerganov/whisper.cpp>
