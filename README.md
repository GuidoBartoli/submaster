```                                                               
  mmmm         #      m    m                 m                 
 #"   " m   m  #mmm   ##  ##  mmm    mmm   mm#mm   mmm    m mm 
 "#mmm  #   #  #" "#  # ## # "   #  #   "    #    #"  #   #"  "
     "# #   #  #   #  # "" # m"""#   """m    #    #""""   #    
 "mmm#" "mm"#  ##m#"  #    # "mm"#  "mmm"    "mm  "#mm"   #    
```

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
whisper/                 Bundled Linux x86_64 CPU fallback for whisper.cpp
models/                  Auto-downloaded whisper.cpp model files
```

## Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`
- a working `whisper.cpp` executable (`whisper-cli` recommended)

`submaster` itself has no third-party Python runtime dependencies. The Conda environment below installs the project plus the toolchain needed to build `whisper.cpp`.

If no external `whisper-cli` is available, `submaster` falls back to the bundled Linux x86_64 CPU binary in [whisper/whisper-cli](/home/guido/Projects/submaster/whisper/whisper-cli).

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
- `whisper.cpp` from `conda-forge`
- `git`, `cmake`, `ninja`, `c-compiler`, and `cxx-compiler` so you can build `whisper.cpp`
- the current project in editable mode via `pip -e .`

Verify the main tools after activation:

```bash
python --version
ffmpeg -version
ffprobe -version
whisper-cli -h
submaster --help
```

If `ffmpeg`, `ffprobe`, or `whisper-cli` is still missing for any reason, reinstall them inside the environment:

```bash
conda install -n submaster -c conda-forge ffmpeg
conda install -n submaster -c conda-forge whisper.cpp
```

## Install `whisper.cpp`

`submaster` calls a local `whisper.cpp` executable. In the normal Conda setup above, `whisper.cpp` should already be installed and `whisper-cli` should be available immediately.

Executable lookup order is:

1. `--whisper-cli`
2. `WHISPER_CPP_CLI`
3. Conda / `PATH` / common local build outputs, preferring a GPU-capable build when one is detected
4. bundled repo fallback at [whisper/whisper-cli](/home/guido/Projects/submaster/whisper/whisper-cli)

Do not use:

```bash
pip install whisper-cli
```

That PyPI package is unrelated to the native `whisper.cpp` executable expected by this project.

### Preferred Conda install

If your environment was created before `whisper.cpp` was added to [environment.yml](/home/guido/Projects/submaster/environment.yml), install it explicitly:

```bash
conda install -n submaster -c conda-forge whisper.cpp
```

Then confirm:

```bash
whisper-cli -h
```

### Build from source instead

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

## GPU Troubleshooting

If `submaster` says GPU mode was requested but the machine still runs at high CPU usage, check the backend that your `whisper-cli` binary was built with:

```bash
ldd "$(which whisper-cli)" | grep ggml
```

If you only see `libggml-cpu` and do not see `libggml-cuda`, your current `whisper.cpp` binary is CPU-only. In that case, installing Python packages will not enable GPU acceleration. You need a CUDA-enabled `whisper.cpp` build instead.

You can also confirm it directly:

```bash
whisper-cli -m ./models/ggml-base.bin -f sample.wav -pp -np
```

If startup logs contain `whisper_backend_init_gpu: no GPU found`, the current executable is not using your NVIDIA GPU.

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
