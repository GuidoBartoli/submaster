# Bundled `whisper-cli`

This folder contains the repo-local `whisper.cpp` fallback used only when no external `whisper-cli` executable is found.

- Upstream project: `ggml-org/whisper.cpp`
- Upstream release: `v1.8.3`
- Upstream commit: `2eeeba56e9edd762b4b38467bab96c2517163158`
- Target platform: Linux `x86_64`
- Build type: CPU-only fallback
- Build command: `cmake -S . -B build-static -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF`

The bundled executable was built on March 9, 2026 from the official upstream source tree.

Runtime notes:

- This is a CPU fallback, not a CUDA build.
- It still depends on standard host libraries such as `glibc`, `libstdc++`, and `libgomp`.
- If a GPU-capable `whisper-cli` is installed in Conda, on `PATH`, or in a common local build folder, `submaster` will prefer that external executable instead.

Upstream `whisper.cpp` is distributed under the MIT license:

```text
MIT License

Copyright (c) 2023-2025 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
