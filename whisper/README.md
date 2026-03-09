# Bundled `whisper-cli`

This folder documents the repo-local `whisper.cpp` fallback that `submaster` may use when no external executable is found.

- Upstream project: `ggml-org/whisper.cpp`
- Upstream release: `v1.8.3`
- Upstream commit: `2eeeba56e9edd762b4b38467bab96c2517163158`
- Bundled artifact: `whisper-cli`
- Target platform: Linux `x86_64`
- Build type: CPU-only fallback
- Build command: `cmake -S . -B build-static -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF`

The bundled executable was built on March 9, 2026 from the official upstream source tree.

## Platform Notes

- This bundled fallback is Linux-only.
- macOS and Windows users must install or build `whisper.cpp` separately.
- `submaster` will not select this Linux binary on macOS or Windows.
- If a GPU-capable `whisper-cli` is available in Conda, on `PATH`, or in a common local build folder, `submaster` will prefer that external executable instead.

## Runtime Notes

- This is a CPU fallback, not a CUDA, Metal, Vulkan, or OpenCL build.
- It still depends on standard host libraries such as `glibc`, `libstdc++`, and `libgomp`.

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
