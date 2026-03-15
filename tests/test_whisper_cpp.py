import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.errors import SubmasterError
from submaster.whisper_cpp import WhisperCppRunner


class WhisperCppRunnerTests(unittest.TestCase):
    """Exercise `whisper.cpp` discovery, parsing, and execution helpers."""

    def _make_executable(self, path: Path) -> None:
        """Create a tiny executable file for discovery tests.

        :param path: Executable path to create.
        :type path: pathlib.Path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    def test_resolve_prefers_gpu_capable_external_build(self) -> None:
        """Verify that GPU-capable discovered builds outrank CPU-only ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conda_prefix = root / "conda"
            gpu_cli = conda_prefix / "bin" / "whisper-cli"
            cpu_cli = root / "path" / "whisper-cli"
            bundled_cli = root / "repo" / "whisper" / "whisper-cli"

            self._make_executable(gpu_cli)
            self._make_executable(cpu_cli)
            self._make_executable(bundled_cli)
            (conda_prefix / "lib").mkdir(parents=True, exist_ok=True)
            (conda_prefix / "lib" / "libggml-cuda.so").write_text("", encoding="utf-8")

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            with patch.dict(os.environ, {"CONDA_PREFIX": str(conda_prefix)}, clear=True):
                with patch("submaster.whisper_cpp.shutil.which", return_value=str(cpu_cli)):
                    with patch.object(WhisperCppRunner, "_project_root", return_value=root / "repo"):
                        resolved = runner._resolve_cli_path(None)

            self.assertEqual(resolved, gpu_cli.resolve())

    def test_resolve_uses_bundled_cpu_fallback_when_external_binary_is_missing(self) -> None:
        """Verify that the bundled binary is used when no external binary is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundled_cli = root / "repo" / "whisper" / "whisper-cli"

            self._make_executable(bundled_cli)

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            with patch.dict(os.environ, {}, clear=True):
                with patch("submaster.whisper_cpp.shutil.which", return_value=None):
                    with patch.object(WhisperCppRunner, "_project_root", return_value=root / "repo"):
                        resolved = runner._resolve_cli_path(None)

            self.assertEqual(resolved, bundled_cli.resolve())

    def test_resolve_finds_windows_release_build_outputs(self) -> None:
        """Verify that Windows release build folders are searched for binaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_cli = root / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"

            self._make_executable(local_cli)

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            with patch.dict(os.environ, {}, clear=True):
                with patch("submaster.whisper_cpp.platform.system", return_value="Windows"):
                    with patch("submaster.whisper_cpp.shutil.which", return_value=None):
                        with patch("submaster.whisper_cpp.Path.cwd", return_value=root):
                            with patch.object(WhisperCppRunner, "_project_root", return_value=root / "repo"):
                                resolved = runner._resolve_cli_path(None)

            self.assertEqual(resolved, local_cli.resolve())

    def test_resolve_skips_linux_bundled_fallback_on_windows(self) -> None:
        """Verify that the Linux-only bundled fallback is ignored on Windows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundled_cli = root / "repo" / "whisper" / "whisper-cli"

            self._make_executable(bundled_cli)

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            with patch.dict(os.environ, {}, clear=True):
                with patch("submaster.whisper_cpp.platform.system", return_value="Windows"):
                    with patch("submaster.whisper_cpp.shutil.which", return_value=None):
                        with patch.object(WhisperCppRunner, "_project_root", return_value=root / "repo"):
                            with self.assertRaises(SubmasterError) as ctx:
                                runner._resolve_cli_path(None)

        self.assertIn("only shipped for Linux x86_64", str(ctx.exception))

    def test_detect_gpu_backends_finds_windows_conda_dlls(self) -> None:
        """Verify that GPU backend detection picks up Conda DLL layouts on Windows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cli_path = root / "conda" / "Scripts" / "whisper-cli.exe"
            self._make_executable(cli_path)
            dll_path = root / "conda" / "Library" / "bin" / "ggml-cuda.dll"
            dll_path.parent.mkdir(parents=True, exist_ok=True)
            dll_path.write_text("", encoding="utf-8")

            runner = WhisperCppRunner.__new__(WhisperCppRunner)

            self.assertEqual(runner._detect_gpu_backends_for(cli_path), {"cuda"})

    def test_verify_gpu_runtime_linkage_accepts_cuda_linked_binary(self) -> None:
        """Verify that CUDA-linked runtimes pass the linkage safety check."""
        runner = WhisperCppRunner.__new__(WhisperCppRunner)
        runner.cli_path = Path("/tmp/whisper-cli")

        with patch("submaster.whisper_cpp.platform.system", return_value="Linux"):
            with patch.object(
                WhisperCppRunner,
                "_inspect_linked_ggml_libraries",
                return_value=({"cuda"}, ["libggml-cuda.so.0 => /tmp/libggml-cuda.so.0"]),
            ):
                runner._verify_gpu_runtime_linkage("gpu", "gpu")

    def test_verify_gpu_runtime_linkage_rejects_cpu_only_runtime_libraries(self) -> None:
        """Verify that CPU-only runtime linkage is rejected in GPU mode."""
        runner = WhisperCppRunner.__new__(WhisperCppRunner)
        runner.cli_path = Path("/tmp/whisper-cli")

        with patch("submaster.whisper_cpp.platform.system", return_value="Linux"):
            with patch.object(
                WhisperCppRunner,
                "_inspect_linked_ggml_libraries",
                return_value=(
                    set(),
                    [
                        "libggml.so.0 => /conda/lib/libggml.so.0",
                        "libggml-cpu.so.0 => /conda/lib/libggml-cpu.so.0",
                    ],
                ),
            ):
                with self.assertRaises(SubmasterError) as ctx:
                    runner._verify_gpu_runtime_linkage("gpu", "gpu")

        self.assertIn("CPU-only ggml libraries", str(ctx.exception))
        self.assertIn("libggml-cpu.so.0", str(ctx.exception))

    def test_resolve_device_auto_prefers_any_detected_gpu_backend(self) -> None:
        """Verify that auto mode selects GPU when any backend is detected."""
        runner = WhisperCppRunner.__new__(WhisperCppRunner)
        runner.gpu_backends = {"vulkan"}

        with patch("submaster.whisper_cpp.platform.system", return_value="Linux"):
            self.assertEqual(runner.resolve_device("auto"), "gpu")

    def test_extract_timing_lines_keeps_whisper_cpp_timings(self) -> None:
        """Verify that timing extraction keeps only timing detail lines."""
        runner = WhisperCppRunner.__new__(WhisperCppRunner)

        lines = [
            "progress = 100%",
            "whisper_print_timings:     load time =   112.42 ms",
            "some other log",
            "whisper_print_timings:    total time =  3466.45 ms",
        ]

        self.assertEqual(
            runner._extract_timing_lines(lines),
            [
                "load time =   112.42 ms",
                "total time =  3466.45 ms",
            ],
        )

    def test_extract_model_load_lines_keeps_whisper_cpp_model_details(self) -> None:
        """Verify that model-load extraction keeps only model metadata lines."""
        runner = WhisperCppRunner.__new__(WhisperCppRunner)

        lines = [
            "whisper_model_load: loading model from 'models/ggml.bin'",
            "progress = 100%",
            "whisper_model_load: n_vocab       = 51864",
            "other log",
            "whisper_model_load: model size    = 73.54 MB",
        ]

        self.assertEqual(
            runner._extract_model_load_lines(lines),
            [
                "loading model from 'models/ggml.bin'",
                "n_vocab       = 51864",
                "model size    = 73.54 MB",
            ],
        )

    def test_run_keeps_timings_enabled_in_whisper_cli(self) -> None:
        """Verify that transcription keeps progress parsing while timings are enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_base = root / "out"
            vad_model = root / "vad.bin"
            output_base.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
            vad_model.write_text("vad", encoding="utf-8")

            progress_calls: list[tuple[str, float | None, str, bool]] = []
            progress_updates: list[tuple[float, str]] = []

            class DummyProgress:
                """Progress recorder used to assert whisper progress behavior."""

                def update(self, completed: float, extra: str = "") -> None:
                    """Capture progress updates for later assertions.

                    :param completed: Completed progress value.
                    :type completed: float
                    :param extra: Optional progress suffix.
                    :type extra: str
                    """
                    progress_updates.append((completed, extra))

                def finish(self, completed: float | None = None, extra: str = "") -> None:
                    """Capture final progress state for later assertions.

                    :param completed: Completed progress value.
                    :type completed: float | None
                    :param extra: Optional progress suffix.
                    :type extra: str
                    """
                    progress_updates.append((completed or 0.0, extra))

            console = SimpleNamespace(
                info=lambda message: None,
                warn=lambda message: None,
                line=lambda message="": None,
                progress=lambda label, total, unit="", show_value=True: (
                    progress_calls.append((label, total, unit, show_value)) or DummyProgress()
                ),
            )

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            runner.console = console
            runner.cli_path = Path("/tmp/whisper-cli")
            runner.supports_no_gpu_flag = True
            runner.gpu_backends = {"cuda"}

            popen_calls: list[list[str]] = []

            class DummyProcess:
                """Minimal process stub that exposes stdout iteration and exit status."""

                def __init__(self) -> None:
                    """Seed the process with canned `whisper.cpp` output."""
                    self.stdout = iter(
                        [
                            "progress = 15%\n",
                            "progress = 100%\n",
                            "whisper_print_timings:     total time =  1234.56 ms\n",
                        ]
                    )

                def wait(self) -> int:
                    """Return a successful exit status.

                    :returns: Zero exit status.
                    :rtype: int
                    """
                    return 0

            def fake_popen(command, **_kwargs):
                """Record the spawned command and return a dummy process.

                :param command: Subprocess command line.
                :type command: list[str]
                :returns: Dummy process instance for the test.
                :rtype: DummyProcess
                """
                popen_calls.append(command)
                return DummyProcess()

            with patch.object(WhisperCppRunner, "resolve_device", return_value="gpu"):
                with patch.object(WhisperCppRunner, "_verify_gpu_runtime_linkage", return_value=None):
                    with patch("submaster.whisper_cpp.subprocess.Popen", side_effect=fake_popen):
                        runner.run(
                            audio_path=root / "audio.wav",
                            model_path=root / "model.bin",
                            output_base=output_base,
                            language="auto",
                            requested_device="gpu",
                            threads=4,
                            max_context=0,
                            vad_model_path=vad_model,
                            show_timings=True,
                            show_model_info=False,
                        )

            self.assertEqual(len(popen_calls), 1)
            self.assertNotIn("-np", popen_calls[0])
            self.assertIn("-pp", popen_calls[0])
            self.assertIn("-l", popen_calls[0])
            self.assertIn("auto", popen_calls[0])
            self.assertIn("-mc", popen_calls[0])
            self.assertIn("0", popen_calls[0])
            self.assertIn("--vad", popen_calls[0])
            self.assertIn(str(vad_model), popen_calls[0])
            self.assertEqual(progress_calls, [("whisper", 100, "%", False)])
            self.assertEqual(progress_updates, [(15, ""), (100, ""), (100, "")])

    def test_resolve_srt_path_accepts_audio_suffix_fallback_name(self) -> None:
        """Verify that alternate `whisper.cpp` SRT naming schemes are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "input.wav"
            output_base = root / "transcript"
            fallback_srt_path = audio_path.with_suffix(".wav.srt")
            warnings: list[str] = []

            fallback_srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhi\n",
                encoding="utf-8",
            )

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            runner.console = SimpleNamespace(warn=lambda message: warnings.append(message))

            resolved = runner._resolve_srt_path(output_base, audio_path)

        self.assertEqual(resolved, fallback_srt_path)
        self.assertEqual(len(warnings), 1)
        self.assertIn("input.wav.srt", warnings[0])


if __name__ == "__main__":
    unittest.main()
