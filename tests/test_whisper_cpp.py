import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.errors import SubmasterError
from submaster.whisper_cpp import WhisperCppRunner


class WhisperCppRunnerTests(unittest.TestCase):
    def _make_executable(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    def test_resolve_prefers_gpu_capable_external_build(self) -> None:
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

    def test_verify_gpu_runtime_linkage_accepts_cuda_linked_binary(self) -> None:
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

    def test_extract_timing_lines_keeps_whisper_cpp_timings(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_base = root / "out"
            output_base.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

            progress_updates: list[tuple[float, str]] = []

            class DummyProgress:
                def update(self, completed: float, extra: str = "") -> None:
                    progress_updates.append((completed, extra))

                def finish(self, completed: float | None = None, extra: str = "") -> None:
                    progress_updates.append((completed or 0.0, extra))

            console = SimpleNamespace(
                info=lambda message: None,
                warn=lambda message: None,
                line=lambda message="": None,
                progress=lambda label, total, unit="": DummyProgress(),
            )

            runner = WhisperCppRunner.__new__(WhisperCppRunner)
            runner.console = console
            runner.cli_path = Path("/tmp/whisper-cli")
            runner.supports_no_gpu_flag = True
            runner.gpu_backends = {"cuda"}

            popen_calls: list[list[str]] = []

            class DummyProcess:
                def __init__(self) -> None:
                    self.stdout = iter(
                        [
                            "progress = 100%\n",
                            "whisper_print_timings:     total time =  1234.56 ms\n",
                        ]
                    )

                def wait(self) -> int:
                    return 0

            def fake_popen(command, **_kwargs):
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
                            show_timings=True,
                            show_model_info=False,
                        )

            self.assertEqual(len(popen_calls), 1)
            self.assertNotIn("-np", popen_calls[0])
            self.assertIn("-pp", popen_calls[0])


if __name__ == "__main__":
    unittest.main()
