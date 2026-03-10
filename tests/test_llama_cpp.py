import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submaster.errors import SubmasterError
from submaster.llama_cpp import LlamaCppRunner


class LlamaCppRunnerTests(unittest.TestCase):
    """Exercise `llama.cpp` discovery and execution helpers."""

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
            gpu_cli = conda_prefix / "bin" / "llama-cli"
            cpu_cli = root / "path" / "llama-cli"

            self._make_executable(gpu_cli)
            self._make_executable(cpu_cli)
            (conda_prefix / "lib").mkdir(parents=True, exist_ok=True)
            (conda_prefix / "lib" / "libggml-cuda.so").write_text("", encoding="utf-8")

            runner = LlamaCppRunner.__new__(LlamaCppRunner)
            with patch.dict(os.environ, {"CONDA_PREFIX": str(conda_prefix)}, clear=True):
                with patch("submaster.llama_cpp.shutil.which", return_value=str(cpu_cli)):
                    with patch.object(LlamaCppRunner, "_project_root", return_value=root / "repo"):
                        resolved = runner._resolve_cli_path(None)

            self.assertEqual(resolved, gpu_cli.resolve())

    def test_resolve_finds_windows_release_build_outputs(self) -> None:
        """Verify that Windows release build folders are searched for binaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_cli = root / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"

            self._make_executable(local_cli)

            runner = LlamaCppRunner.__new__(LlamaCppRunner)
            with patch.dict(os.environ, {}, clear=True):
                with patch("submaster.llama_cpp.platform.system", return_value="Windows"):
                    with patch("submaster.llama_cpp.shutil.which", return_value=None):
                        with patch("submaster.llama_cpp.Path.cwd", return_value=root):
                            with patch.object(LlamaCppRunner, "_project_root", return_value=root / "repo"):
                                resolved = runner._resolve_cli_path(None)

            self.assertEqual(resolved, local_cli.resolve())

    def test_resolve_raises_when_no_binary_is_found(self) -> None:
        """Verify that discovery fails cleanly when no executable exists."""
        runner = LlamaCppRunner.__new__(LlamaCppRunner)
        with patch.dict(os.environ, {}, clear=True):
            with patch("submaster.llama_cpp.shutil.which", return_value=None):
                with patch.object(LlamaCppRunner, "_project_root", return_value=Path("/tmp/repo")):
                    with self.assertRaises(SubmasterError):
                        runner._resolve_cli_path(None)

    def test_run_prompt_adds_ngl_zero_for_cpu(self) -> None:
        """Verify that CPU mode forces zero GPU layers when the flag is supported."""
        runner = LlamaCppRunner.__new__(LlamaCppRunner)
        runner.console = type(
            "ConsoleStub",
            (),
            {
                "warn": lambda self, message: None,
                "info": lambda self, message: None,
                "spinner": lambda self, label: _SpinnerStub(),
            },
        )()
        runner.cli_path = Path("/tmp/llama-cli")
        runner.supports_ngl_flag = True
        runner.supports_simple_io = True
        runner.supports_no_display_prompt = True
        runner.supports_no_warmup = True
        runner.gpu_backends = set()
        runner._announced_modes = set()

        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            """Capture the command and return a successful translation response.

            :param command: Subprocess command line.
            :type command: list[str]
            :returns: Minimal completed-process stub with translated output.
            :rtype: object
            """
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "[[[cue:1]]]\nciao\n[[[/cue]]]", "stderr": ""})()

        with patch.object(LlamaCppRunner, "_verify_gpu_runtime_linkage", return_value=None):
            with patch("submaster.llama_cpp.subprocess.run", side_effect=fake_run):
                output = runner.run_prompt(
                    model_path=Path("/tmp/model.gguf"),
                    prompt="translate this",
                    requested_device="cpu",
                    threads=4,
                )

        self.assertEqual(output, "[[[cue:1]]]\nciao\n[[[/cue]]]")
        self.assertIn("-ngl", calls[0])
        self.assertIn("0", calls[0])


class _SpinnerStub:
    """Context-manager stub that mimics the console spinner in tests."""

    def __enter__(self):
        """Enter the stub context manager.

        :returns: The stub itself.
        :rtype: _SpinnerStub
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the stub context manager without suppressing exceptions.

        :param exc_type: Exception type raised inside the context, if any.
        :type exc_type: type | None
        :param exc: Exception instance raised inside the context, if any.
        :type exc: BaseException | None
        :param tb: Traceback object for the active exception, if any.
        :type tb: object
        :returns: `False` so exceptions are never suppressed.
        :rtype: bool
        """
        return False


if __name__ == "__main__":
    unittest.main()
