import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
