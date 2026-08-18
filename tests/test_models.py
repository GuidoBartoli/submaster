import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from submaster.config import ModelSpec
from submaster.errors import SubmasterError
from submaster.models import (
    ensure_model_available,
    resolve_cleanup_model_spec,
    resolve_model_spec,
    resolve_named_model_spec,
    resolve_translation_model_spec,
    resolve_vad_model_spec,
)


class ModelSpecTests(unittest.TestCase):
    """Exercise model registry resolution helpers."""

    def _console(self) -> SimpleNamespace:
        """Create a console stub that records status and progress events."""
        events: list[tuple[str, object]] = []

        class DummyProgress:
            def update(self, completed: float, extra: str = "") -> None:
                events.append(("update", (completed, extra)))

            def finish(self, completed: float | None = None, extra: str = "") -> None:
                events.append(("finish", (completed, extra)))

        return SimpleNamespace(
            events=events,
            info=lambda message: events.append(("info", message)),
            success=lambda message: events.append(("success", message)),
            progress=lambda label, total=None, show_value=True: (
                events.append(("progress", (label, total, show_value))) or DummyProgress()
            ),
        )

    def test_large_model_maps_to_large_v3(self) -> None:
        """Verify that the `large` whisper alias resolves to the large-v3 artifact."""
        spec = resolve_model_spec("large")

        self.assertEqual(spec.filename, "ggml-large-v3.bin")
        self.assertIn("ggml-large-v3.bin", spec.download_url)

    def test_translation_large_model_maps_to_hy_mt_q4_k_m(self) -> None:
        """Verify that the large translation alias resolves to the expected GGUF file."""
        spec = resolve_translation_model_spec("large")

        self.assertEqual(spec.filename, "HY-MT1.5-7B-Q4_K_M.gguf")
        self.assertIn("HY-MT1.5-7B-Q4_K_M.gguf", spec.download_url)

    def test_cleanup_model_maps_to_qwen_q4_k_m(self) -> None:
        """Verify that transcript cleanup resolves to the expected Qwen GGUF file."""
        spec = resolve_cleanup_model_spec("qwen3.5-9b")

        self.assertEqual(spec.filename, "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertIn("Qwen3.5-9B-Q4_K_M.gguf", spec.download_url)
        self.assertIn("lmstudio-community/Qwen3.5-9B-GGUF", spec.download_url)

    def test_vad_model_maps_to_whisper_vad_artifact(self) -> None:
        """Verify that the named VAD alias resolves to the expected Hugging Face file."""
        spec = resolve_vad_model_spec("silero-v6.2.0")

        self.assertEqual(spec.filename, "ggml-silero-v6.2.0.bin")
        self.assertIn("ggml-org/whisper-vad", spec.download_url)
        self.assertIn("ggml-silero-v6.2.0.bin", spec.download_url)

    def test_model_resolution_accepts_vad_filename_alias(self) -> None:
        """Verify that VAD models can be resolved by their published filename."""
        spec = resolve_vad_model_spec("ggml-silero-v5.1.2.bin")

        self.assertEqual(spec.name, "silero-v5.1.2")

    def test_named_model_resolution_rejects_unknown_name_with_choices(self) -> None:
        """Verify that unknown model names fail with a useful choice list."""
        specs = {
            "alpha": ModelSpec("alpha", "alpha.bin", "https://example.test/alpha.bin", "Alpha"),
            "bravo": ModelSpec("bravo", "bravo.bin", "https://example.test/bravo.bin", "Bravo"),
        }

        with self.assertRaisesRegex(SubmasterError, "alpha, bravo"):
            resolve_named_model_spec("charlie", specs, model_label="test model")

    def test_ensure_model_available_reuses_existing_non_empty_file(self) -> None:
        """Verify that an existing model file is reused without opening the network."""
        spec = ModelSpec("tiny-test", "tiny-test.bin", "https://example.test/tiny.bin", "Tiny")
        console = self._console()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / spec.filename
            model_path.write_bytes(b"already here")

            with patch("submaster.models.urllib.request.urlopen") as urlopen_mock:
                resolved = ensure_model_available(
                    "tiny-test",
                    Path(tmpdir),
                    console,
                    model_specs={"tiny-test": spec},
                )

        self.assertEqual(resolved, model_path)
        urlopen_mock.assert_not_called()
        self.assertEqual(console.events[0][0], "info")

    def test_ensure_model_available_downloads_and_replaces_part_file(self) -> None:
        """Verify that missing models are streamed to disk and finalized atomically."""
        spec = ModelSpec("tiny-test", "tiny-test.bin", "https://example.test/tiny.bin", "Tiny")
        console = self._console()

        class DummyResponse:
            headers = {"Content-Length": "6"}

            def __init__(self) -> None:
                self._chunks = [b"abc", b"def", b""]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            with patch("submaster.models.urllib.request.urlopen", return_value=DummyResponse()):
                resolved = ensure_model_available(
                    "tiny-test",
                    models_dir,
                    console,
                    model_specs={"tiny-test": spec},
                )

            self.assertEqual(resolved.name, "tiny-test.bin")
            self.assertEqual(resolved.read_bytes(), b"abcdef")
            self.assertFalse(resolved.with_suffix(".bin.part").exists())
        self.assertIn(("finish", (6, "6.0 B / 6.0 B")), console.events)
        self.assertEqual(console.events[-1][0], "info")

    def test_ensure_model_available_removes_partial_file_on_download_failure(self) -> None:
        """Verify that interrupted downloads leave no reusable part file behind."""
        spec = ModelSpec("tiny-test", "tiny-test.bin", "https://example.test/tiny.bin", "Tiny")
        console = self._console()

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            with patch("submaster.models.urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(SubmasterError, "Failed to download"):
                    ensure_model_available(
                        "tiny-test",
                        models_dir,
                        console,
                        model_specs={"tiny-test": spec},
                    )

            self.assertFalse((models_dir / "tiny-test.bin.part").exists())


if __name__ == "__main__":
    unittest.main()
