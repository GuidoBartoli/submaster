import unittest

from submaster.models import (
    resolve_cleanup_model_spec,
    resolve_model_spec,
    resolve_translation_model_spec,
    resolve_vad_model_spec,
)


class ModelSpecTests(unittest.TestCase):
    """Exercise model registry resolution helpers."""

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


if __name__ == "__main__":
    unittest.main()
