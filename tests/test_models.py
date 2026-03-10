import unittest

from submaster.models import resolve_model_spec, resolve_translation_model_spec


class ModelSpecTests(unittest.TestCase):
    def test_large_model_maps_to_large_v3(self) -> None:
        spec = resolve_model_spec("large")

        self.assertEqual(spec.filename, "ggml-large-v3.bin")
        self.assertIn("ggml-large-v3.bin", spec.download_url)

    def test_translation_large_model_maps_to_hy_mt_q4_k_m(self) -> None:
        spec = resolve_translation_model_spec("large")

        self.assertEqual(spec.filename, "HY-MT1.5-7B-Q4_K_M.gguf")
        self.assertIn("HY-MT1.5-7B-Q4_K_M.gguf", spec.download_url)


if __name__ == "__main__":
    unittest.main()
