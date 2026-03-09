import unittest

from submaster.models import resolve_model_spec


class ModelSpecTests(unittest.TestCase):
    def test_large_model_maps_to_large_v3(self) -> None:
        spec = resolve_model_spec("large")

        self.assertEqual(spec.filename, "ggml-large-v3.bin")
        self.assertIn("ggml-large-v3.bin", spec.download_url)


if __name__ == "__main__":
    unittest.main()
