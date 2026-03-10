import unittest
from pathlib import Path
from types import SimpleNamespace

from submaster.srt import Cue
from submaster.translation import SubtitleTranslator, resolve_translation_language


class DummyRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def run_prompt(self, model_path: Path, prompt: str, requested_device: str, threads: int) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No dummy translation responses left.")
        return self.responses.pop(0)


class TranslationTests(unittest.TestCase):
    def _console(self) -> SimpleNamespace:
        class DummyProgress:
            def update(self, _completed: float, extra: str = "") -> None:
                return None

            def finish(self, _completed: float | None = None, extra: str = "") -> None:
                return None

        return SimpleNamespace(
            note=lambda message: None,
            warn=lambda message: None,
            progress=lambda label, total, unit="": DummyProgress(),
        )

    def test_resolve_translation_language_accepts_alias(self) -> None:
        language = resolve_translation_language("Italian")

        self.assertEqual(language.code, "it")
        self.assertEqual(language.name, "Italian")

    def test_translate_cues_preserves_timings(self) -> None:
        runner = DummyRunner(
            [
                "[[[cue:1]]]\nciao\n[[[/cue]]]\n\n[[[cue:2]]]\ncome stai?\n[[[/cue]]]",
            ]
        )
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        translated = translator.translate_cues(
            [
                Cue(start_ms=0, end_ms=1_000, text=["hello"]),
                Cue(start_ms=1_000, end_ms=2_000, text=["how are you?"]),
            ]
        )

        self.assertEqual(translated[0].text, ["ciao"])
        self.assertEqual(translated[1].text, ["come stai?"])
        self.assertEqual(translated[0].start_ms, 0)
        self.assertEqual(translated[1].end_ms, 2_000)

    def test_translate_batch_falls_back_to_single_cues_when_markup_is_invalid(self) -> None:
        runner = DummyRunner(
            [
                "This is not parseable.",
                "[[[cue:1]]]\nciao\n[[[/cue]]]",
                "[[[cue:1]]]\narrivederci\n[[[/cue]]]",
            ]
        )
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        translated = translator.translate_cues(
            [
                Cue(start_ms=0, end_ms=1_000, text=["hello"]),
                Cue(start_ms=1_000, end_ms=2_000, text=["goodbye"]),
            ]
        )

        self.assertEqual(translated[0].text, ["ciao"])
        self.assertEqual(translated[1].text, ["arrivederci"])
        self.assertEqual(len(runner.prompts), 3)

    def test_build_batch_prompt_uses_source_language_hint(self) -> None:
        runner = DummyRunner(["[[[cue:1]]]\nciao\n[[[/cue]]]"])
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        prompt = translator._build_batch_prompt([Cue(start_ms=0, end_ms=1_000, text=["hello"])])

        self.assertIn("Translate the following subtitles into Italian.", prompt)
        self.assertIn("Source language: English.", prompt)
        self.assertIn("[[[cue:1]]]", prompt)


if __name__ == "__main__":
    unittest.main()
