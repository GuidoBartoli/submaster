import unittest
from pathlib import Path
from types import SimpleNamespace

from submaster.srt import Cue
from submaster.translation import SubtitleTranslator, resolve_translation_language


class DummyRunner:
    """Minimal runner stub that returns scripted translation responses."""

    def __init__(self, responses: list[str]) -> None:
        """Store queued responses and capture prompts for assertions.

        :param responses: Translation outputs to return on successive calls.
        :type responses: list[str]
        """
        self.responses = responses
        self.prompts: list[str] = []

    def run_prompt(self, model_path: Path, prompt: str, requested_device: str, threads: int) -> str:
        """Return the next scripted response and remember the incoming prompt.

        :param model_path: Model path provided by the translator.
        :type model_path: pathlib.Path
        :param prompt: Prompt text passed to the runner.
        :type prompt: str
        :param requested_device: Requested device mode for the call.
        :type requested_device: str
        :param threads: Thread count for the call.
        :type threads: int
        :returns: Next scripted translation response.
        :rtype: str
        """
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No dummy translation responses left.")
        return self.responses.pop(0)


class TranslationTests(unittest.TestCase):
    """Exercise subtitle translation batching and fallback behavior."""

    def _console(self) -> SimpleNamespace:
        """Create a console stub with no-op progress reporting.

        :returns: Console-like namespace for translator tests.
        :rtype: types.SimpleNamespace
        """
        class DummyProgress:
            """No-op progress helper used by translation tests."""

            def update(self, _completed: float, extra: str = "") -> None:
                """Ignore intermediate progress updates in tests.

                :param _completed: Completed work value.
                :type _completed: float
                :param extra: Optional progress suffix.
                :type extra: str
                """
                return None

            def finish(self, _completed: float | None = None, extra: str = "") -> None:
                """Ignore final progress updates in tests.

                :param _completed: Completed work value.
                :type _completed: float | None
                :param extra: Optional progress suffix.
                :type extra: str
                """
                return None

        return SimpleNamespace(
            note=lambda message: None,
            warn=lambda message: None,
            progress=lambda label, total, unit="": DummyProgress(),
        )

    def test_resolve_translation_language_accepts_alias(self) -> None:
        """Verify that common language aliases normalize to canonical codes."""
        language = resolve_translation_language("Italian")

        self.assertEqual(language.code, "it")
        self.assertEqual(language.name, "Italian")

    def test_translate_cues_preserves_timings(self) -> None:
        """Verify that translated cues keep their original timing boundaries."""
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
        """Verify that invalid batch markup triggers single-cue retry behavior."""
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
        """Verify that prompts include the normalized source-language hint."""
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
