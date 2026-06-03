import unittest
from pathlib import Path
from types import SimpleNamespace

from submaster.errors import SubmasterError
from submaster.srt import Cue
from submaster.translation import SubtitleTranslator, resolve_translation_language


class DummyRunner:
    """Minimal runner stub that returns scripted translation responses."""

    def __init__(
        self,
        responses: list[str],
        *,
        supports_conversation: bool = False,
        supports_single_turn: bool = False,
    ) -> None:
        """Store queued responses and capture prompts for assertions.

        :param responses: Translation outputs to return on successive calls.
        :type responses: list[str]
        """
        self.responses = responses
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []
        self.supports_conversation = supports_conversation
        self.supports_single_turn = supports_single_turn

    def run_prompt(
        self,
        model_path: Path,
        prompt: str,
        requested_device: str,
        threads: int,
        system_prompt: str | None = None,
        show_spinner: bool = True,
    ) -> str:
        """Return the next scripted response and remember the incoming prompt.

        :param model_path: Model path provided by the translator.
        :type model_path: pathlib.Path
        :param prompt: Prompt text passed to the runner.
        :type prompt: str
        :param requested_device: Requested device mode for the call.
        :type requested_device: str
        :param threads: Thread count for the call.
        :type threads: int
        :param system_prompt: Optional system prompt used by chat-style runtimes.
        :type system_prompt: str | None
        :param show_spinner: Whether the caller requested spinner output.
        :type show_spinner: bool
        :returns: Next scripted translation response.
        :rtype: str
        """
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
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
            info=lambda message: None,
            warn=lambda message: None,
            progress=lambda label, total, unit="": DummyProgress(),
        )

    def test_resolve_translation_language_accepts_alias(self) -> None:
        """Verify that common language aliases normalize to canonical codes."""
        language = resolve_translation_language("Italian")

        self.assertEqual(language.code, "it")
        self.assertEqual(language.name, "Italian")

    def test_resolve_translation_language_rejects_empty_values(self) -> None:
        """Verify that blank target languages fail before prompting the model."""
        with self.assertRaisesRegex(SubmasterError, "cannot be empty"):
            resolve_translation_language("   ")

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

    def test_translate_srt_renders_translated_document(self) -> None:
        """Verify full SRT translation parses and renders canonical SRT output."""
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

        translated = translator.translate_srt(
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
        )

        self.assertEqual(translated, "1\r\n00:00:00,000 --> 00:00:01,000\r\nciao\r\n")

    def test_translate_cues_rejects_empty_input(self) -> None:
        """Verify translation refuses empty cue lists."""
        translator = SubtitleTranslator(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        with self.assertRaisesRegex(SubmasterError, "No subtitle cues"):
            translator.translate_cues([])

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

    def test_translate_cues_emits_fallback_warning_only_once_per_run(self) -> None:
        """Verify repeated batch parse failures do not spam the same warning."""
        warnings: list[str] = []

        class DummyProgress:
            """No-op progress helper used by translation tests."""

            def update(self, _completed: float, extra: str = "") -> None:
                return None

            def finish(self, _completed: float | None = None, extra: str = "") -> None:
                return None

        console = SimpleNamespace(
            info=lambda message: None,
            warn=warnings.append,
            progress=lambda label, total, unit="": DummyProgress(),
        )
        runner = DummyRunner(
            [
                "not parseable batch one",
                "[[[cue:1]]]\nciao\n[[[/cue]]]",
                "[[[cue:1]]]\narrivederci\n[[[/cue]]]",
                "not parseable batch two",
                "[[[cue:1]]]\ngrazie\n[[[/cue]]]",
                "[[[cue:1]]]\nprego\n[[[/cue]]]",
            ]
        )
        translator = SubtitleTranslator(
            console=console,
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
                Cue(start_ms=2_000, end_ms=3_000, text=["thanks"]),
                Cue(start_ms=3_000, end_ms=4_000, text=["you are welcome"]),
            ]
        )

        self.assertEqual([cue.text for cue in translated], [["ciao"], ["arrivederci"], ["grazie"], ["prego"]])
        self.assertEqual(
            warnings,
            [
                "Some translation batch responses could not be parsed cleanly. "
                "Retrying failed batches one cue at a time."
            ],
        )

    def test_small_model_uses_more_conservative_default_batch_size(self) -> None:
        """Verify that the default batch cap is lowered for the 1.8B translation model."""
        runner = DummyRunner([])
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        self.assertEqual(translator.max_batch_cues, 2)

    def test_batch_cues_prefers_sentence_boundary_over_mid_sentence_end(self) -> None:
        """Verify that batching avoids ending on a mid-sentence fragment when it can stop cleanly."""
        translator = SubtitleTranslator(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/HY-MT1.5-7B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
            max_batch_cues=3,
            max_batch_chars=120,
        )

        batches = translator._batch_cues(
            [
                Cue(start_ms=0, end_ms=800, text=["This is"]),
                Cue(start_ms=850, end_ms=1_600, text=["one sentence."]),
                Cue(start_ms=1_650, end_ms=2_300, text=["This next bit"]),
                Cue(start_ms=2_350, end_ms=3_200, text=["continues onward."]),
            ]
        )

        self.assertEqual([[line for cue in batch for line in cue.text] for batch in batches], [
            ["This is", "one sentence."],
            ["This next bit", "continues onward."],
        ])

    def test_batch_cues_keeps_lowercase_continuation_with_previous_cues(self) -> None:
        """Verify that likely sentence continuations stay with the preceding batch."""
        translator = SubtitleTranslator(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/HY-MT1.5-7B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
            max_batch_cues=3,
            max_batch_chars=160,
        )

        batches = translator._batch_cues(
            [
                Cue(start_ms=0, end_ms=600, text=["I think"]),
                Cue(start_ms=650, end_ms=1_250, text=["this works"]),
                Cue(start_ms=1_300, end_ms=2_000, text=["well now."]),
                Cue(start_ms=2_700, end_ms=3_200, text=["Next sentence."]),
            ]
        )

        self.assertEqual([len(batch) for batch in batches], [3, 1])

    def test_batch_cues_prefers_long_pause_as_boundary(self) -> None:
        """Verify that a large timing gap becomes a natural batch break."""
        translator = SubtitleTranslator(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/HY-MT1.5-7B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
            max_batch_cues=4,
            max_batch_chars=180,
        )

        batches = translator._batch_cues(
            [
                Cue(start_ms=0, end_ms=700, text=["First part"]),
                Cue(start_ms=750, end_ms=1_400, text=["continues"]),
                Cue(start_ms=2_500, end_ms=3_100, text=["After a pause"]),
                Cue(start_ms=3_150, end_ms=3_900, text=["we continue."]),
            ]
        )

        self.assertEqual([len(batch) for batch in batches], [2, 2])

    def test_translate_single_cue_accepts_missing_closing_marker(self) -> None:
        """Verify that single-cue recovery accepts outputs missing the closing tag."""
        runner = DummyRunner(["[[[cue:1]]]\nciao"])
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-1.8B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        translated = translator._translate_single_cue(
            Cue(start_ms=0, end_ms=1_000, text=["hello"])
        )

        self.assertEqual(translated.text, ["ciao"])

    def test_parse_batch_output_rejects_duplicate_cue_ids(self) -> None:
        """Verify batch parsing rejects responses that duplicate cue markers."""
        translator = SubtitleTranslator(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/HY-MT1.5-7B-Q4_K_M.gguf"),
            target_language="it",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        parsed = translator._parse_batch_output(
            "[[[cue:1]]]\nciao\n[[[/cue]]]\n[[[cue:1]]]\naddio\n[[[/cue]]]",
            [
                Cue(start_ms=0, end_ms=1_000, text=["hello"]),
                Cue(start_ms=1_000, end_ms=2_000, text=["goodbye"]),
            ],
        )

        self.assertIsNone(parsed)

    def test_build_model_prompt_uses_system_prompt_for_chat_runners(self) -> None:
        """Verify that chat-capable runners receive instructions via system prompt."""
        runner = DummyRunner(
            ["[[[cue:1]]]\nciao\n[[[/cue]]]"],
            supports_conversation=True,
            supports_single_turn=True,
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

        payload, system_prompt = translator._build_model_prompt(
            [Cue(start_ms=0, end_ms=1_000, text=["hello"])]
        )

        self.assertEqual(payload, "[[[cue:1]]]\nhello\n[[[/cue]]]\n")
        self.assertIsNotNone(system_prompt)
        self.assertIn("Translate the user's subtitle cues into Italian.", system_prompt)
        self.assertIn("Source language: English.", system_prompt)

    def test_build_model_prompt_uses_chinese_instructions_for_chinese_targets(self) -> None:
        """Verify that Chinese source or target languages use the Chinese prompt template."""
        runner = DummyRunner([], supports_conversation=True, supports_single_turn=True)
        translator = SubtitleTranslator(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/HY-MT1.5-7B-Q4_K_M.gguf"),
            target_language="zh",
            source_language="en",
            requested_device="cpu",
            threads=2,
        )

        _payload, system_prompt = translator._build_model_prompt(
            [Cue(start_ms=0, end_ms=1_000, text=["hello"])]
        )

        self.assertIsNotNone(system_prompt)
        self.assertIn("请将用户提供的字幕翻译成Chinese", system_prompt)

    def test_build_model_prompt_inlines_instructions_without_chat_support(self) -> None:
        """Verify that non-chat runners keep instructions inside the prompt payload."""
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

        prompt, system_prompt = translator._build_model_prompt(
            [Cue(start_ms=0, end_ms=1_000, text=["hello"])]
        )

        self.assertIsNone(system_prompt)
        self.assertIn("Translate the following subtitles into Italian.", prompt)
        self.assertIn("Source language: English.", prompt)
        self.assertIn("[[[cue:1]]]", prompt)


if __name__ == "__main__":
    unittest.main()
