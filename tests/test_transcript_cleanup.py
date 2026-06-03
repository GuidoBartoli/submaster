import unittest
from pathlib import Path
from types import SimpleNamespace

from submaster.config import TRANSCRIPT_CLEANUP_SYSTEM_PROMPT
from submaster.transcript_cleanup import TranscriptCleaner


class DummyRunner:
    """Minimal runner stub that returns scripted cleanup responses."""

    def __init__(self, responses: list[str]) -> None:
        """Store queued cleanup responses and capture call arguments."""
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self.supports_conversation = True
        self.supports_single_turn = True

    def run_prompt(
        self,
        model_path: Path,
        prompt: str,
        requested_device: str,
        threads: int,
        system_prompt: str | None = None,
        show_spinner: bool = True,
        context_size: int = 0,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float = 0.0,
        repeat_penalty: float = 0.0,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
        spinner_label: str = "",
    ) -> str:
        """Return the next scripted cleanup response and remember the prompt."""
        self.calls.append(
            {
                "model_path": model_path,
                "prompt": prompt,
                "requested_device": requested_device,
                "threads": threads,
                "system_prompt": system_prompt,
                "show_spinner": show_spinner,
                "context_size": context_size,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "max_tokens": max_tokens,
                "disable_thinking": disable_thinking,
                "spinner_label": spinner_label,
            }
        )
        if not self.responses:
            raise AssertionError("No dummy cleanup responses left.")
        return self.responses.pop(0)


class TranscriptCleanupTests(unittest.TestCase):
    """Exercise transcript chunking and cleanup orchestration."""

    def _console(self) -> SimpleNamespace:
        """Create a console stub with no-op progress reporting."""
        class DummyProgress:
            def update(self, _completed: float, extra: str = "") -> None:
                return None

            def finish(self, _completed: float | None = None, extra: str = "") -> None:
                return None

        return SimpleNamespace(
            note=lambda message: None,
            info=lambda message: None,
            progress=lambda label, total, unit="": DummyProgress(),
        )

    def test_chunk_text_prefers_sentence_boundaries(self) -> None:
        """Verify that chunking keeps complete sentence units whenever possible."""
        cleaner = TranscriptCleaner(
            console=self._console(),
            runner=DummyRunner([]),
            model_path=Path("/tmp/Qwen3.5-9B-Q4_K_M.gguf"),
            max_chunk_chars=18,
        )

        chunks = cleaner._chunk_text("One two.\nThree four?\nFive six!")

        self.assertEqual(chunks, ["One two.", "Three four?", "Five six!"])
        self.assertTrue(all(len(chunk) <= 18 for chunk in chunks))

    def test_clean_text_returns_empty_without_calling_model_for_blank_input(self) -> None:
        """Verify blank transcripts do not trigger cleanup model calls."""
        runner = DummyRunner([])
        cleaner = TranscriptCleaner(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/Qwen3.5-9B-Q4_K_M.gguf"),
        )

        self.assertEqual(cleaner.clean_text(" \n\n "), "")
        self.assertEqual(runner.calls, [])

    def test_build_prompt_inlines_system_prompt_for_non_chat_runners(self) -> None:
        """Verify legacy llama.cpp modes receive cleanup instructions in the prompt."""
        runner = DummyRunner([])
        runner.supports_conversation = False
        runner.supports_single_turn = False
        cleaner = TranscriptCleaner(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/Qwen3.5-9B-Q4_K_M.gguf"),
        )

        prompt, system_prompt = cleaner._build_prompt("hello there")

        self.assertIsNone(system_prompt)
        self.assertIn(TRANSCRIPT_CLEANUP_SYSTEM_PROMPT, prompt)
        self.assertIn("Clean this transcript:\n\nhello there", prompt)

    def test_clean_text_runs_chunk_passes_and_final_pass(self) -> None:
        """Verify that multi-chunk cleanup performs an additional merged pass when possible."""
        runner = DummyRunner(
            [
                "Alpha beta.",
                "Gamma delta.",
                "Epsilon zeta.",
                "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta.",
            ]
        )
        cleaner = TranscriptCleaner(
            console=self._console(),
            runner=runner,
            model_path=Path("/tmp/Qwen3.5-9B-Q4_K_M.gguf"),
            requested_device="cpu",
            threads=2,
            max_chunk_chars=18,
            final_pass_max_chars=128,
        )

        cleaned = cleaner.clean_text("alpha beta.\ngamma delta.\nepsilon zeta.")

        self.assertEqual(cleaned, "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta.\n")
        self.assertEqual(len(runner.calls), 4)
        self.assertEqual(runner.calls[0]["system_prompt"], TRANSCRIPT_CLEANUP_SYSTEM_PROMPT)
        self.assertEqual(runner.calls[0]["prompt"], "Clean this transcript:\n\nalpha beta.")
        self.assertFalse(runner.calls[0]["show_spinner"])
        self.assertTrue(runner.calls[-1]["show_spinner"])
        self.assertEqual(runner.calls[-1]["spinner_label"], "Running transcript cleanup pass.")
        self.assertTrue(all(call["disable_thinking"] for call in runner.calls))
        self.assertTrue(all(call["top_k"] is None for call in runner.calls))


if __name__ == "__main__":
    unittest.main()
