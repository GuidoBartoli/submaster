from __future__ import annotations

import re
from pathlib import Path

from .config import (
    TRANSCRIPT_CLEANUP_CONTEXT_SIZE,
    TRANSCRIPT_CLEANUP_FINAL_PASS_MAX_CHARS,
    TRANSCRIPT_CLEANUP_MAX_CHUNK_CHARS,
    TRANSCRIPT_CLEANUP_REPEAT_PENALTY,
    TRANSCRIPT_CLEANUP_SYSTEM_PROMPT,
    TRANSCRIPT_CLEANUP_TEMPERATURE,
    TRANSCRIPT_CLEANUP_TOP_P,
)
from .console import Console


_BLANK_LINE_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")


def _split_sentences(text: str) -> list[str]:
    """Split prose on likely sentence boundaries while preserving punctuation."""
    sentences: list[str] = []
    current: list[str] = []
    normalized = _MULTI_SPACE_RE.sub(" ", text.strip())
    index = 0

    while index < len(normalized):
        character = normalized[index]
        current.append(character)
        index += 1

        if character not in ".!?":
            continue

        while index < len(normalized) and normalized[index] in "\"')]}":
            current.append(normalized[index])
            index += 1

        while index < len(normalized) and normalized[index].isspace():
            index += 1

        sentences.append("".join(current).strip())
        current = []

    trailing = "".join(current).strip()
    if trailing:
        sentences.append(trailing)
    return sentences


class TranscriptCleaner:
    """Clean transcript prose through a local llama.cpp-compatible model."""

    def __init__(
        self,
        console: Console,
        runner,
        model_path: Path,
        requested_device: str = "auto",
        threads: int = 4,
        context_size: int = TRANSCRIPT_CLEANUP_CONTEXT_SIZE,
        max_chunk_chars: int = TRANSCRIPT_CLEANUP_MAX_CHUNK_CHARS,
        final_pass_max_chars: int = TRANSCRIPT_CLEANUP_FINAL_PASS_MAX_CHARS,
        temperature: float = TRANSCRIPT_CLEANUP_TEMPERATURE,
        top_p: float = TRANSCRIPT_CLEANUP_TOP_P,
        repeat_penalty: float = TRANSCRIPT_CLEANUP_REPEAT_PENALTY,
    ) -> None:
        """Configure transcript cleanup for one CLI invocation.

        :param console: Console used for progress and status output.
        :type console: Console
        :param runner: Runner object exposing `run_prompt` for llama.cpp inference.
        :type runner: object
        :param model_path: Cleanup model path used by the runner.
        :type model_path: pathlib.Path
        :param requested_device: Preferred runtime device selection.
        :type requested_device: str
        :param threads: Number of CPU threads to pass to the runner.
        :type threads: int
        :param context_size: Context window used for cleanup prompts.
        :type context_size: int
        :param max_chunk_chars: Maximum chunk size before another pass is required.
        :type max_chunk_chars: int
        :param final_pass_max_chars: Maximum merged size eligible for a final cleanup pass.
        :type final_pass_max_chars: int
        :param temperature: Sampling temperature used for cleanup prompts.
        :type temperature: float
        :param top_p: Top-p sampling parameter used for cleanup prompts.
        :type top_p: float
        :param repeat_penalty: Repeat penalty used for cleanup prompts.
        :type repeat_penalty: float
        """
        self.console = console
        self.runner = runner
        self.model_path = model_path
        self.requested_device = requested_device
        self.threads = max(1, threads)
        self.context_size = max(4_096, context_size)
        self.max_chunk_chars = max(1, max_chunk_chars)
        self.final_pass_max_chars = max(self.max_chunk_chars, final_pass_max_chars)
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

    def clean_text(self, raw_text: str) -> str:
        """Clean a plain-text transcript and return polished prose."""
        normalized_input = self._normalize_input(raw_text)
        if not normalized_input:
            return ""

        chunks = self._chunk_text(normalized_input)
        chunk_label = "chunk" if len(chunks) == 1 else "chunks"
        self.console.note(
            f"Prepared {len(chunks)} transcript cleanup {chunk_label}; progress updates after each chunk finishes."
        )
        progress = self.console.progress("llama", total=len(chunks), unit=" chunks")
        cleaned_chunks: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            cleaned_chunks.append(self._cleanup_pass(chunk, show_spinner=False))
            progress.update(index)

        progress.finish(len(chunks))
        merged_text = self._merge_chunks(cleaned_chunks)

        if len(cleaned_chunks) > 1 and len(merged_text) <= self.final_pass_max_chars:
            self.console.note("Running final transcript cleanup pass.")
            merged_text = self._cleanup_pass(merged_text, show_spinner=True)

        return self._finalize_output(merged_text)

    def _cleanup_pass(self, text: str, *, show_spinner: bool) -> str:
        """Run one cleanup pass against the local language model."""
        prompt, system_prompt = self._build_prompt(text)
        return self.runner.run_prompt(
            model_path=self.model_path,
            prompt=prompt,
            requested_device=self.requested_device,
            threads=self.threads,
            system_prompt=system_prompt,
            show_spinner=show_spinner,
            context_size=self.context_size,
            temperature=self.temperature,
            top_k=None,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            max_tokens=self._estimate_max_tokens(text),
            disable_thinking=True,
            spinner_label="Running transcript cleanup pass.",
        )

    def _build_prompt(self, text: str) -> tuple[str, str | None]:
        """Build the effective prompt payload and optional system prompt."""
        payload = f"Clean this transcript:\n\n{text.strip()}"
        supports_chat_turn = bool(
            getattr(self.runner, "supports_conversation", False)
            and getattr(self.runner, "supports_single_turn", False)
        )
        if supports_chat_turn:
            return payload, TRANSCRIPT_CLEANUP_SYSTEM_PROMPT
        return f"{TRANSCRIPT_CLEANUP_SYSTEM_PROMPT}\n\n{payload}", None

    def _estimate_max_tokens(self, text: str) -> int:
        """Estimate an output budget that leaves room for prompt tokens."""
        estimated = int(len(text) / 2) + 512
        return max(512, min(self.context_size - 1_024, estimated))

    def _normalize_input(self, text: str) -> str:
        """Normalize transcript whitespace before chunking."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = "\n".join(line.strip() for line in normalized.split("\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = _MULTI_SPACE_RE.sub(" ", normalized)
        return normalized.strip()

    def _merge_chunks(self, chunks: list[str]) -> str:
        """Merge cleaned chunks into one transcript body."""
        normalized_chunks = [self._finalize_output(chunk).strip() for chunk in chunks if chunk.strip()]
        return "\n\n".join(normalized_chunks).strip()

    def _finalize_output(self, text: str) -> str:
        """Normalize cleaned model output into the final transcript format."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return f"{normalized}\n" if normalized else ""

    def _chunk_text(self, text: str) -> list[str]:
        """Split transcript text into model-friendly chunks."""
        return self._chunk_with_strategy(text, strategy_index=0)

    def _chunk_with_strategy(self, text: str, strategy_index: int) -> list[str]:
        """Split text recursively while preferring higher-level boundaries first."""
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.max_chunk_chars:
            return [normalized]

        splitters = (
            (self._split_paragraphs, "\n\n"),
            (self._split_lines, "\n"),
            (_split_sentences, " "),
            (self._split_words, " "),
        )
        if strategy_index >= len(splitters):
            return [normalized]

        splitter, joiner = splitters[strategy_index]
        units = splitter(normalized)
        if len(units) <= 1:
            return self._chunk_with_strategy(normalized, strategy_index + 1)

        chunks: list[str] = []
        current_units: list[str] = []
        current_length = 0

        for unit in units:
            stripped_unit = unit.strip()
            if not stripped_unit:
                continue
            if len(stripped_unit) > self.max_chunk_chars:
                if current_units:
                    chunks.append(joiner.join(current_units).strip())
                    current_units = []
                    current_length = 0
                chunks.extend(self._chunk_with_strategy(stripped_unit, strategy_index + 1))
                continue

            projected_length = len(stripped_unit)
            if current_units:
                projected_length += current_length + len(joiner)

            if current_units and projected_length > self.max_chunk_chars:
                chunks.append(joiner.join(current_units).strip())
                current_units = [stripped_unit]
                current_length = len(stripped_unit)
                continue

            current_units.append(stripped_unit)
            if len(current_units) == 1:
                current_length = len(stripped_unit)
            else:
                current_length += len(joiner) + len(stripped_unit)

        if current_units:
            chunks.append(joiner.join(current_units).strip())
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split transcript text on blank-line paragraph separators."""
        return [part.strip() for part in _BLANK_LINE_RE.split(text) if part.strip()]

    def _split_lines(self, text: str) -> list[str]:
        """Split transcript text on individual non-empty lines."""
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _split_words(self, text: str) -> list[str]:
        """Split transcript text on whitespace as a last resort."""
        return text.split()
