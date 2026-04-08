from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import LLAMA_MAX_BATCH_CHARS, LLAMA_MAX_BATCH_CUES, TRANSLATION_LANGUAGES
from .console import Console
from .errors import SubmasterError
from .srt import Cue, parse_srt, render_srt


_CHINESE_CODES = {"zh", "zh-cn", "zh-hans", "zh-hant", "yue"}
_BATCH_CUE_RE = re.compile(
    r"\[\[\[cue:(?P<id>\d+)\]\]\]\s*(?P<text>.*?)\s*\[\[\[/cue\]\]\]",
    re.DOTALL,
)
_SINGLE_CUE_START_RE = re.compile(r"\[\[\[cue:(?P<id>\d+)\]\]\]\s*", re.DOTALL)
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]+[\"')\\]»”’]*$")
_CLAUSE_END_RE = re.compile(r"[,;:，；、]+[\"')\\]»”’]*$")
_CONTINUATION_START_RE = re.compile(r"^[,;:，；、.!?。！？)\]»”’]")
_LOWERCASE_START_RE = re.compile(r"^[a-zà-öø-ÿ]")
_SOFT_BATCH_CHAR_RATIO = 0.7
_STRONG_GAP_MS = 700
_SOFT_GAP_MS = 350


@dataclass(frozen=True)
class TranslationLanguage:
    """Represent a normalized translation language target or source.

    :param code: Language code passed to prompts and internal checks.
    :type code: str
    :param name: Human-readable language label used in logs and prompts.
    :type name: str
    """

    code: str
    name: str


def resolve_translation_language(raw_language: str) -> TranslationLanguage:
    """Normalize a user-provided translation language value.

    :param raw_language: Raw language code or name supplied by the user.
    :type raw_language: str
    :returns: Normalized translation language metadata.
    :rtype: TranslationLanguage
    :raises SubmasterError: If the input is empty after normalization.
    """
    # Accept either explicit language codes or common English aliases.
    normalized = raw_language.strip().lower()
    if not normalized:
        raise SubmasterError("Translation target language cannot be empty.")

    aliases = {
        "arabic": "ar",
        "bengali": "bn",
        "burmese": "my",
        "cantonese": "yue",
        "chinese": "zh",
        "czech": "cs",
        "dutch": "nl",
        "english": "en",
        "filipino": "tl",
        "french": "fr",
        "german": "de",
        "gujarati": "gu",
        "hebrew": "he",
        "hindi": "hi",
        "indonesian": "id",
        "italian": "it",
        "japanese": "ja",
        "kazakh": "kk",
        "khmer": "km",
        "korean": "ko",
        "malay": "ms",
        "marathi": "mr",
        "mongolian": "mn",
        "persian": "fa",
        "polish": "pl",
        "portuguese": "pt",
        "russian": "ru",
        "spanish": "es",
        "tamil": "ta",
        "telugu": "te",
        "thai": "th",
        "traditional chinese": "zh-hant",
        "turkish": "tr",
        "tibetan": "bo",
        "ukrainian": "uk",
        "urdu": "ur",
        "uyghur": "ug",
        "vietnamese": "vi",
    }

    language_code = aliases.get(normalized, normalized)
    display_name = TRANSLATION_LANGUAGES.get(language_code)
    if display_name:
        return TranslationLanguage(code=language_code, name=display_name)

    # Allow advanced users to pass a free-form language name even if it is not
    # in the published HY-MT support list. The model may still handle it.
    return TranslationLanguage(code=language_code, name=raw_language.strip())


def _looks_like_chinese(language: TranslationLanguage | None) -> bool:
    """Check whether a language should use the Chinese prompt template.

    :param language: Language metadata to inspect.
    :type language: TranslationLanguage | None
    :returns: `True` when Chinese-specific prompting should be used.
    :rtype: bool
    """
    if language is None:
        return False
    normalized = language.code.lower()
    return normalized in _CHINESE_CODES or language.name.lower().endswith("chinese")


class SubtitleTranslator:
    """Translate subtitle cues through a llama.cpp-compatible runner."""

    def __init__(
        self,
        console: Console,
        runner,
        model_path: Path,
        target_language: str,
        source_language: str = "auto",
        requested_device: str = "auto",
        threads: int = 4,
        max_batch_cues: int = LLAMA_MAX_BATCH_CUES,
        max_batch_chars: int = LLAMA_MAX_BATCH_CHARS,
    ) -> None:
        """Configure the subtitle translator for one target language.

        :param console: Console used for progress and warning output.
        :type console: Console
        :param runner: Runner object exposing `run_prompt` for llama.cpp inference.
        :type runner: object
        :param model_path: Translation model path used by the runner.
        :type model_path: pathlib.Path
        :param target_language: Requested translation target code or name.
        :type target_language: str
        :param source_language: Optional source language hint, or `auto`.
        :type source_language: str
        :param requested_device: Preferred runtime device selection.
        :type requested_device: str
        :param threads: Number of CPU threads to pass to the runner.
        :type threads: int
        :param max_batch_cues: Maximum subtitle cues per translation batch.
        :type max_batch_cues: int
        :param max_batch_chars: Maximum approximate characters per translation batch.
        :type max_batch_chars: int
        """
        self.console = console
        self.runner = runner
        self.model_path = model_path
        self.target_language = resolve_translation_language(target_language)
        self.source_language = (
            None if source_language.strip().lower() == "auto" else resolve_translation_language(source_language)
        )
        self.requested_device = requested_device
        self.threads = threads
        default_batch_cap = max_batch_cues == LLAMA_MAX_BATCH_CUES
        if default_batch_cap and "1.8b" in model_path.name.lower():
            max_batch_cues = min(max_batch_cues, 2)
        self.max_batch_cues = max(1, max_batch_cues)
        self.max_batch_chars = max(256, max_batch_chars)
        self._fallback_warning_emitted = False

    def translate_srt(self, raw_srt: str) -> str:
        """Translate an entire SRT document and return normalized SRT text.

        :param raw_srt: Raw subtitle text to translate.
        :type raw_srt: str
        :returns: Translated subtitle text in canonical SRT format.
        :rtype: str
        :raises SubmasterError: If the SRT cannot be parsed or translation fails.
        """
        cues = parse_srt(raw_srt)
        translated_cues = self.translate_cues(cues)
        return render_srt(translated_cues)

    def translate_cues(self, cues: list[Cue]) -> list[Cue]:
        """Translate a list of subtitle cues while preserving cue timings.

        :param cues: Subtitle cues to translate.
        :type cues: list[Cue]
        :returns: Translated cues with original timing metadata.
        :rtype: list[Cue]
        :raises SubmasterError: If no cues are provided or translation fails.
        """
        if not cues:
            raise SubmasterError("No subtitle cues were provided for translation.")

        # Report translation progress at cue granularity even though work runs in batches.
        self.console.info(
            f"Translating subtitles to {self.target_language.name} with Tencent HY-MT."
        )
        batches = self._batch_cues(cues)
        batch_label = "batch" if len(batches) == 1 else "batches"
        self.console.info(
            f"Prepared {len(batches)} translation {batch_label}; progress updates after each batch finishes."
        )
        progress = self.console.progress("llama", total=len(cues), unit=" cues")
        translated: list[Cue] = []

        for batch in batches:
            translated_batch = self._translate_batch(batch)
            translated.extend(translated_batch)
            progress.update(len(translated))

        progress.finish(len(translated))
        return translated

    def _batch_cues(self, cues: list[Cue]) -> list[list[Cue]]:
        """Split subtitle cues into model-friendly translation batches.

        :param cues: Subtitle cues to batch.
        :type cues: list[Cue]
        :returns: Ordered cue batches constrained by size limits.
        :rtype: list[list[Cue]]
        """
        batches: list[list[Cue]] = []
        start_index = 0

        while start_index < len(cues):
            end_index = self._choose_batch_end(cues, start_index)
            batches.append(cues[start_index : end_index + 1])
            start_index = end_index + 1
        return batches

    def _choose_batch_end(self, cues: list[Cue], start_index: int) -> int:
        """Choose the best batch end from one start index within hard limits."""
        best_end = start_index
        best_score = float("-inf")
        batch_chars = 0

        for end_index in range(start_index, min(len(cues), start_index + self.max_batch_cues)):
            cue_chars = len(self._cue_text(cues[end_index]))
            if end_index > start_index and batch_chars + cue_chars > self.max_batch_chars:
                break

            batch_chars += cue_chars
            score = self._score_batch_end(cues, start_index, end_index, batch_chars)
            if score > best_score:
                best_score = score
                best_end = end_index

        return best_end

    def _score_batch_end(
        self,
        cues: list[Cue],
        start_index: int,
        end_index: int,
        batch_chars: int,
    ) -> float:
        """Score one candidate batch end, preferring sentence-complete boundaries."""
        score = float(batch_chars)
        batch_size = end_index - start_index + 1
        current_text = self._cue_text(cues[end_index])
        next_text = self._cue_text(cues[end_index + 1]) if end_index + 1 < len(cues) else ""
        gap_after_ms = self._gap_after_ms(cues, end_index)
        soft_char_target = max(1, int(self.max_batch_chars * _SOFT_BATCH_CHAR_RATIO))

        score += batch_size * 120
        if batch_chars >= soft_char_target:
            score += 200

        if self._ends_sentence(current_text):
            score += 3_000
        elif self._ends_clause(current_text):
            score += 1_200
        elif end_index + 1 < len(cues):
            score -= 900

        if gap_after_ms >= _STRONG_GAP_MS:
            score += 2_000
        elif gap_after_ms >= _SOFT_GAP_MS:
            score += 800

        if next_text:
            if self._starts_continuation(next_text):
                score -= 2_500
            elif self._starts_new_sentence(next_text):
                score += 500

        if batch_size == 1 and end_index + 1 < len(cues) and batch_chars < soft_char_target:
            score -= 700
        return score

    def _cue_text(self, cue: Cue) -> str:
        """Normalize cue text for batching heuristics."""
        return " ".join(line.strip() for line in cue.text if line.strip()).strip()

    def _gap_after_ms(self, cues: list[Cue], index: int) -> int:
        """Return the source timing gap after one cue, or zero at the end."""
        if index + 1 >= len(cues):
            return 0
        return max(0, cues[index + 1].start_ms - cues[index].end_ms)

    def _ends_sentence(self, text: str) -> bool:
        """Check whether text ends with sentence-closing punctuation."""
        return bool(_SENTENCE_END_RE.search(text.strip()))

    def _ends_clause(self, text: str) -> bool:
        """Check whether text ends with softer clause punctuation."""
        return bool(_CLAUSE_END_RE.search(text.strip()))

    def _starts_continuation(self, text: str) -> bool:
        """Check whether the next cue likely continues the same sentence."""
        stripped = text.strip()
        if not stripped:
            return False
        return bool(
            _CONTINUATION_START_RE.match(stripped)
            or _LOWERCASE_START_RE.match(stripped)
        )

    def _starts_new_sentence(self, text: str) -> bool:
        """Check whether the next cue looks like a fresh sentence start."""
        stripped = text.strip()
        if not stripped:
            return False
        if self._starts_continuation(stripped):
            return False
        first_character = stripped[0]
        return first_character.isupper() or first_character.isdigit()

    def _translate_batch(self, batch: list[Cue]) -> list[Cue]:
        """Translate one cue batch and preserve cue order.

        :param batch: Cue batch to translate in a single prompt.
        :type batch: list[Cue]
        :returns: Translated cues for the batch.
        :rtype: list[Cue]
        :raises SubmasterError: If fallback single-cue translation also fails.
        """
        # Try the efficient batched prompt first and only fall back when cue markup is corrupted.
        prompt, system_prompt = self._build_model_prompt(batch)
        translated_text = self.runner.run_prompt(
            model_path=self.model_path,
            prompt=prompt,
            requested_device=self.requested_device,
            threads=self.threads,
            system_prompt=system_prompt,
            show_spinner=False,
        )
        parsed_text_by_id = self._parse_batch_output(translated_text, batch)
        if parsed_text_by_id is None:
            if not self._fallback_warning_emitted:
                self.console.warn(
                    "Some translation batch responses could not be parsed cleanly. "
                    "Retrying failed batches one cue at a time."
                )
                self._fallback_warning_emitted = True
            return [self._translate_single_cue(cue) for cue in batch]

        # Rebuild the batch cue-by-cue so timings stay attached to the original entries.
        translated_batch: list[Cue] = []
        for index, cue in enumerate(batch, start=1):
            translated_lines = self._split_translated_lines(parsed_text_by_id[index])
            translated_batch.append(
                Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=translated_lines)
            )
        return translated_batch

    def _translate_single_cue(self, cue: Cue) -> Cue:
        """Translate a single cue as a recovery path for failed batch parsing.

        :param cue: Cue to translate.
        :type cue: Cue
        :returns: Translated cue preserving the original timing.
        :rtype: Cue
        :raises SubmasterError: If the single-cue response still cannot be parsed.
        """
        prompt, system_prompt = self._build_model_prompt([cue])
        translated_text = self.runner.run_prompt(
            model_path=self.model_path,
            prompt=prompt,
            requested_device=self.requested_device,
            threads=self.threads,
            system_prompt=system_prompt,
            show_spinner=False,
        )
        parsed_text_by_id = self._parse_batch_output(translated_text, [cue])
        if parsed_text_by_id is None:
            permissive_text = self._parse_permissive_single_output(translated_text)
            if permissive_text is None:
                raise SubmasterError(
                    "llama.cpp returned an invalid single-cue translation response. "
                    "Try the larger translation model or shorten the subtitle batch."
                )
            parsed_text_by_id = {1: permissive_text}

        translated_lines = self._split_translated_lines(parsed_text_by_id[1])
        return Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=translated_lines)

    def _build_batch_prompt(self, batch: list[Cue]) -> str:
        """Build the cue payload for one translation batch.

        :param batch: Cue batch to encode into the prompt.
        :type batch: list[Cue]
        :returns: Cue-tagged payload without model instructions.
        :rtype: str
        """
        payload_blocks: list[str] = []

        # Wrap each cue in explicit tags so the response can be parsed deterministically.
        for index, cue in enumerate(batch, start=1):
            cue_text = "\n".join(line.strip() for line in cue.text if line.strip())
            payload_blocks.append(f"[[[cue:{index}]]]\n{cue_text}\n[[[/cue]]]")
        return "\n\n".join(payload_blocks) + "\n"

    def _build_system_prompt(self) -> str | None:
        """Build a translation system prompt when the runner supports chat turns.

        :returns: System prompt text used to steer translation output.
        :rtype: str | None
        """
        supports_chat_turn = bool(
            getattr(self.runner, "supports_conversation", False)
            and getattr(self.runner, "supports_single_turn", False)
        )
        if not supports_chat_turn:
            return None

        target_name = self.target_language.name
        if _looks_like_chinese(self.target_language) or _looks_like_chinese(self.source_language):
            instructions = [f"请将用户提供的字幕翻译成{target_name}。"]
            instructions.append("严格保留每个[[[cue:N]]]与[[[/cue]]]标记，不要新增说明。")
            instructions.append("不要合并、删除或重排任何字幕块。")
            instructions.append("只输出翻译后的字幕块，并保持原有顺序。")
        else:
            instructions = [f"Translate the user's subtitle cues into {target_name}."]
            instructions.append("Preserve every [[[cue:N]]] and [[[/cue]]] marker exactly.")
            instructions.append("Do not add explanations.")
            instructions.append("Do not merge, remove, or reorder any cue.")
            instructions.append("Output only the translated cues in the same order.")

        if self.source_language is not None:
            instructions.append(f"Source language: {self.source_language.name}.")
        return " ".join(instructions)

    def _build_model_prompt(self, batch: list[Cue]) -> tuple[str, str | None]:
        """Build the effective prompt payload and optional system prompt.

        :param batch: Cue batch to encode for the model.
        :type batch: list[Cue]
        :returns: Prompt payload plus optional system prompt.
        :rtype: tuple[str, str | None]
        """
        payload = self._build_batch_prompt(batch)
        system_prompt = self._build_system_prompt()
        if system_prompt is not None:
            return payload, system_prompt
        return f"{self._build_legacy_instructions()}\n{payload}", None

    def _build_legacy_instructions(self) -> str:
        """Build the inline instruction header for non-chat llama.cpp modes.

        :returns: Prompt instructions prepended to the batch payload.
        :rtype: str
        """
        target_name = self.target_language.name
        if _looks_like_chinese(self.target_language) or _looks_like_chinese(self.source_language):
            instructions = (
                f"请将以下字幕翻译成{target_name}。\n"
                "严格保留每个[[[cue:N]]]与[[[/cue]]]标记，不要新增说明，不要合并或删除任何字幕块。\n"
                "只输出翻译后的内容，并保持原有顺序。\n"
            )
        else:
            instructions = (
                f"Translate the following subtitles into {target_name}.\n"
                "Preserve every [[[cue:N]]] and [[[/cue]]] marker exactly.\n"
                "Do not add explanations.\n"
                "Do not merge, remove, or reorder any cue.\n"
                "Output only the translated cues.\n"
            )

        if self.source_language is not None:
            instructions += f"Source language: {self.source_language.name}.\n"
        return instructions

    def _parse_batch_output(self, raw_output: str, batch: list[Cue]) -> dict[int, str] | None:
        """Parse a tagged batch response into cue-id keyed text.

        :param raw_output: Raw translation model output.
        :type raw_output: str
        :param batch: Original cue batch used to validate the response shape.
        :type batch: list[Cue]
        :returns: Mapping from cue id to translated text, or `None` when invalid.
        :rtype: dict[int, str] | None
        """
        # Reject responses that lose tags, duplicate ids, or omit expected cues.
        matches = list(_BATCH_CUE_RE.finditer(raw_output))
        if len(matches) != len(batch):
            return None

        translated_by_id: dict[int, str] = {}
        for match in matches:
            cue_id = int(match.group("id"))
            translated_text = match.group("text").strip()
            if cue_id in translated_by_id:
                return None
            translated_by_id[cue_id] = translated_text

        expected_ids = set(range(1, len(batch) + 1))
        if set(translated_by_id) != expected_ids:
            return None
        return translated_by_id

    def _split_translated_lines(self, raw_text: str) -> list[str]:
        """Normalize translated cue text into subtitle line fragments.

        :param raw_text: Raw translated cue body.
        :type raw_text: str
        :returns: Cleaned subtitle lines with empty lines removed.
        :rtype: list[str]
        """
        # Preserve at least one line even when the model returns only whitespace.
        lines = [line.rstrip() for line in raw_text.replace("\r\n", "\n").split("\n")]
        cleaned = [line for line in lines if line.strip()]
        return cleaned or [raw_text.strip() or ""]

    def _parse_permissive_single_output(self, raw_output: str) -> str | None:
        """Recover a single-cue translation when the closing tag is missing.

        :param raw_output: Raw model output for a single subtitle cue.
        :type raw_output: str
        :returns: Recovered translated text, or `None` when recovery is unsafe.
        :rtype: str | None
        """
        normalized = raw_output.strip()
        if not normalized:
            return None

        match = _SINGLE_CUE_START_RE.search(normalized)
        if match:
            if int(match.group("id")) != 1:
                return None
            normalized = normalized[match.end():].strip()

        if "[[[cue:" in normalized or "[[[/cue]]]" in normalized:
            return None
        return normalized or None
