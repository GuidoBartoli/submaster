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


@dataclass(frozen=True)
class TranslationLanguage:
    code: str
    name: str


def resolve_translation_language(raw_language: str) -> TranslationLanguage:
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
    if language is None:
        return False
    normalized = language.code.lower()
    return normalized in _CHINESE_CODES or language.name.lower().endswith("chinese")


class SubtitleTranslator:
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
        self.console = console
        self.runner = runner
        self.model_path = model_path
        self.target_language = resolve_translation_language(target_language)
        self.source_language = (
            None if source_language.strip().lower() == "auto" else resolve_translation_language(source_language)
        )
        self.requested_device = requested_device
        self.threads = threads
        self.max_batch_cues = max(1, max_batch_cues)
        self.max_batch_chars = max(256, max_batch_chars)

    def translate_srt(self, raw_srt: str) -> str:
        cues = parse_srt(raw_srt)
        translated_cues = self.translate_cues(cues)
        return render_srt(translated_cues)

    def translate_cues(self, cues: list[Cue]) -> list[Cue]:
        if not cues:
            raise SubmasterError("No subtitle cues were provided for translation.")

        self.console.note(
            f"Translating subtitles to {self.target_language.name} with Tencent HY-MT."
        )
        progress = self.console.progress("translate", total=len(cues), unit=" cues")
        translated: list[Cue] = []

        for batch in self._batch_cues(cues):
            translated_batch = self._translate_batch(batch)
            translated.extend(translated_batch)
            progress.update(len(translated))

        progress.finish(len(translated))
        return translated

    def _batch_cues(self, cues: list[Cue]) -> list[list[Cue]]:
        batches: list[list[Cue]] = []
        current_batch: list[Cue] = []
        current_chars = 0

        for cue in cues:
            cue_chars = len("\n".join(cue.text))
            if current_batch and (
                len(current_batch) >= self.max_batch_cues
                or current_chars + cue_chars > self.max_batch_chars
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(cue)
            current_chars += cue_chars

        if current_batch:
            batches.append(current_batch)
        return batches

    def _translate_batch(self, batch: list[Cue]) -> list[Cue]:
        prompt = self._build_batch_prompt(batch)
        translated_text = self.runner.run_prompt(
            model_path=self.model_path,
            prompt=prompt,
            requested_device=self.requested_device,
            threads=self.threads,
        )
        parsed_text_by_id = self._parse_batch_output(translated_text, batch)
        if parsed_text_by_id is None:
            self.console.warn(
                "The translation batch response could not be parsed cleanly. "
                "Retrying the current batch one cue at a time."
            )
            return [self._translate_single_cue(cue) for cue in batch]

        translated_batch: list[Cue] = []
        for index, cue in enumerate(batch, start=1):
            translated_lines = self._split_translated_lines(parsed_text_by_id[index])
            translated_batch.append(
                Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=translated_lines)
            )
        return translated_batch

    def _translate_single_cue(self, cue: Cue) -> Cue:
        prompt = self._build_batch_prompt([cue])
        translated_text = self.runner.run_prompt(
            model_path=self.model_path,
            prompt=prompt,
            requested_device=self.requested_device,
            threads=self.threads,
        )
        parsed_text_by_id = self._parse_batch_output(translated_text, [cue])
        if parsed_text_by_id is None:
            raise SubmasterError(
                "llama.cpp returned an invalid single-cue translation response. "
                "Try the larger translation model or shorten the subtitle batch."
            )

        translated_lines = self._split_translated_lines(parsed_text_by_id[1])
        return Cue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=translated_lines)

    def _build_batch_prompt(self, batch: list[Cue]) -> str:
        target_name = self.target_language.name
        payload_blocks: list[str] = []
        for index, cue in enumerate(batch, start=1):
            cue_text = "\n".join(line.strip() for line in cue.text if line.strip())
            payload_blocks.append(f"[[[cue:{index}]]]\n{cue_text}\n[[[/cue]]]")
        payload = "\n\n".join(payload_blocks)

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

        return f"{instructions}\n{payload}\n"

    def _parse_batch_output(self, raw_output: str, batch: list[Cue]) -> dict[int, str] | None:
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
        lines = [line.rstrip() for line in raw_text.replace("\r\n", "\n").split("\n")]
        cleaned = [line for line in lines if line.strip()]
        return cleaned or [raw_text.strip() or ""]
