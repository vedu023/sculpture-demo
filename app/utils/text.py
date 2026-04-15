from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_MARKER_RE = re.compile(r"^\s*(?:[-*#]+|\d+\.)\s*", re.MULTILINE)
_MARKDOWN_RE = re.compile(r"[`*_#>]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sanitize_spoken_response(text: str, max_sentences: int = 2) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _LEADING_MARKER_RE.sub("", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _MARKDOWN_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return ""

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    if not sentences:
        return cleaned

    return " ".join(sentences[:max_sentences]).strip()


def truncate_for_log(text: str, limit: int = 72) -> str:
    compact = _WHITESPACE_RE.sub(" ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
