from __future__ import annotations

import re

from app.tools.registry import ToolRegistry
from app.tools.types import ToolCall

_TIME_RE = re.compile(
    r"\b(?:what(?:'s| is)?(?: the)? time(?: is it| it is)?|what time(?: is it| it is)?|current time|time is it)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:what(?:'s| is)?(?: the)? date(?: today)?|what date(?: is it| today)?|today'?s date|what day is it|which day is it)\b",
    re.IGNORECASE,
)
_DEVICE_RE = re.compile(
    r"\b(?:audio devices|sound devices|list devices|available microphones|available speakers|microphone list|speaker list)\b",
    re.IGNORECASE,
)
_RUNTIME_RE = re.compile(
    r"\b(?:runtime status|what model|which model|which llm|what are you running|what backend)\b",
    re.IGNORECASE,
)
_CALENDAR_RE = re.compile(
    r"\b(?:calendar|schedule|agenda|appointments|appointment|my events|my event|meetings|meeting)\b",
    re.IGNORECASE,
)
_NEXT_EVENT_RE = re.compile(r"\b(?:next event|what'?s next|up next|next meeting|first meeting)\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_MUTE_RE = re.compile(r"\b(?:mute|silent|silence)\b", re.IGNORECASE)
_MAX_VOLUME_RE = re.compile(r"\b(?:max volume|full volume|volume max)\b", re.IGNORECASE)
_VOLUME_KEYWORD_RE = re.compile(r"\b(?:volume|sound|speaker)\b", re.IGNORECASE)
_VOLUME_UP_RE = re.compile(r"\b(?:volume up|turn it up|turn up the volume|louder|increase volume|raise volume)\b", re.IGNORECASE)
_VOLUME_DOWN_RE = re.compile(r"\b(?:volume down|turn it down|quieter|decrease volume|lower volume)\b", re.IGNORECASE)
_DIGIT_PERCENT_RE = re.compile(r"\b(\d{1,3})\s*(?:percent|%)?\b", re.IGNORECASE)
_NUMBER_TOKENS_RE = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and|-)+\b", re.IGNORECASE)

_UNIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _clamp_percent(value: int) -> int:
    return max(0, min(100, value))


def _parse_number_words(number_text: str) -> int | None:
    tokens = re.findall(r"[a-z]+", number_text.lower())
    if not tokens:
        return None

    current = 0
    found = False
    for token in tokens:
        if token in _UNIT_WORDS:
            current += _UNIT_WORDS[token]
            found = True
            continue
        if token in _TENS_WORDS:
            current += _TENS_WORDS[token]
            found = True
            continue
        if token == "hundred":
            current = max(1, current) * 100
            found = True
            continue
        if token == "and":
            continue
        if found:
            break
        return None

    if not found:
        return None
    return _clamp_percent(current)


def _extract_volume_percent(text: str, current_volume_percent: int | None) -> int | None:
    if _MUTE_RE.search(text):
        return 0
    if _MAX_VOLUME_RE.search(text):
        return 100

    has_volume_language = (
        _VOLUME_KEYWORD_RE.search(text)
        or _VOLUME_UP_RE.search(text)
        or _VOLUME_DOWN_RE.search(text)
    )
    if not has_volume_language:
        return None

    digit_match = _DIGIT_PERCENT_RE.search(text)
    if digit_match:
        return _clamp_percent(int(digit_match.group(1)))

    word_match = _NUMBER_TOKENS_RE.search(text)
    if word_match:
        parsed = _parse_number_words(word_match.group(0))
        if parsed is not None:
            return parsed

    if current_volume_percent is None:
        return None

    if _VOLUME_UP_RE.search(text):
        return _clamp_percent(current_volume_percent + 10)
    if _VOLUME_DOWN_RE.search(text):
        return _clamp_percent(current_volume_percent - 10)
    return None


def route_tool_intent(
    transcript: str,
    registry: ToolRegistry,
    current_volume_percent: int | None = None,
) -> ToolCall | None:
    text = " ".join(transcript.strip().split())
    if not text:
        return None

    if registry.get("get_calendar_events") and _CALENDAR_RE.search(text):
        date_scope = "tomorrow" if _TOMORROW_RE.search(text) else "today"
        mode = "next" if _NEXT_EVENT_RE.search(text) else "summary"
        return ToolCall("get_calendar_events", {"date_scope": date_scope, "mode": mode})

    if registry.get("set_output_volume"):
        volume_percent = _extract_volume_percent(text, current_volume_percent)
        if volume_percent is not None:
            return ToolCall("set_output_volume", {"volume_percent": volume_percent})

    if registry.get("get_time") and _TIME_RE.search(text):
        return ToolCall("get_time", {})

    if registry.get("get_date") and _DATE_RE.search(text):
        return ToolCall("get_date", {})

    if registry.get("list_audio_devices") and _DEVICE_RE.search(text):
        return ToolCall("list_audio_devices", {})

    if registry.get("get_runtime_status") and _RUNTIME_RE.search(text):
        return ToolCall("get_runtime_status", {})

    return None
