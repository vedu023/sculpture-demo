from __future__ import annotations

import re
from datetime import datetime

from app.tools.types import ToolCall, ToolResult


_WS_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")

_AFFIRMATIVE_TOKENS = (
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "confirm",
    "go ahead",
    "do it",
    "please do",
    "haan",
    "haanji",
    "han",
    "ji haan",
    "हाँ",
    "हां",
    "ठीक है",
    "कर दो",
    "हाँ जी",
    "ಹೌದು",
    "ಸರಿ",
    "ಮಾಡು",
    "ಮಾಡಿ",
)
_NEGATIVE_TOKENS = (
    "no",
    "nope",
    "cancel",
    "stop",
    "do not",
    "don't",
    "never mind",
    "nahin",
    "nahi",
    "मत",
    "नहीं",
    "रहने दो",
    "ಬೇಡ",
    "ಇಲ್ಲ",
    "ನಿಲ್ಲು",
)

_LANGUAGE_MODES = {"auto", "english", "indic"}


def script_counts(text: str) -> dict[str, int]:
    normalized = normalize_text(text)
    return {
        "hi": len(_DEVANAGARI_RE.findall(normalized)),
        "kn": len(_KANNADA_RE.findall(normalized)),
    }


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def normalize_language_mode(language_mode: str | None) -> str:
    normalized = (language_mode or "").strip().lower()
    if normalized in _LANGUAGE_MODES:
        return normalized
    return "auto"


def detect_text_language(text: str, default: str = "en") -> str:
    counts = script_counts(text)
    kannada_count = counts["kn"]
    devanagari_count = counts["hi"]
    if not normalize_text(text):
        return default

    if kannada_count > devanagari_count and kannada_count > 0:
        return "kn"
    if devanagari_count > 0:
        return "hi"
    return default


def is_mixed_indic_script(text: str) -> bool:
    counts = script_counts(text)
    return counts["hi"] > 0 and counts["kn"] > 0


def language_instruction(language: str) -> str:
    if language == "hi":
        return "Reply only in natural Hindi written in Devanagari script."
    if language == "kn":
        return "Reply only in natural Kannada script."
    return "Reply only in natural English."


def default_reply_language(
    configured_language: str | None,
    language_mode: str | None = None,
) -> str:
    language = (configured_language or "").strip().lower()
    if language in {"en", "hi", "kn"}:
        return language
    if normalize_language_mode(language_mode) == "indic":
        return "hi"
    return "en"


def is_affirmative(text: str) -> bool:
    normalized = normalize_text(text).lower()
    return any(token in normalized for token in _AFFIRMATIVE_TOKENS)


def is_negative(text: str) -> bool:
    normalized = normalize_text(text).lower()
    return any(token in normalized for token in _NEGATIVE_TOKENS)


def fallback_reply(language: str) -> str:
    if language == "hi":
        return "माफ़ कीजिए, अभी मैं ठीक से जवाब नहीं दे पा रही हूँ।"
    if language == "kn":
        return "ಕ್ಷಮಿಸಿ, ಈಗ ನಾನು ಸರಿಯಾಗಿ ಉತ್ತರಿಸಲು ಆಗುತ್ತಿಲ್ಲ."
    return "I apologize, but I'm having trouble processing that right now."


def greeting_fallback(language: str) -> str:
    if language == "hi":
        return "नमस्ते, शुरू करें?"
    if language == "kn":
        return "ನಮಸ್ಕಾರ, ಶುರು ಮಾಡೋಣ?"
    return "Hey, good to see you."


def confirmation_prompt(tool_call: ToolCall, language: str) -> str:
    if tool_call.tool_name == "set_output_volume":
        volume_percent = tool_call.arguments.get("volume_percent")
        if volume_percent is not None:
            if language == "hi":
                return f"क्या मैं आउटपुट वॉल्यूम {volume_percent} प्रतिशत कर दूँ?"
            if language == "kn":
                return f"ಔಟ್ಪುಟ್ ವಾಲ್ಯೂಮ್ ಅನ್ನು {volume_percent} ಪ್ರತಿಶತಕ್ಕೆ ಬದಲಿಸಲೇ?"
            return f"Do you want me to set the output volume to {volume_percent} percent?"
    if language == "hi":
        return "क्या मैं यह कर दूँ?"
    if language == "kn":
        return "ಇದನ್ನು ಮುಂದುವರಿಸಲೇ?"
    return "Do you want me to go ahead with that?"


def confirmation_cancelled(language: str) -> str:
    if language == "hi":
        return "ठीक है, मैं इसे वैसे ही रहने दूँगी।"
    if language == "kn":
        return "ಸರಿ, ಅದನ್ನು ಹಾಗೆಯೇ ಬಿಡುತ್ತೇನೆ."
    return "Okay, I will leave that unchanged."


def confirmation_retry(language: str) -> str:
    if language == "hi":
        return "जारी रखने के लिए हाँ कहिए, या रद्द करने के लिए नहीं कहिए।"
    if language == "kn":
        return "ಮುಂದುವರಿಸಲು ಹೌದು ಎಂದು ಹೇಳಿ, ರದ್ದು ಮಾಡಲು ಇಲ್ಲ ಎಂದು ಹೇಳಿ."
    return "Please say yes to continue or no to cancel."


def localize_tool_result(tool_result: ToolResult, language: str) -> str:
    if language not in {"hi", "kn"}:
        return tool_result.spoken_response

    if not tool_result.ok:
        return _localized_tool_error(tool_result, language)

    if tool_result.tool_name == "get_time":
        return _time_reply(tool_result.data.get("time", ""), language)
    if tool_result.tool_name == "get_date":
        return _date_reply(tool_result.data.get("date", ""), language)
    if tool_result.tool_name == "list_audio_devices":
        return _audio_devices_reply(tool_result.data, language)
    if tool_result.tool_name == "get_runtime_status":
        return _runtime_status_reply(tool_result.data, language)
    if tool_result.tool_name == "set_output_volume":
        volume_percent = tool_result.data.get("volume_percent", "")
        if language == "hi":
            return f"ठीक है, मैंने आउटपुट वॉल्यूम {volume_percent} प्रतिशत कर दिया है।"
        return f"ಸರಿ, ಔಟ್ಪುಟ್ ವಾಲ್ಯೂಮ್ ಅನ್ನು {volume_percent} ಪ್ರತಿಶತಕ್ಕೆ ಹಾಕಿದ್ದೇನೆ."
    if tool_result.tool_name == "get_calendar_events":
        return _calendar_reply(tool_result.data, language)
    return tool_result.spoken_response


def _localized_tool_error(tool_result: ToolResult, language: str) -> str:
    if tool_result.tool_name == "get_calendar_events":
        if language == "hi":
            return "मैं इस Mac पर कैलेंडर नहीं पढ़ पाई। कैलेंडर अनुमति जाँचकर फिर कोशिश करें।"
        return "ಈ ಮ್ಯಾಕ್‌ನಲ್ಲಿ ಕ್ಯಾಲೆಂಡರ್‌ನ್ನು ಓದಲು ಆಗಲಿಲ್ಲ. ಕ್ಯಾಲೆಂಡರ್ ಅನುಮತಿಯನ್ನು ನೋಡಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
    if language == "hi":
        return "मैंने कोशिश की, लेकिन वह काम अभी नहीं हुआ।"
    return "ಪ್ರಯತ್ನಿಸಿದೆ, ಆದರೆ ಅದು ಈಗ ಕೆಲಸ ಆಗಲಿಲ್ಲ."


def _time_reply(value: str, language: str) -> str:
    if language == "hi":
        return f"अभी {value} बजे हैं।"
    return f"ಈಗ {value} ಆಗಿದೆ."


def _date_reply(value: str, language: str) -> str:
    if language == "hi":
        return f"आज {value} है।"
    return f"ಇಂದು {value}."


def _audio_devices_reply(data: dict[str, object], language: str) -> str:
    devices = list(data.get("devices", []) or [])
    count = int(data.get("count", len(devices)) or 0)
    names = [str(device.get("name", f"Device {index}")) for index, device in enumerate(devices[:3])]
    if not devices:
        if language == "hi":
            return "मुझे कोई ऑडियो डिवाइस नहीं मिला।"
        return "ಯಾವುದೇ ಆಡಿಯೊ ಸಾಧನ ಸಿಗಲಿಲ್ಲ."
    names_text = ", ".join(names)
    if language == "hi":
        if count == 1:
            return f"मुझे 1 ऑडियो डिवाइस मिला: {names_text}।"
        return f"मुझे {count} ऑडियो डिवाइस मिले। पहले कुछ हैं: {names_text}।"
    if count == 1:
        return f"ನನಗೆ 1 ಆಡಿಯೊ ಸಾಧನ ಸಿಕ್ಕಿತು: {names_text}."
    return f"ನನಗೆ {count} ಆಡಿಯೊ ಸಾಧನಗಳು ಸಿಕ್ಕಿವೆ. ಮೊದಲ ಕೆಲವು ಇವು: {names_text}."


def _runtime_status_reply(data: dict[str, object], language: str) -> str:
    assistant_name = str(data.get("assistant_name", "Smruti"))
    llm_model = str(data.get("llm_model", ""))
    asr_model = str(data.get("asr_model", ""))
    tts_backend = str(data.get("tts_backend", ""))
    volume = data.get("output_volume_percent", "")
    if language == "hi":
        return (
            f"आप {assistant_name} से बात कर रहे हैं। मैं भाषा के लिए {llm_model}, "
            f"सुनने के लिए {asr_model}, और आवाज़ के लिए {tts_backend} इस्तेमाल कर रही हूँ। "
            f"आउटपुट वॉल्यूम {volume} प्रतिशत है।"
        )
    return (
        f"ನೀವು {assistant_name} ಜೊತೆ ಮಾತನಾಡುತ್ತಿದ್ದೀರಿ. ನಾನು ಭಾಷೆಗೆ {llm_model}, "
        f"ಕೇಳಲು {asr_model}, ಮತ್ತು ಮಾತಿಗೆ {tts_backend} ಬಳಸುತ್ತಿದ್ದೇನೆ. "
        f"ಔಟ್ಪುಟ್ ವಾಲ್ಯೂಮ್ {volume} ಪ್ರತಿಶತ ಇದೆ."
    )


def _calendar_reply(data: dict[str, object], language: str) -> str:
    events = list(data.get("events", []) or [])
    count = int(data.get("count", len(events)) or 0)
    date_scope = str(data.get("date_scope", "today"))
    mode = str(data.get("mode", "summary"))
    label = _calendar_label(date_scope, language)
    if not events:
        if language == "hi":
            return f"{label} आपके कैलेंडर में कुछ नहीं है।"
        return f"{label} ನಿಮ್ಮ ಕ್ಯಾಲೆಂಡರ್‌ನಲ್ಲಿ ಏನೂ ಇಲ್ಲ."

    if mode == "next":
        event = events[0]
        title = str(event.get("title", "Untitled"))
        start_text = _clock_time(str(event.get("start_text", "")))
        if language == "hi":
            return f"{label} आपका अगला इवेंट {title} {start_text} पर है।"
        return f"{label} ನಿಮ್ಮ ಮುಂದಿನ ಈವೆಂಟ್ {title}, {start_text}ಕ್ಕೆ ಇದೆ."

    if count == 1:
        event = events[0]
        title = str(event.get("title", "Untitled"))
        start_text = _clock_time(str(event.get("start_text", "")))
        if language == "hi":
            return f"{label} आपका 1 इवेंट है: {title}, {start_text} पर।"
        return f"{label} ನಿಮ್ಮ 1 ಈವೆಂಟ್ ಇದೆ: {title}, {start_text}ಕ್ಕೆ."

    first_event = events[0]
    second_event = events[1]
    first_title = str(first_event.get("title", "Untitled"))
    first_time = _clock_time(str(first_event.get("start_text", "")))
    second_title = str(second_event.get("title", "Untitled"))
    second_time = _clock_time(str(second_event.get("start_text", "")))
    if language == "hi":
        return (
            f"{label} आपके {count} इवेंट हैं। पहले {first_title} {first_time} पर, "
            f"फिर {second_title} {second_time} पर।"
        )
    return (
        f"{label} ನಿಮ್ಮ {count} ಈವೆಂಟ್‌ಗಳು ಇವೆ. ಮೊದಲಿಗೆ {first_title} {first_time}ಕ್ಕೆ, "
        f"ನಂತರ {second_title} {second_time}ಕ್ಕೆ."
    )


def _calendar_label(date_scope: str, language: str) -> str:
    if language == "hi":
        return "कल" if date_scope == "tomorrow" else "आज"
    return "ನಾಳೆ" if date_scope == "tomorrow" else "ಇಂದು"


def _clock_time(timestamp_text: str) -> str:
    try:
        parsed = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp_text
    return parsed.strftime("%I:%M %p").lstrip("0")
