from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import sounddevice as sd

from app.audio.output import AudioOutput
from app.config import AppConfig
from app.tools.calendar import MacOSCalendarProvider
from app.tools.registry import ToolRegistry
from app.tools.types import ToolDefinition, ToolResult

TimeProvider = Callable[[], datetime]
DeviceProvider = Callable[[], list[dict[str, Any]]]
CalendarProvider = Callable[[str], list[dict[str, Any]]]


def _coerce_volume_percent(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    volume_percent = int(float(value))
    return max(0, min(100, volume_percent))


def _format_device_summary(devices: list[dict[str, Any]]) -> str:
    if not devices:
        return "I could not find any audio devices."

    names = [device.get("name", f"Device {index}") for index, device in enumerate(devices[:3])]
    if len(devices) == 1:
        return f"I found 1 audio device: {names[0]}."
    return f"I found {len(devices)} audio devices. The first few are {', '.join(names)}."


def _format_clock_time(timestamp_text: str) -> str:
    try:
        parsed = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp_text
    return parsed.strftime("%I:%M %p").lstrip("0")


def _describe_configured_tts(config: AppConfig) -> str:
    pocket_name = f"pocket_tts/{config.tts.pocket_voice}"
    spark_name = f"spark_somya_tts/{config.tts.spark_model_dir.name}"
    if config.tts.backend == "spark_somya_tts":
        return spark_name
    if config.tts.backend == "auto":
        return f"auto ({pocket_name} for English, {spark_name} for hi/kn)"
    return pocket_name


def _summarize_calendar_events(
    events: list[dict[str, Any]],
    *,
    date_scope: str,
    mode: str,
    now_value: datetime,
) -> str:
    label = "tomorrow" if date_scope == "tomorrow" else "today"
    if not events:
        return f"You have nothing on your calendar {label}."

    if mode == "next":
        selected = events[0]
        if date_scope == "today":
            for event in events:
                try:
                    event_start = datetime.strptime(event["start_text"], "%Y-%m-%d %H:%M")
                except ValueError:
                    selected = event
                    break
                if event_start >= now_value.replace(tzinfo=None):
                    selected = event
                    break
        return (
            f"Your next event {label} is {selected['title']} at "
            f"{_format_clock_time(selected['start_text'])}."
        )

    if len(events) == 1:
        event = events[0]
        return f"You have 1 event {label}: {event['title']} at {_format_clock_time(event['start_text'])}."

    first_event = events[0]
    second_event = events[1]
    return (
        f"You have {len(events)} events {label}. First is {first_event['title']} at "
        f"{_format_clock_time(first_event['start_text'])}, then {second_event['title']} at "
        f"{_format_clock_time(second_event['start_text'])}."
    )


def build_builtin_tool_registry(
    config: AppConfig,
    speaker: AudioOutput,
    device_provider: DeviceProvider | None = None,
    now_provider: TimeProvider | None = None,
    calendar_provider: CalendarProvider | None = None,
) -> ToolRegistry:
    device_provider = device_provider or sd.query_devices
    now_provider = now_provider or (lambda: datetime.now().astimezone())
    calendar_provider = calendar_provider or MacOSCalendarProvider()

    def get_time(_: dict[str, Any]) -> ToolResult:
        now = now_provider()
        formatted = now.strftime("%I:%M %p").lstrip("0")
        return ToolResult(
            tool_name="get_time",
            ok=True,
            data={"time": formatted, "iso": now.isoformat()},
            spoken_response=f"It is {formatted}.",
        )

    def get_date(_: dict[str, Any]) -> ToolResult:
        now = now_provider()
        formatted = now.strftime("%A, %B %d, %Y")
        return ToolResult(
            tool_name="get_date",
            ok=True,
            data={"date": formatted, "iso": now.isoformat()},
            spoken_response=f"Today is {formatted}.",
        )

    def list_audio_devices(_: dict[str, Any]) -> ToolResult:
        devices = list(device_provider())
        payload = [
            {
                "index": index,
                "name": device.get("name", f"Device {index}"),
                "max_input_channels": int(device.get("max_input_channels", 0)),
                "max_output_channels": int(device.get("max_output_channels", 0)),
            }
            for index, device in enumerate(devices)
        ]
        return ToolResult(
            tool_name="list_audio_devices",
            ok=True,
            data={"devices": payload, "count": len(payload)},
            spoken_response=_format_device_summary(payload),
        )

    def get_runtime_status(_: dict[str, Any]) -> ToolResult:
        tts_backend = _describe_configured_tts(config)
        if config.asr.backend in {"onnx_runtime", "onnx_asr"}:
            asr_model = config.asr.indic_model_name
        elif config.asr.backend == "auto":
            asr_model = (
                f"{config.asr.indic_model_name} (hi/kn auto) with "
                f"{config.asr.model_name} as English fallback"
            )
        else:
            asr_model = config.asr.model_name
        payload = {
            "assistant_name": config.llm.assistant_name,
            "llm_model": config.llm.model_name,
            "asr_backend": config.asr.backend,
            "asr_model": asr_model,
            "tts_backend": tts_backend,
            "input_device": config.audio.input_device,
            "output_device": config.audio.output_device,
            "output_volume_percent": speaker.get_volume_percent(),
        }
        return ToolResult(
            tool_name="get_runtime_status",
            ok=True,
            data=payload,
            spoken_response=(
                f"You are talking to {config.llm.assistant_name}. "
                f"I am using {config.llm.model_name} for language, {asr_model} for listening, "
                f"and {tts_backend} for speech. Output volume is {speaker.get_volume_percent()} percent."
            ),
        )

    def set_output_volume(arguments: dict[str, Any]) -> ToolResult:
        volume_percent = _coerce_volume_percent(arguments.get("volume_percent", 50))
        speaker.set_volume(volume_percent / 100.0)
        return ToolResult(
            tool_name="set_output_volume",
            ok=True,
            data={"volume_percent": speaker.get_volume_percent()},
            spoken_response=f"Okay, I set the output volume to {speaker.get_volume_percent()} percent.",
        )

    def get_calendar_events(arguments: dict[str, Any]) -> ToolResult:
        date_scope = str(arguments.get("date_scope", "today")).strip().lower()
        if date_scope not in {"today", "tomorrow"}:
            date_scope = "today"
        mode = str(arguments.get("mode", "summary")).strip().lower()
        if mode not in {"summary", "next"}:
            mode = "summary"

        try:
            events = list(calendar_provider(date_scope))
        except Exception as exc:
            return ToolResult(
                tool_name="get_calendar_events",
                ok=False,
                spoken_response="I could not access Calendar on this Mac. Check Calendar permissions and try again.",
                error=str(exc),
            )

        now_value = now_provider()
        return ToolResult(
            tool_name="get_calendar_events",
            ok=True,
            data={"events": events, "count": len(events), "date_scope": date_scope, "mode": mode},
            spoken_response=_summarize_calendar_events(
                events,
                date_scope=date_scope,
                mode=mode,
                now_value=now_value,
            ),
        )

    definitions = [
        ToolDefinition(
            name="get_time",
            description="Read the current local time.",
            parameters={},
            side_effect=False,
            handler=get_time,
        ),
        ToolDefinition(
            name="get_date",
            description="Read the current local date.",
            parameters={},
            side_effect=False,
            handler=get_date,
        ),
        ToolDefinition(
            name="list_audio_devices",
            description="List available audio input and output devices.",
            parameters={},
            side_effect=False,
            handler=list_audio_devices,
        ),
        ToolDefinition(
            name="get_runtime_status",
            description="Report the assistant runtime configuration and current output volume.",
            parameters={},
            side_effect=False,
            handler=get_runtime_status,
        ),
        ToolDefinition(
            name="get_calendar_events",
            description="Read Calendar events for today or tomorrow, or report the next event.",
            parameters={
                "date_scope": 'either "today" or "tomorrow"',
                "mode": 'either "summary" or "next"',
            },
            side_effect=False,
            handler=get_calendar_events,
            parameter_schema={
                "date_scope": {
                    "type": "string",
                    "enum": ["today", "tomorrow"],
                    "description": "Which day to inspect.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["summary", "next"],
                    "description": "Whether to summarize the day or return just the next event.",
                },
            },
            required_parameters=("date_scope", "mode"),
        ),
        ToolDefinition(
            name="set_output_volume",
            description="Set the assistant output volume from 0 to 100 percent.",
            parameters={"volume_percent": "integer from 0 to 100"},
            side_effect=True,
            handler=set_output_volume,
            parameter_schema={
                "volume_percent": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Desired output volume percentage.",
                },
            },
            required_parameters=("volume_percent",),
        ),
    ]
    return ToolRegistry(definitions)
