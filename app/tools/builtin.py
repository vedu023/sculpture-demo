from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import sounddevice as sd

from app.audio.output import AudioOutput
from app.config import AppConfig
from app.tools.registry import ToolRegistry
from app.tools.types import ToolDefinition, ToolResult

TimeProvider = Callable[[], datetime]
DeviceProvider = Callable[[], list[dict[str, Any]]]


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


def build_builtin_tool_registry(
    config: AppConfig,
    speaker: AudioOutput,
    device_provider: DeviceProvider | None = None,
    now_provider: TimeProvider | None = None,
) -> ToolRegistry:
    device_provider = device_provider or sd.query_devices
    now_provider = now_provider or (lambda: datetime.now().astimezone())

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
        tts_backend = f"{config.tts.backend}/{config.tts.pocket_voice}"
        payload = {
            "assistant_name": config.llm.assistant_name,
            "llm_model": config.llm.model_name,
            "asr_model": config.asr.model_name,
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
                f"I am using {config.llm.model_name} for language, {config.asr.model_name} for listening, "
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
            name="set_output_volume",
            description="Set the assistant output volume from 0 to 100 percent.",
            parameters={"volume_percent": "integer from 0 to 100"},
            side_effect=True,
            handler=set_output_volume,
        ),
    ]
    return ToolRegistry(definitions)
