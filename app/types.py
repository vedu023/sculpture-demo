from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _isoformat_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().isoformat()


@dataclass
class CapturedUtterance:
    samples: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float
    duration_ms: float
    speech_ms: float
    trailing_silence_ms: float
    chunk_count: int
    queue_overflow_count: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "started_at": self.started_at,
            "started_at_iso": _isoformat_timestamp(self.started_at),
            "ended_at": self.ended_at,
            "ended_at_iso": _isoformat_timestamp(self.ended_at),
            "duration_ms": round(self.duration_ms, 1),
            "speech_ms": round(self.speech_ms, 1),
            "trailing_silence_ms": round(self.trailing_silence_ms, 1),
            "chunk_count": self.chunk_count,
            "queue_overflow_count": self.queue_overflow_count,
            "sample_count": int(self.samples.size),
        }


@dataclass
class SynthesizedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_ms: float

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_ms": round(self.duration_ms, 1),
            "sample_count": int(self.samples.size),
        }


@dataclass
class TurnResult:
    turn_id: int
    status: str
    assistant_name: str = ""
    tts_backend: str = ""
    reply_language: str = ""
    transcript: str = ""
    reply: str = ""
    utterance: CapturedUtterance | None = None
    synthesized_audio: SynthesizedAudio | None = None
    capture_ms: float = 0.0
    asr_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    tts_ms: float = 0.0
    playback_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_status: str = ""
    tool_result: dict[str, Any] | None = None
    confirmation_required: bool = False
    state_path: tuple[str, ...] = ()

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "status": self.status,
            "assistant_name": self.assistant_name,
            "tts_backend": self.tts_backend,
            "reply_language": self.reply_language,
            "transcript": self.transcript,
            "reply": self.reply,
            "capture_ms": round(self.capture_ms, 1),
            "asr_ms": round(self.asr_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "tool_ms": round(self.tool_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "playback_ms": round(self.playback_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "error": self.error,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_status": self.tool_status,
            "tool_result": self.tool_result,
            "confirmation_required": self.confirmation_required,
            "state_path": list(self.state_path),
            "utterance": self.utterance.to_log_dict() if self.utterance else None,
            "synthesized_audio": (
                self.synthesized_audio.to_log_dict() if self.synthesized_audio else None
            ),
        }
