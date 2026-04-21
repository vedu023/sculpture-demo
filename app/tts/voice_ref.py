from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]
ModelLoader = Callable[[], Any]
StateExporter = Callable[[Any, str], None]

_VOICE_FILTER = (
    "highpass=f=80,"
    "afftdn=nf=-25,"
    "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB,"
    "areverse,"
    "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB,"
    "areverse,"
    "loudnorm=I=-16:LRA=11:TP=-1.5"
)


def voice_state_is_stale(audio_prompt_path: Path | None, voice_state_path: Path | None) -> bool:
    if audio_prompt_path is None or not audio_prompt_path.exists():
        return False
    if voice_state_path is None or not voice_state_path.exists():
        return True
    return audio_prompt_path.stat().st_mtime > voice_state_path.stat().st_mtime


def should_use_cached_voice_state(audio_prompt_path: Path | None, voice_state_path: Path | None) -> bool:
    if voice_state_path is None or not voice_state_path.exists():
        return False
    return not voice_state_is_stale(audio_prompt_path, voice_state_path)


def process_voice_reference(
    input_path: str | Path,
    output_path: str | Path,
    duration: int = 10,
    *,
    runner: Runner | None = None,
    probe_runner: Runner | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    active_runner = runner or subprocess.run
    probe_runner = probe_runner or active_runner

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-af", _VOICE_FILTER,
        "-ar", "24000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-t", str(duration),
        str(output_file),
    ]
    result = active_runner(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed while processing voice reference")

    metadata: dict[str, Any] = {}
    probe_command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", str(output_file),
    ]
    probe_result = probe_runner(probe_command, capture_output=True, text=True)
    if probe_result.returncode == 0 and probe_result.stdout.strip():
        streams = json.loads(probe_result.stdout).get("streams", [])
        if streams:
            metadata = dict(streams[0])

    return {
        "command": command,
        "output_path": output_file,
        "metadata": metadata,
    }


def encode_voice_state(
    audio_prompt_path: str | Path,
    output_path: str | Path,
    *,
    model_loader: ModelLoader | None = None,
    state_exporter: StateExporter | None = None,
) -> Path:
    audio_prompt = Path(audio_prompt_path)
    output_file = Path(output_path)
    if not audio_prompt.exists():
        raise FileNotFoundError(f"Voice reference not found: {audio_prompt}")

    if model_loader is None or state_exporter is None:
        from pocket_tts import TTSModel
        from pocket_tts.models.tts_model import export_model_state
        model_loader = model_loader or TTSModel.load_model
        state_exporter = state_exporter or export_model_state

    model = model_loader()
    voice_state = model.get_state_for_audio_prompt(
        audio_conditioning=str(audio_prompt)
    )
    state_exporter(voice_state, str(output_file))
    return output_file
