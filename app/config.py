from dataclasses import dataclass, field
from pathlib import Path

from app.persona import build_system_prompt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 512
    input_device: int | None = None
    output_device: int | None = None
    queue_maxsize: int = 128
    max_utterance_ms: int | None = 6000


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 350
    preroll_ms: int = 300
    trim_trailing_silence: bool = True


@dataclass
class ASRConfig:
    model_name: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    language: str = "en"
    condition_on_previous_text: bool = False


@dataclass
class LLMConfig:
    model_name: str = "gemma4:e4b"
    max_tokens: int = 48
    max_sentences: int = 2
    assistant_name: str = "Smruti"
    persona_style: str = "smruti"

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(
            assistant_name=self.assistant_name,
            persona_style=self.persona_style,
            max_sentences=self.max_sentences,
        )


@dataclass
class TTSConfig:
    backend: str = "pocket_tts"
    device: str = "cpu"
    audio_prompt_path: Path = field(default_factory=lambda: PROJECT_ROOT / "voice_ref_clean.wav")
    voice_state_path: Path = field(default_factory=lambda: PROJECT_ROOT / "voice_ref_clean.safetensors")
    pocket_voice: str = "alba"
    pocket_speed: float = 1.0


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    logs_dir: Path = field(default_factory=lambda: DEFAULT_LOGS_DIR)
