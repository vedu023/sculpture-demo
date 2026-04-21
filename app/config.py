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
    backend: str = "auto"
    model_name: str = "small.en"
    indic_model_name: str = "somyalab/Vyasa_mini_rnnt_onnx_v2"
    language_mode: str = "auto"
    device: str = "cpu"
    compute_type: str = "float32"
    beam_size: int = 2
    fallback_beam_size: int = 2
    language: str = "auto"
    condition_on_previous_text: bool = False


@dataclass
class LLMConfig:
    model_name: str = "gemma4:e4b"
    max_tokens: int = 100
    max_sentences: int = 2
    max_history_turns: int = 10
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
    backend: str = "auto"
    device: str = "cpu"
    audio_prompt_path: Path = field(default_factory=lambda: PROJECT_ROOT / "voice_ref_clean.wav")
    voice_state_path: Path = field(default_factory=lambda: PROJECT_ROOT / "voice_ref_clean.safetensors")
    pocket_voice: str = "alba"
    pocket_speed: float = 1.0
    spark_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "Spark_somya_TTS")
    spark_repo_id: str = "somyalab/Spark_somya_TTS"
    spark_temperature: float = 0.7
    spark_top_k: int = 50
    spark_top_p: float = 0.95


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    logs_dir: Path = field(default_factory=lambda: DEFAULT_LOGS_DIR)
