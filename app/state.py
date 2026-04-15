from enum import Enum, auto


class BotState(Enum):
    LISTENING = auto()
    PROCESSING_ASR = auto()
    PROCESSING_LLM = auto()
    PROCESSING_TTS = auto()
    SPEAKING = auto()
    ERROR = auto()
