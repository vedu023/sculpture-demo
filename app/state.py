from enum import Enum, auto


class BotState(Enum):
    LISTENING = auto()
    PROCESSING_ASR = auto()
    PLANNING = auto()
    EXECUTING_TOOL = auto()
    CONFIRMING_ACTION = auto()
    PROCESSING_LLM = auto()
    PROCESSING_TTS = auto()
    SPEAKING = auto()
    ERROR = auto()
