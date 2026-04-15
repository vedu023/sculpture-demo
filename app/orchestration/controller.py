from __future__ import annotations

import logging
from pathlib import Path

from app.asr.engine import ASREngine
from app.audio.capture import CaptureSession
from app.audio.output import AudioOutput
from app.audio.save import save_utterance
from app.config import AppConfig
from app.llm.engine import LLMEngine
from app.persona import build_greeting_prompt
from app.state import BotState
from app.tts.engine import TTSEngine
from app.types import CapturedUtterance, TurnResult
from app.utils.session_log import SessionLogger
from app.utils.text import truncate_for_log
from app.utils.timers import Timer

logger = logging.getLogger(__name__)


class Controller:
    """Sequential half-duplex controller for the local voice bot demo."""

    def __init__(
        self,
        config: AppConfig,
        capture_session: CaptureSession | None = None,
        asr: ASREngine | None = None,
        llm: LLMEngine | None = None,
        tts: TTSEngine | None = None,
        speaker: AudioOutput | None = None,
        session_logger: SessionLogger | None = None,
    ):
        self.config = config
        self.state = BotState.LISTENING
        self._capture = capture_session or CaptureSession(config)
        self._asr = asr or ASREngine(
            model_name=config.asr.model_name,
            device=config.asr.device,
            compute_type=config.asr.compute_type,
            beam_size=config.asr.beam_size,
            language=config.asr.language,
            condition_on_previous_text=config.asr.condition_on_previous_text,
        )
        self._llm = llm or LLMEngine(
            model_name=config.llm.model_name,
            system_prompt=config.llm.system_prompt,
            max_tokens=config.llm.max_tokens,
            max_sentences=config.llm.max_sentences,
        )
        self._tts = tts or TTSEngine(
            backend=config.tts.backend,
            voice=config.tts.pocket_voice,
            speed=config.tts.pocket_speed,
            audio_prompt_path=config.tts.audio_prompt_path,
            voice_state_path=config.tts.voice_state_path,
        )
        self._speaker = speaker or AudioOutput(
            default_sample_rate=config.audio.sample_rate,
            device=config.audio.output_device,
        )
        self._session_logger = session_logger or SessionLogger(config.logs_dir, "chat")
        self._turn_counter = 0
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if self._started:
            return
        self._capture.start()
        self._started = True

    def stop(self):
        if not self._started:
            return
        self._capture.stop()
        self._started = False
        logger.debug("Controller stopped after %d turn(s)", self._turn_counter)

    def _set_state(self, next_state: BotState, state_path: list[str] | None = None):
        if next_state == self.state:
            return
        logger.debug("State: %s -> %s", self.state.name, next_state.name)
        self.state = next_state
        if state_path is not None:
            state_path.append(next_state.name)

    def warmup(self, include_asr: bool = True, include_llm: bool = True, include_tts: bool = True):
        logger.info("Warming up %s...", self.config.llm.assistant_name)
        if include_asr:
            self._asr.warmup()
        if include_llm:
            self._llm.warmup()
        if include_tts:
            self._tts.warmup()
        logger.info("%s online | tts=%s", self.config.llm.assistant_name, self._tts.describe())

    def capture_once(self, save: bool = True) -> tuple[CapturedUtterance, Path | None]:
        self.start()
        logger.info("Listening...")
        utterance = self._capture.next_utterance()
        saved_path = save_utterance(utterance, output_dir=self.config.logs_dir) if save else None
        self._session_logger.log_event(
            "capture",
            {
                "utterance": utterance.to_log_dict(),
                "saved_path": str(saved_path) if saved_path else None,
            },
        )
        logger.info(
            "capture duration=%.0fms speech=%.0fms trailing=%.0fms file=%s",
            utterance.duration_ms,
            utterance.speech_ms,
            utterance.trailing_silence_ms,
            saved_path or "-",
        )
        return utterance, saved_path

    def transcribe_once(self) -> TurnResult:
        self.start()
        self.warmup(include_llm=False, include_tts=False)
        self._turn_counter += 1
        state_path = [self.state.name]

        result = TurnResult(
            turn_id=self._turn_counter,
            status="ok",
            assistant_name=self.config.llm.assistant_name,
            tts_backend=self._tts.describe(),
        )
        with Timer() as total_timer:
            try:
                self._set_state(BotState.LISTENING, state_path)
                logger.info("Listening...")
                with Timer() as capture_timer:
                    utterance = self._capture.next_utterance()
                result.capture_ms = capture_timer.elapsed_ms
                result.utterance = utterance

                self._set_state(BotState.PROCESSING_ASR, state_path)
                with Timer() as asr_timer:
                    transcript = self._asr.transcribe(utterance.samples)
                result.asr_ms = asr_timer.elapsed_ms
                result.transcript = transcript
                if not transcript:
                    result.status = "empty_transcript"
                result.state_path = tuple(state_path)
            except Exception as exc:
                self._set_state(BotState.ERROR, state_path)
                result.status = "error"
                result.error = str(exc)
            finally:
                self._set_state(BotState.LISTENING, state_path)
                result.state_path = tuple(state_path)

        result.total_ms = total_timer.elapsed_ms

        self._session_logger.log_event("turn", result.to_log_dict())
        self._log_turn_summary(result)
        return result

    def run_turn(self, play_audio: bool = True) -> TurnResult:
        self.start()
        self._turn_counter += 1
        state_path = [self.state.name]
        result = TurnResult(
            turn_id=self._turn_counter,
            status="ok",
            assistant_name=self.config.llm.assistant_name,
            tts_backend=self._tts.describe(),
        )

        with Timer() as total_timer:
            try:
                self._set_state(BotState.LISTENING, state_path)
                logger.info("Listening...")
                with Timer() as capture_timer:
                    utterance = self._capture.next_utterance()
                result.capture_ms = capture_timer.elapsed_ms
                result.utterance = utterance

                self._set_state(BotState.PROCESSING_ASR, state_path)
                with Timer() as asr_timer:
                    transcript = self._asr.transcribe(utterance.samples)
                result.asr_ms = asr_timer.elapsed_ms
                result.transcript = transcript
                if not transcript:
                    result.status = "empty_transcript"
                else:
                    self._set_state(BotState.PROCESSING_LLM, state_path)
                    with Timer() as llm_timer:
                        reply = self._llm.generate(transcript).strip()
                    result.llm_ms = llm_timer.elapsed_ms
                    result.reply = reply
                    if not reply:
                        result.status = "empty_reply"
                    else:
                        self._set_state(BotState.PROCESSING_TTS, state_path)
                        with Timer() as tts_timer:
                            synthesized = self._tts.synthesize(reply)
                        result.tts_ms = tts_timer.elapsed_ms
                        result.synthesized_audio = synthesized
                        if synthesized.samples.size == 0:
                            result.status = "empty_audio"
                        elif play_audio:
                            self._set_state(BotState.SPEAKING, state_path)
                            with Timer() as playback_timer:
                                self._speaker.play(
                                    synthesized.samples,
                                    sample_rate=synthesized.sample_rate,
                                )
                            result.playback_ms = playback_timer.elapsed_ms
            except Exception as exc:
                self._set_state(BotState.ERROR, state_path)
                logger.exception("Pipeline error on turn %d", self._turn_counter)
                result.status = "error"
                result.error = str(exc)
            finally:
                self._set_state(BotState.LISTENING, state_path)
                result.state_path = tuple(state_path)
        result.total_ms = total_timer.elapsed_ms
        self._session_logger.log_event("turn", result.to_log_dict())
        self._log_turn_summary(result)

        return result

    def _speak_greeting(self):
        """Generate and speak a short greeting when the bot comes online."""
        greeting_prompt = build_greeting_prompt(
            assistant_name=self.config.llm.assistant_name,
            persona_style=self.config.llm.persona_style,
        )
        greeting = self._llm.generate(greeting_prompt)
        if greeting:
            logger.info("Greeting: %s", greeting)
            synthesized = self._tts.synthesize(greeting)
            if synthesized.samples.size > 0:
                self._speaker.play(synthesized.samples, sample_rate=synthesized.sample_rate)

    def run(self, max_turns: int | None = None):
        """Run the voice pipeline until interrupted or max_turns is reached."""
        if max_turns is not None and max_turns <= 0:
            return

        self.start()
        self.warmup()
        self._speak_greeting()
        logger.info("Voice chat loop started. Press Ctrl+C to stop.")

        turn_count = 0
        try:
            while max_turns is None or turn_count < max_turns:
                turn_count += 1
                self.run_turn(play_audio=True)
        except KeyboardInterrupt:
            logger.info("Chat ended.")

    def _log_turn_summary(self, result: TurnResult):
        utterance_duration_ms = result.utterance.duration_ms if result.utterance else 0.0
        vad_end = "-"
        if result.utterance is not None:
            vad_end = result.utterance.to_log_dict()["ended_at_iso"]
        logger.info(
            "turn=%03d name=%s tts=%s status=%s utterance=%.0fms vad_end=%s asr=%.0fms llm=%.0fms tts=%.0fms total=%.0fms text=\"%s\" reply=\"%s\"%s",
            result.turn_id,
            result.assistant_name or self.config.llm.assistant_name,
            result.tts_backend or self._tts.describe(),
            result.status,
            utterance_duration_ms,
            vad_end,
            result.asr_ms,
            result.llm_ms,
            result.tts_ms,
            result.total_ms,
            truncate_for_log(result.transcript),
            truncate_for_log(result.reply),
            f" error=\"{truncate_for_log(result.error)}\"" if result.error else "",
        )
