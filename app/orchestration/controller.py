from __future__ import annotations

import logging
from pathlib import Path

from app.asr.engine import ASREngine
from app.audio.capture import CaptureSession
from app.audio.output import AudioOutput
from app.audio.save import save_utterance
from app.config import AppConfig
from app.language import (
    confirmation_cancelled,
    confirmation_prompt,
    confirmation_retry,
    default_reply_language,
    detect_text_language,
    fallback_reply,
    greeting_fallback,
    is_affirmative,
    is_mixed_indic_script,
    is_negative,
    localize_tool_result,
    normalize_language_mode,
)
from app.llm.engine import LLMEngine
from app.persona import build_greeting_prompt, build_system_prompt, build_tool_planner_prompt
from app.state import BotState
from app.tools.builtin import build_builtin_tool_registry
from app.tools.executor import ToolExecutor
from app.tools.router import route_tool_intent
from app.tools.types import ToolCall
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
        tool_registry=None,
        tool_executor: ToolExecutor | None = None,
        tool_router=None,
    ):
        self.config = config
        self.state = BotState.LISTENING
        self._capture = capture_session or CaptureSession(config)
        self._asr = asr or ASREngine(
            backend=config.asr.backend,
            model_name=config.asr.model_name,
            indic_model_name=config.asr.indic_model_name,
            language_mode=config.asr.language_mode,
            device=config.asr.device,
            compute_type=config.asr.compute_type,
            beam_size=config.asr.beam_size,
            fallback_beam_size=config.asr.fallback_beam_size,
            language=config.asr.language,
            condition_on_previous_text=config.asr.condition_on_previous_text,
        )
        self._llm = llm or LLMEngine(
            model_name=config.llm.model_name,
            system_prompt=config.llm.system_prompt,
            max_tokens=config.llm.max_tokens,
            max_sentences=config.llm.max_sentences,
            max_history_turns=config.llm.max_history_turns,
        )
        self._tts = tts or TTSEngine(
            backend=config.tts.backend,
            voice=config.tts.pocket_voice,
            speed=config.tts.pocket_speed,
            device=config.tts.device,
            audio_prompt_path=config.tts.audio_prompt_path,
            voice_state_path=config.tts.voice_state_path,
            spark_model_dir=config.tts.spark_model_dir,
            spark_repo_id=config.tts.spark_repo_id,
            spark_temperature=config.tts.spark_temperature,
            spark_top_k=config.tts.spark_top_k,
            spark_top_p=config.tts.spark_top_p,
        )
        self._speaker = speaker or AudioOutput(
            default_sample_rate=config.audio.sample_rate,
            device=config.audio.output_device,
        )
        self._tool_registry = tool_registry or build_builtin_tool_registry(config, self._speaker)
        self._tool_executor = tool_executor or ToolExecutor(self._tool_registry)
        self._tool_router = tool_router or route_tool_intent
        self._session_logger = session_logger or SessionLogger(config.logs_dir, "chat")
        self._turn_counter = 0
        self._started = False
        self._pending_tool_call: ToolCall | None = None
        self._last_reply_language = default_reply_language(
            config.asr.language,
            config.asr.language_mode,
        )

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
        if hasattr(self._session_logger, "close"):
            try:
                self._session_logger.close()
            except Exception:
                logger.debug("Failed to close session logger cleanly", exc_info=True)
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
                logger.info("ASR output: %s", transcript if transcript else "(empty)")
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
                logger.info("ASR output: %s", transcript if transcript else "(empty)")
                if not transcript:
                    result.status = "empty_transcript"
                else:
                    reply_language = self._detect_reply_language(transcript)
                    result.reply_language = reply_language
                    reply = self._reply_for_transcript(transcript, reply_language, result, state_path)
                    result.reply = reply
                    if not reply:
                        result.status = "empty_reply"
                    else:
                        self._llm.remember_turn(transcript, reply)
                        self._set_state(BotState.PROCESSING_TTS, state_path)
                        result.tts_backend = self._tts.describe(reply_language)
                        with Timer() as tts_timer:
                            synthesized = self._tts.synthesize(reply, language=reply_language)
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

    def _reply_for_transcript(
        self,
        transcript: str,
        reply_language: str,
        result: TurnResult,
        state_path: list[str],
    ) -> str:
        if self._pending_tool_call is not None:
            return self._handle_pending_confirmation(transcript, reply_language, result, state_path)

        self._set_state(BotState.PLANNING, state_path)
        current_volume_percent = None
        if hasattr(self._speaker, "get_volume_percent"):
            current_volume_percent = self._speaker.get_volume_percent()
        routed_tool_call = self._tool_router(
            transcript,
            self._tool_registry,
            current_volume_percent=current_volume_percent,
        )
        if routed_tool_call is not None:
            logger.info("Tool router matched transcript to %s", routed_tool_call.tool_name)
            return self._handle_tool_call(routed_tool_call, reply_language, result, state_path)

        planner_prompt = build_tool_planner_prompt(
            assistant_name=self.config.llm.assistant_name,
            persona_style=self.config.llm.persona_style,
            max_sentences=self.config.llm.max_sentences,
            tool_specs=self._tool_registry.specs_for_prompt(),
            reply_language=reply_language,
        )
        with Timer() as planner_timer:
            decision = self._llm.plan_turn(
                transcript,
                planner_prompt,
                tool_schemas=self._tool_registry.ollama_tools(),
            )
        result.llm_ms += planner_timer.elapsed_ms

        if decision.is_tool_call:
            return self._handle_tool_call(
                ToolCall(tool_name=decision.tool_name, arguments=decision.arguments),
                reply_language,
                result,
                state_path,
            )

        reply = decision.spoken_response.strip()
        if reply:
            return reply

        self._set_state(BotState.PROCESSING_LLM, state_path)
        system_prompt = build_system_prompt(
            assistant_name=self.config.llm.assistant_name,
            persona_style=self.config.llm.persona_style,
            max_sentences=self.config.llm.max_sentences,
            reply_language=reply_language,
        )
        with Timer() as llm_timer:
            reply = self._llm.generate_with_system_prompt(
                system_prompt=system_prompt,
                user_text=transcript,
                remember=False,
                fallback_reply=fallback_reply(reply_language),
            ).strip()
        result.llm_ms += llm_timer.elapsed_ms
        return reply

    def _handle_tool_call(
        self,
        tool_call: ToolCall,
        reply_language: str,
        result: TurnResult,
        state_path: list[str],
    ) -> str:
        result.tool_name = tool_call.tool_name
        result.tool_args = dict(tool_call.arguments)

        definition = self._tool_registry.get(tool_call.tool_name)
        if definition is None:
            result.status = "tool_error"
            result.tool_status = "error"
            if reply_language == "hi":
                return "मेरे पास उसके लिए सही टूल अभी नहीं है।"
            if reply_language == "kn":
                return "ಅದಕ್ಕೆ ತಕ್ಕ ಟೂಲ್ ಈಗ ನನ್ನ ಬಳಿ ಇಲ್ಲ."
            return "I do not have a matching tool for that yet."

        if definition.side_effect:
            self._pending_tool_call = tool_call
            result.status = "confirmation_required"
            result.confirmation_required = True
            result.tool_status = "pending_confirmation"
            self._set_state(BotState.CONFIRMING_ACTION, state_path)
            return self._build_confirmation_prompt(tool_call, reply_language)

        return self._execute_tool_call(tool_call, reply_language, result, state_path)

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        reply_language: str,
        result: TurnResult,
        state_path: list[str],
    ) -> str:
        self._set_state(BotState.EXECUTING_TOOL, state_path)
        with Timer() as tool_timer:
            tool_result = self._tool_executor.execute(tool_call)
        result.tool_ms += tool_timer.elapsed_ms
        result.tool_name = tool_result.tool_name or tool_call.tool_name
        result.tool_args = dict(tool_call.arguments)
        result.tool_status = "ok" if tool_result.ok else "error"
        result.tool_result = tool_result.data
        if not tool_result.ok:
            result.status = "tool_error"
        return localize_tool_result(tool_result, reply_language)

    def _handle_pending_confirmation(
        self,
        transcript: str,
        reply_language: str,
        result: TurnResult,
        state_path: list[str],
    ) -> str:
        pending_tool_call = self._pending_tool_call
        if pending_tool_call is None:
            return ""

        result.tool_name = pending_tool_call.tool_name
        result.tool_args = dict(pending_tool_call.arguments)
        result.confirmation_required = True

        if is_negative(transcript):
            self._pending_tool_call = None
            result.status = "cancelled"
            result.tool_status = "cancelled"
            self._set_state(BotState.CONFIRMING_ACTION, state_path)
            return confirmation_cancelled(reply_language)

        if is_affirmative(transcript):
            self._pending_tool_call = None
            return self._execute_tool_call(pending_tool_call, reply_language, result, state_path)

        self._set_state(BotState.CONFIRMING_ACTION, state_path)
        result.status = "confirmation_required"
        result.tool_status = "pending_confirmation"
        return confirmation_retry(reply_language)

    def _build_confirmation_prompt(self, tool_call: ToolCall, reply_language: str) -> str:
        return confirmation_prompt(tool_call, reply_language)

    def _speak_greeting(self):
        """Generate and speak a short greeting when the bot comes online."""
        reply_language = default_reply_language(
            self.config.asr.language,
            self.config.asr.language_mode,
        )
        greeting_prompt = build_greeting_prompt(
            assistant_name=self.config.llm.assistant_name,
            persona_style=self.config.llm.persona_style,
            reply_language=reply_language,
        )
        greeting = self._llm.generate_with_system_prompt(
            system_prompt=greeting_prompt,
            user_text="Greet the audience now.",
            remember=False,
            fallback_reply=greeting_fallback(reply_language),
        )
        if greeting:
            logger.info("Greeting: %s", greeting)
            synthesized = self._tts.synthesize(greeting, language=reply_language)
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
        assistant_name = result.assistant_name or self.config.llm.assistant_name
        transcript = truncate_for_log(result.transcript)
        reply = truncate_for_log(result.reply)
        logger.info(
            "Turn %03d | %s | %s | status=%s | timing: utt=%4.0fms asr=%4.0fms llm=%4.0fms tool=%4.0fms tts=%4.0fms playback=%4.0fms total=%4.0fms | vad_end=%s | ASR=%s | reply=%s",
            result.turn_id,
            assistant_name,
            result.tts_backend or self._tts.describe(),
            result.status,
            utterance_duration_ms,
            result.asr_ms,
            result.llm_ms,
            result.tool_ms,
            result.tts_ms,
            result.playback_ms,
            result.total_ms,
            vad_end,
            transcript if transcript else "(no speech)",
            reply if reply else "(none)",
        )
        if result.tool_name:
            logger.info(
                "  tool=%s status=%s confirm=%s args=%s",
                result.tool_name,
                result.tool_status or "-",
                result.confirmation_required,
                truncate_for_log(str(result.tool_args)),
            )
        if result.error:
            logger.warning("  error=%s", truncate_for_log(result.error))

    def _detect_reply_language(self, transcript: str) -> str:
        language_mode = normalize_language_mode(self.config.asr.language_mode)
        if language_mode == "english":
            self._last_reply_language = "en"
            return "en"

        configured_language = self.config.asr.language
        fallback_language = default_reply_language(configured_language, language_mode)
        if is_mixed_indic_script(transcript):
            if language_mode == "indic" and self._last_reply_language not in {"hi", "kn"}:
                self._last_reply_language = fallback_language
            return self._last_reply_language
        detected_language = detect_text_language(transcript, default=fallback_language)
        if language_mode == "indic" and detected_language not in {"hi", "kn"}:
            return self._last_reply_language
        if detected_language in {"hi", "kn"}:
            self._last_reply_language = detected_language
        return detected_language
