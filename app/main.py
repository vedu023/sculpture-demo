from __future__ import annotations

import argparse
import logging

import sounddevice as sd

from app.audio.input import AudioInput
from app.audio.output import AudioOutput
from app.config import AppConfig
from app.llm.engine import LLMEngine
from app.orchestration.controller import Controller
from app.asr.engine import ASREngine
from app.tts.bootstrap import bootstrap_runtime_models
from app.tts.engine import TTSEngine
from app.tts.voice_ref import encode_voice_state, process_voice_reference
from app.utils.logging import setup_logging
from app.utils.session_log import SessionLogger

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smruti Voice Bot Demo")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument(
        "--tts-backend",
        choices=("auto", "pocket_tts", "spark_somya_tts"),
        default=None,
        help="Override the default TTS backend",
    )
    parser.add_argument(
        "--asr-backend",
        choices=("auto", "faster_whisper", "onnx_runtime", "onnx_asr"),
        default=None,
        help="Override ASR backend selection",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Override the default Whisper ASR model (default from config)",
    )
    parser.add_argument(
        "--asr-indic-model",
        default=None,
        help="Override the Hindi/Kannada Vyasa ONNX bundle repo",
    )
    parser.add_argument(
        "--language-mode",
        choices=("auto", "english", "indic"),
        default=None,
        help="Keep the session in English only, Indic only, or auto-switch by transcript",
    )
    parser.add_argument(
        "--asr-beam-size",
        type=int,
        default=None,
        help="Override ASR beam size (default from config)",
    )
    parser.add_argument(
        "--asr-language",
        default=None,
        help="Override ASR language hint (for example: en, hi, kn, auto)",
    )
    parser.add_argument(
        "--asr-fallback-beam-size",
        type=int,
        default=None,
        help="Optional fallback beam size if first ASR pass is empty",
    )
    parser.add_argument(
        "--audio-block-size",
        type=int,
        default=None,
        help="Override capture block size (samples)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override VAD speech probability threshold",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=None,
        help="Override VAD min speech ms",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=None,
        help="Override VAD min silence ms",
    )
    parser.add_argument(
        "--max-utterance-ms",
        type=int,
        default=None,
        help="Override maximum utterance duration in ms",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("devices", help="List audio devices")
    subparsers.add_parser("test-audio", help="Record and playback a short test clip")
    subparsers.add_parser("capture", help="Capture one utterance and save it")
    subparsers.add_parser("transcribe", help="Capture one utterance and transcribe it")
    subparsers.add_parser("chat", help="Run one capture -> ASR -> LLM -> TTS turn")
    subparsers.add_parser("bootstrap-models", help="Download and cache model files")
    prepare_voice_parser = subparsers.add_parser(
        "prepare-voice",
        help="Process a voice ref WAV and cache the matching .safetensors state",
    )
    prepare_voice_parser.add_argument(
        "--input",
        default=None,
        help="Optional source audio to preprocess into the configured voice WAV before encoding",
    )
    prepare_voice_parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Maximum source duration in seconds when --input is provided",
    )
    return parser


def bootstrap_models(config: AppConfig, force: bool = False):
    asr = ASREngine(
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
    llm = LLMEngine(
        model_name=config.llm.model_name,
        system_prompt=config.llm.system_prompt,
        max_tokens=config.llm.max_tokens,
        max_sentences=config.llm.max_sentences,
        max_history_turns=config.llm.max_history_turns,
    )
    tts = TTSEngine(
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
    bootstrap_runtime_models(
        config,
        asr_engine=asr,
        llm_engine=llm,
        tts_engine=tts,
        force=force,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    logger.info("Starting Smruti Voice Bot Demo")

    config = AppConfig()

    # Override config from CLI args
    if args.input_device is not None:
        config.audio.input_device = args.input_device
    if args.output_device is not None:
        config.audio.output_device = args.output_device
    if args.tts_backend is not None:
        config.tts.backend = args.tts_backend
    if args.asr_backend is not None:
        config.asr.backend = args.asr_backend
    if args.asr_model is not None:
        model_name = args.asr_model.strip()
        if model_name == "samll.en":
            model_name = "small.en"
        config.asr.model_name = model_name
    if args.asr_indic_model is not None:
        config.asr.indic_model_name = args.asr_indic_model.strip()
    if args.language_mode is not None:
        config.asr.language_mode = args.language_mode.strip().lower()
    if args.asr_beam_size is not None:
        config.asr.beam_size = args.asr_beam_size
    if args.asr_fallback_beam_size is not None:
        config.asr.fallback_beam_size = args.asr_fallback_beam_size
    if args.asr_language is not None:
        config.asr.language = args.asr_language.strip().lower()
    if args.audio_block_size is not None:
        config.audio.block_size = args.audio_block_size
    if args.vad_threshold is not None:
        config.vad.threshold = args.vad_threshold
    if args.vad_min_speech_ms is not None:
        config.vad.min_speech_ms = args.vad_min_speech_ms
    if args.vad_min_silence_ms is not None:
        config.vad.min_silence_ms = args.vad_min_silence_ms
    if args.max_utterance_ms is not None:
        config.audio.max_utterance_ms = args.max_utterance_ms

    # Ensure logs directory exists
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "devices":
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
        return

    if args.command == "test-audio":
        print("Testing audio input/output...")
        print("Recording 2 seconds of audio...")
        audio_input = AudioInput(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            block_size=config.audio.block_size,
            device=config.audio.input_device,
        )
        audio_output = AudioOutput(
            default_sample_rate=config.audio.sample_rate,
            device=config.audio.output_device,
        )
        frames = audio_input.record(duration=2.0)
        print("Playing back captured audio...")
        audio_output.play(frames)
        print("Audio test complete!")
        return

    if args.command == "capture":
        print("Starting voice capture... (Ctrl+C to stop)")
        from app.audio.capture import CaptureSession
        from app.audio.save import save_utterance
        try:
            with CaptureSession(config) as capture:
                utterance = capture.next_utterance()
            save_path = save_utterance(utterance, output_dir=config.logs_dir)
            print(f"Utterance saved to: {save_path}")
        except KeyboardInterrupt:
            print("Capture interrupted")
        return

    if args.command == "transcribe":
        print("Starting transcription test... (Ctrl+C to stop)")
        from app.audio.capture import CaptureSession
        asr = ASREngine(
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
        try:
            with CaptureSession(config) as capture:
                utterance = capture.next_utterance()
            print("Transcribing...")
            transcript = asr.transcribe(utterance.samples)
            print(f"Transcript: {transcript}")
        except KeyboardInterrupt:
            print("Transcription interrupted")
        return

    if args.command == "bootstrap-models":
        print("Bootstrapping models...")
        bootstrap_models(config, force=True)
        print("Models bootstrapped successfully!")
        return

    if args.command == "prepare-voice":
        wav_path = config.tts.audio_prompt_path
        out_path = config.tts.voice_state_path
        try:
            if args.input:
                print(f"Processing voice reference from: {args.input}")
                process_result = process_voice_reference(
                    args.input,
                    wav_path,
                    duration=args.duration,
                )
                metadata = process_result.get("metadata", {})
                if metadata:
                    duration_seconds = float(metadata.get("duration", 0) or 0)
                    sample_rate = metadata.get("sample_rate", "?")
                    print(f"Processed WAV: {duration_seconds:.1f}s at {sample_rate} Hz")

            if not wav_path.exists():
                print(f"Error: voice ref not found: {wav_path}")
                print("Pass --input to preprocess a new source file or run process_voice_ref.py first.")
                return

            print(f"Encoding voice from: {wav_path}")
            encode_voice_state(wav_path, out_path)
            print(f"Voice state cached to: {out_path}")
            print("TTS will now load instantly on startup.")
        except Exception as exc:
            print(f"Error: {exc}")
        return

    if args.command == "chat":
        print("Starting chat mode... (Ctrl+C to stop)")
        print("Make sure Ollama is running: ollama serve")

        session_logger = SessionLogger(config.logs_dir, "chat")

        try:
            controller = Controller(config, session_logger=session_logger)
            controller.run()
        except KeyboardInterrupt:
            print("\nChat session ended")
        except Exception as e:
            logger.error("Chat session failed: %s", e)
            raise
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
