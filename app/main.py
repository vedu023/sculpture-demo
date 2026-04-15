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
        choices=("pocket_tts",),
        default=None,
        help="Override the default TTS backend",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("devices", help="List audio devices")
    subparsers.add_parser("test-audio", help="Record and playback a short test clip")
    subparsers.add_parser("capture", help="Capture one utterance and save it")
    subparsers.add_parser("transcribe", help="Capture one utterance and transcribe it")
    subparsers.add_parser("chat", help="Run one capture -> ASR -> LLM -> TTS turn")
    subparsers.add_parser("bootstrap-models", help="Download and cache model files")
    subparsers.add_parser("prepare-voice", help="Encode voice ref and cache as .safetensors")
    return parser


def bootstrap_models(config: AppConfig, force: bool = False):
    asr = ASREngine(
        model_name=config.asr.model_name,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        beam_size=config.asr.beam_size,
        language=config.asr.language,
        condition_on_previous_text=config.asr.condition_on_previous_text,
    )
    llm = LLMEngine(
        model_name=config.llm.model_name,
        system_prompt=config.llm.system_prompt,
        max_tokens=config.llm.max_tokens,
        max_sentences=config.llm.max_sentences,
    )
    tts = TTSEngine(
        backend=config.tts.backend,
        voice=config.tts.pocket_voice,
        speed=config.tts.pocket_speed,
        audio_prompt_path=config.tts.audio_prompt_path,
        voice_state_path=config.tts.voice_state_path,
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
            model_name=config.asr.model_name,
            device=config.asr.device,
            compute_type=config.asr.compute_type,
            beam_size=config.asr.beam_size,
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
        if not wav_path.exists():
            print(f"Error: voice ref not found: {wav_path}")
            print("Run process_voice_ref.py first to create it.")
            return
        print(f"Encoding voice from: {wav_path}")
        from pocket_tts import TTSModel
        from pocket_tts.models.tts_model import export_model_state
        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(
            audio_conditioning=str(wav_path)
        )
        export_model_state(voice_state, str(out_path))
        print(f"Voice state cached to: {out_path}")
        print("TTS will now load instantly on startup.")
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
