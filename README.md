# Smruti — Local Voice Assistant Demo

A real-time voice assistant that runs entirely on your Mac. Speak into a mic, get a spoken response back. No cloud APIs, no latency surprises.

```
Mic → VAD → ASR → LLM → TTS → Speaker
```

## How It Works

1. **Listen** — captures audio from your microphone
2. **Detect** — Silero VAD detects when you start and stop speaking
3. **Transcribe** — faster-whisper converts speech to text locally
4. **Think** — Ollama runs a local LLM to generate a short reply
5. **Speak** — Pocket-TTS synthesizes the reply and plays it back
6. **Repeat** — goes back to listening

Average response time after you stop speaking: **~1-2 seconds**.

## Stack

| Component | Library | Model |
|-----------|---------|-------|
| Voice Activity Detection | [silero-vad-lite](https://github.com/snakers4/silero-vad) | Silero VAD |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `base.en` |
| Language Model | [Ollama](https://ollama.ai) | `gemma4:e4b` |
| Text-to-Speech | [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) | Pocket-TTS |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io) | — |

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.12+
- [Ollama](https://ollama.ai) installed
- Microphone + speaker/headphones

## Setup

```bash
# Clone and install
git clone <your-repo-url>
cd sculpture-demo
uv sync

# Pull the LLM model
ollama pull gemma4:e4b

# Start Ollama server (keep this running)
ollama serve
```

### Voice Reference (Optional)

To use a custom voice for TTS output, provide a short (5-10s) WAV recording:

```bash
# Process and clean the audio
python process_voice_ref.py your_recording.wav -o voice_ref_clean.wav

# Cache the voice state for instant loading
uv run voice-bot prepare-voice
```

Without a voice reference, Pocket-TTS uses its default `alba` voice.

## Usage

```bash
# Start the voice assistant
uv run voice-bot chat

# List audio devices
uv run voice-bot devices

# Record and play back a test clip
uv run voice-bot test-audio

# Capture one utterance and save to logs/
uv run voice-bot capture

# Capture and transcribe
uv run voice-bot transcribe

# Select specific audio devices
uv run voice-bot --input-device 1 --output-device 2 chat
```

Use `--debug` for detailed logs with state transitions and timing breakdowns.

## Project Structure

```
app/
  audio/           capture, VAD, playback, save
  asr/             speech recognition (faster-whisper)
  llm/             language model (Ollama)
  tts/             text-to-speech (Pocket-TTS)
  orchestration/   state machine controller
  utils/           logging, text processing, timers
tests/             unit tests
```

## Architecture

The system runs a sequential half-duplex pipeline controlled by a simple state machine:

```
LISTENING → PROCESSING_ASR → PROCESSING_LLM → PROCESSING_TTS → SPEAKING → LISTENING
```

Each turn captures one utterance, processes it through the pipeline, speaks the response, then returns to listening. The controller tracks timing at every stage for observability.

## Configuration

All defaults are in `app/config.py`. Key settings:

| Setting | Default | What it does |
|---------|---------|-------------|
| ASR model | `base.en` | Whisper model size (tiny/base/small) |
| LLM model | `gemma4:e4b` | Any Ollama model name |
| Max tokens | 48 | Caps LLM response length |
| VAD silence | 350ms | How long silence = end of speech |
| Max utterance | 6s | Caps recording length |

## Persona

The assistant personality is defined in `app/persona.py`. The default persona (Smruti) is tuned for live demos — conversational, expressive, and concise. You can switch to `neutral` or add your own style in the `PERSONA_STYLE_PROMPTS` dict.

## License

MIT
