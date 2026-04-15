# AGENT.md

## Project

Simple local demo of a real-time interaction bot on a Mac mini.

Goal: build a voice loop that feels interactive enough for a demo:

`Mic -> ASR -> LLM -> TTS -> Speaker`

This is not a production system. It is a fast, local proof-of-concept meant to validate flow, latency, and user experience.

---

## Core Objective

Build a local assistant that can:

* listen from a microphone
* detect when the user is speaking
* transcribe speech into text
* generate a short response with an LLM
* synthesize that response into speech
* play the response back through speakers
* repeat continuously

Success means:

* one person can talk to it naturally
* response feels reasonably quick
* system runs reliably on a Mac mini
* setup is simple enough to demo repeatedly

---

## Non-Goals

For this demo, do **not** optimize for:

* many concurrent users
* cloud deployment
* telephony
* multilingual complexity at first
* perfect voice cloning
* agent tools, web search, memory, or RAG
* full duplex conversation from day one

If we chase all of that now, the demo will collapse under its own ambition.

---

## Philosophy

This demo should be built with the following priorities:

1. **Stability over sophistication**
2. **Low latency over maximum model quality**
3. **Short responses over long responses**
4. **Simple architecture over clever architecture**
5. **Observable pipeline over black-box magic**

The first version should feel alive, not impressive on paper.

---

## Target User Experience

The bot should behave like this:

1. user speaks into mic
2. bot waits until speech segment completes
3. ASR converts speech to text
4. LLM generates a concise reply
5. TTS speaks that reply aloud
6. bot goes back to listening

Desired demo behavior:

* response starts within roughly 1–2 seconds after the user stops speaking
* replies are short and clear
* bot avoids rambling
* interruption handling can be added later

---

## Hardware Assumptions

### Required

* Mac mini
* external microphone or USB headset
* speaker or headphones

### Recommended

* USB headset for easiest echo control
* quiet room
* stable power

### Notes

Mac mini is fine as the local compute box for a simple single-user demo. Do not overcomplicate hardware initially.

---

## High-Level Architecture

```text
Audio Input
  -> VAD / speech segmentation
  -> ASR
  -> Dialogue controller
  -> LLM
  -> TTS
  -> Audio Output
```

### Modules

#### 1. Audio Input

Captures microphone audio continuously.

Responsibilities:

* read PCM frames from microphone
* buffer incoming audio
* pass frames to VAD

#### 2. VAD / Speech Segmentation

Determines when the user starts and stops speaking.

Responsibilities:

* detect speech start
* detect speech end
* send complete utterance chunk to ASR

#### 3. ASR

Turns speech into text.

Responsibilities:

* transcribe utterance
* return plain text
* optionally return confidence/timestamps later

#### 4. Dialogue Controller

Coordinates pipeline state.

Responsibilities:

* decide when bot is listening vs speaking
* pass transcript to LLM
* enforce short response policy
* prevent overlapping calls accidentally

#### 5. LLM

Generates short assistant response.

Responsibilities:

* accept user transcript
* generate concise spoken-style reply
* keep context small

#### 6. TTS

Synthesizes audio response.

Responsibilities:

* convert reply text to waveform
* return playable audio chunk

#### 7. Audio Output

Plays bot speech.

Responsibilities:

* send waveform to speakers
* block input loop only if required in v1

---

## Initial Product Strategy

We should build this in **three stages**.

### Stage 1 — Sequential demo

Simplest possible flow:

* press enter or auto-trigger after speech end
* ASR on full utterance
* LLM generates response
* TTS speaks response
* return to listening

No streaming yet.

This is the correct starting point.

### Stage 2 — Better interactivity

Improve user experience with:

* automatic VAD-based end-of-speech detection
* shorter LLM answers
* reduced latency
* cleaner audio playback

Still keep pipeline mostly sequential if needed.

### Stage 3 — Partial streaming improvements

Only after Stage 1 and 2 are stable:

* partial ASR
* sentence-wise TTS
* interruption / cancel current playback

Streaming is useful, but it is not where we should start.

---

## Recommended First Model Strategy

Use the fastest reasonable models, not the fanciest ones.

### ASR

Use a Mac-friendly ASR runtime.

Desired qualities:

* runs locally on Apple Silicon
* stable enough for live microphone input
* good enough English or chosen demo language

Selection principle:

* choose the model that gives acceptable transcription with low delay
* do not start with a huge multilingual research model for this demo

### LLM

Use a small instruct model that can run locally with decent speed.

Desired qualities:

* conversational
* concise
* low memory footprint
* fast first-token latency

Selection principle:

* 3B to 7B class is sensible for local demo
* keep prompt tiny

### TTS

Use the fastest TTS that sounds acceptable.

Desired qualities:

* low startup latency
* easy local inference
* stable output

Selection principle:

* avoid very heavy autoregressive TTS in v1 unless already well-optimized
* a simple good-enough voice is better than a beautiful slow voice

---

## Runtime Design

The system should maintain a small state machine.

### States

* `LISTENING`
* `PROCESSING_ASR`
* `PROCESSING_LLM`
* `PROCESSING_TTS`
* `SPEAKING`
* `ERROR`

### Rules

* while speaking, do not start a new response in v1
* after speaking completes, return to listening
* if ASR result is empty, return to listening
* if any stage fails, log error and recover gracefully

This keeps the demo understandable and debuggable.

---

## Conversation Policy

The bot should be optimized for spoken interaction, not essay writing.

### System behavior

* keep replies under 2–3 sentences
* prefer clear and direct language
* avoid markdown, lists, and long reasoning
* ask follow-up questions only when useful
* sound like a demo assistant, not a lecturer

### Example instruction style for LLM

> You are a simple local voice assistant. Reply briefly, naturally, and clearly. Keep answers short enough to speak aloud comfortably. Avoid bullet points and avoid long explanations.

---

## Logging Requirements

Every run should expose timing.

Track at minimum:

* VAD end-of-speech time
* ASR latency
* LLM latency
* TTS latency
* total turnaround time

Also log:

* transcript text
* reply text
* errors

Without this, we will not know where the system is actually slow.

---

## Suggested Repository Structure

```text
voice-bot-demo/
├── AGENT.md
├── README.md
├── requirements.txt
├── .env
├── app/
│   ├── main.py
│   ├── config.py
│   ├── state.py
│   ├── audio/
│   │   ├── input.py
│   │   ├── output.py
│   │   └── vad.py
│   ├── asr/
│   │   └── engine.py
│   ├── llm/
│   │   └── engine.py
│   ├── tts/
│   │   └── engine.py
│   ├── orchestration/
│   │   └── controller.py
│   └── utils/
│       ├── logging.py
│       └── timers.py
└── logs/
```

Keep module boundaries clean from the start.

---

## Implementation Plan

## Phase 0 — Environment setup

Deliverable:

* local Python environment runs cleanly on Mac mini
* audio input and output devices detected

Tasks:

* set up project directory
* create virtual environment
* install dependencies
* verify microphone capture
* verify speaker playback

Exit criteria:

* record audio from mic and play test audio successfully

---

## Phase 1 — Audio loop

Deliverable:

* app continuously captures audio
* VAD identifies one user utterance

Tasks:

* implement microphone capture
* implement rolling buffer
* integrate VAD
* detect speech start and speech end
* save captured utterance for debugging

Exit criteria:

* spoken utterance is correctly segmented and stored as one audio clip

---

## Phase 2 — ASR integration

Deliverable:

* utterance audio becomes text

Tasks:

* load local ASR engine
* pass segmented audio to ASR
* print transcription to terminal
* handle empty or noisy results

Exit criteria:

* user can speak simple prompts and get usable text transcription

---

## Phase 3 — LLM integration

Deliverable:

* transcript becomes short response text

Tasks:

* load local LLM
* define system prompt for short spoken answers
* pass ASR text into LLM
* print reply text
* cap output length

Exit criteria:

* bot produces concise answer text consistently

---

## Phase 4 — TTS integration

Deliverable:

* reply text becomes audible output

Tasks:

* load local TTS engine
* synthesize response audio
* play it through system output
* clean temporary audio handling if needed

Exit criteria:

* full voice round-trip works end to end

---

## Phase 5 — Controller and polish

Deliverable:

* one command runs full demo loop reliably

Tasks:

* create state machine
* connect all modules
* add timing logs
* add graceful error handling
* add config file for model paths and thresholds

Exit criteria:

* demo can run repeatedly without manual patching between turns

---

## Latency Targets

For demo quality, these are reasonable goals:

* end-of-speech detection: under 300 ms after user stops
* ASR: under 700 ms for short utterances
* LLM first complete response text: under 1000 ms
* TTS generation start: under 700 ms
* total response start after user finishes: ideally 1–2 seconds

These are goals, not rigid promises.

---

## Risks

### 1. Audio device issues

Mac audio device selection can be annoying.

Mitigation:

* use fixed device config
* prefer USB headset

### 2. Echo / feedback

Speaker output may leak into microphone.

Mitigation:

* use headset for v1 demo
* later add echo control if needed

### 3. Model latency too high

Large models will make the bot feel dead.

Mitigation:

* start with smaller models
* shorten max generation lengths

### 4. TTS startup delay

Some TTS pipelines have poor first-audio latency.

Mitigation:

* use short outputs
* preload model
* test multiple TTS backends early

### 5. Too much architecture too early

Trying to build production design for a demo wastes time.

Mitigation:

* keep v1 sequential and local

---

## Development Rules

### Rule 1

Do not optimize before end-to-end loop exists.

### Rule 2

Do not add streaming before sequential version works reliably.

### Rule 3

Do not add memory, tools, RAG, or web features in the first demo.

### Rule 4

Always measure latency before making claims.

### Rule 5

Each module must be replaceable without rewriting the full stack.

---

## Minimum Demo Definition

The demo is considered successful when:

* user speaks one question into mic
* system transcribes it correctly enough
* system produces a short answer
* system speaks the answer back
* average turnaround feels acceptable for a live demo
* the demo can be repeated several times without crashing

That is enough.

Anything beyond this is version two.

---

## Immediate Next Actions

1. create repo and folder structure
2. verify mic input and speaker output on Mac mini
3. choose one ASR backend
4. choose one local LLM backend
5. choose one fast TTS backend
6. implement sequential loop first
7. measure latency at each stage

---

## Decision Heuristics

When stuck between two options, choose the one that is:

* simpler to run locally
* easier to debug
* faster to first working demo
* less magical

For this project, boring is good.

---

## Final Note

This project should not begin with the question:

> How do we build the perfect real-time voice agent?

It should begin with:

> How do we make one simple conversation work reliably on a Mac mini?

That is the correct first victory.

