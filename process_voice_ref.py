#!/usr/bin/env python3
"""
Convert and preprocess audio for voice cloning reference.

Usage:
    python process_voice_ref.py input.webm -o voice_ref_clean.wav
    python process_voice_ref.py input.mp3 -o voice_ref_clean.wav --duration 15
"""

import subprocess
import argparse
import sys
from pathlib import Path


def process_audio(input_path: str, output_path: str, duration: int = 10):
    """
    Process audio for voice cloning reference.

    Processing steps:
    - High-pass filter at 80 Hz (removes room rumble, handling noise)
    - FFT-based noise reduction (cleans background hiss)
    - Trim silence from start and end
    - Normalize loudness to -16 LUFS
    - Convert to 24 kHz mono 16-bit PCM (pocket-tts native sample rate)
    - Trim to max duration (default 10s — pocket-tts works best with 5-15s)
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Audio filter chain:
    # 1. highpass: remove low-frequency rumble below 80 Hz
    # 2. afftdn: FFT-based noise reduction (noise floor -25 dB)
    # 3. silenceremove (forward + reverse): trim silence from both ends
    # 4. loudnorm: normalize loudness to broadcast standard
    af_filter = (
        "highpass=f=80,"
        "afftdn=nf=-25,"
        "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB,"
        "areverse,"
        "loudnorm=I=-16:LRA=11:TP=-1.5"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-af", af_filter,
        "-ar", "24000",         # 24 kHz — pocket-tts native sample rate
        "-ac", "1",             # Mono
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-t", str(duration),    # Trim to max duration
        str(output_path),
    ]

    print(f"Processing: {input_path} -> {output_path}")
    print(f"Max duration: {duration}s")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(f"Saved to: {output_path}")

    # Print info about the output file
    info_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", str(output_path)
    ]
    info_result = subprocess.run(info_cmd, capture_output=True, text=True)
    if info_result.returncode == 0:
        import json
        streams = json.loads(info_result.stdout).get("streams", [])
        if streams:
            s = streams[0]
            dur = float(s.get("duration", 0))
            sr = s.get("sample_rate", "?")
            ch = s.get("channels", "?")
            print(f"Output: {dur:.1f}s, {sr} Hz, {ch}ch, {s.get('codec_name', '?')}")


def main():
    parser = argparse.ArgumentParser(
        description="Process audio for voice cloning reference"
    )
    parser.add_argument("input", help="Input audio file (webm, mp3, wav, etc.)")
    parser.add_argument("-o", "--output", default="voice_ref_clean.wav",
                       help="Output WAV file (default: voice_ref_clean.wav)")
    parser.add_argument("--duration", type=int, default=10,
                       help="Maximum duration in seconds (default: 10)")

    args = parser.parse_args()
    process_audio(args.input, args.output, args.duration)


if __name__ == "__main__":
    main()
