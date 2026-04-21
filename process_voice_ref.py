#!/usr/bin/env python3
import argparse
import sys

from app.tts.voice_ref import process_voice_reference

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
    try:
        result = process_voice_reference(args.input, args.output, args.duration)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Processing: {args.input} -> {args.output}")
    print(f"Max duration: {args.duration}s")
    print(f"Command: {' '.join(result['command'])}")
    print(f"Saved to: {result['output_path']}")

    metadata = result.get("metadata", {})
    if metadata:
        duration_seconds = float(metadata.get("duration", 0) or 0)
        sample_rate = metadata.get("sample_rate", "?")
        channels = metadata.get("channels", "?")
        codec_name = metadata.get("codec_name", "?")
        print(f"Output: {duration_seconds:.1f}s, {sample_rate} Hz, {channels}ch, {codec_name}")


if __name__ == "__main__":
    main()
