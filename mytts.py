import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from f5_tts.api import F5TTS


def main():
    parser = argparse.ArgumentParser(description="Voice-cloning TTS using F5-TTS")
    parser.add_argument("--text", "-t", help="Text to speak")
    parser.add_argument("--text-file", "-f", help="Read text from a file instead")
    parser.add_argument(
        "--ref", "-r", default="Recording.m4a",
        help="Reference audio file of your voice (default: Recording.m4a)",
    )
    parser.add_argument(
        "--ref-text", default="",
        help="Transcript of the reference audio (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out", "-o", default="output.wav",
        help="Output wav path (default: output.wav)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Run on CPU instead of GPU",
    )
    parser.add_argument(
        "--cfg-strength", type=float, default=2.0,
        help="Voice similarity vs fluency (higher=more like you, lower=more fluent, default: 2.0)",
    )
    parser.add_argument(
        "--nfe-step", type=int, default=32,
        help="Diffusion steps (higher=better quality but slower, default: 32)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech rate (default: 1.0)",
    )
    parser.add_argument(
        "--cross-fade", type=float, default=0.15,
        help="Crossfade between chunks in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--target-rms", type=float, default=0.1,
        help="Output volume level (default: 0.1, try 0.2 for louder)",
    )
    args = parser.parse_args()

    # Resolve text
    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.is_file():
            sys.exit(f"Text file not found: {text_path}")
        text = text_path.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text
    else:
        sys.exit("Provide either --text or --text-file")

    if not text:
        sys.exit("Text is empty")

    # Validate reference audio
    ref_audio = Path(args.ref)
    if not ref_audio.is_file():
        sys.exit(f"Reference audio not found: {ref_audio}")

    # Convert non-wav reference audio to wav using ffmpeg
    tmp_wav = None
    if ref_audio.suffix.lower() != ".wav":
        print(f"Converting {ref_audio.name} to wav...")
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ref_audio), "-ar", "24000", "-ac", "1", tmp_wav.name],
            check=True, capture_output=True,
        )
        ref_audio = Path(tmp_wav.name)

    device = "cpu" if args.no_gpu else "cuda"
    print(f"Loading F5-TTS model (device={device})...")
    tts = F5TTS(device=device)

    print(f"Generating speech ({len(text)} chars)...")
    tts.infer(
        ref_file=str(ref_audio),
        ref_text=args.ref_text,
        gen_text=text,
        file_wave=args.out,
        cfg_strength=args.cfg_strength,
        nfe_step=args.nfe_step,
        speed=args.speed,
        cross_fade_duration=args.cross_fade,
        seed=args.seed,
        target_rms=args.target_rms,
    )

    if tmp_wav:
        Path(tmp_wav.name).unlink(missing_ok=True)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
