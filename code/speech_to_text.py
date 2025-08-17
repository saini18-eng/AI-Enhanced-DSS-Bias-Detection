import argparse
import os
import whisper

def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to text using local Whisper (no API key needed).
    Returns the full transcript text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # Load a light model for demo; use 'small' or 'medium' for better accuracy
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result.get("text", "").strip()

def main():
    ap = argparse.ArgumentParser(description="Transcribe meeting audio to text using local Whisper.")
    ap.add_argument("--audio", required=True, help="Path to audio file (e.g., .wav, .mp3)")
    ap.add_argument("--out", required=False, default=None, help="Path to save transcript .txt")
    args = ap.parse_args()

    text = transcribe(args.audio)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] Transcript saved to: {args.out}")
    else:
        print(text)

if __name__ == "__main__":
    main()
