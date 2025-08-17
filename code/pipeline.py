import argparse
import os
import csv
from typing import List
from preprocess_text import split_sentences, normalize_for_model
from bias_detection import BiasDetector
from decision_support import recommend_for_sentence

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_csv(rows: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline: transcript -> bias scores -> recommendations")
    ap.add_argument("--audio", help="Path to audio file (optional). If provided, transcribe first.")
    ap.add_argument("--transcript", help="Path to transcript .txt (optional if --audio supplied).")
    ap.add_argument("--hf_model", help="Optional Hugging Face model id for classification", default=None)
    ap.add_argument("--out", help="Output CSV report path", default="reports/bias_report.csv")
    ap.add_argument("--recs_out", help="Output recommendations CSV path", default="reports/recommendations.csv")

    args = ap.parse_args()

    # Step 1: get transcript
    if args.audio and not args.transcript:
        # lazy import to avoid whisper dependency unless needed
        from speech_to_text import transcribe
        print("[i] Transcribing audio with Whisper...")
        text = transcribe(args.audio)
    else:
        if not args.transcript:
            raise SystemExit("Provide --transcript or --audio")
        text = read_text(args.transcript)

    # Step 2: preprocess + split into sentences
    print("[i] Splitting transcript into sentences...")
    sentences = split_sentences(text)
    norm_sentences = [normalize_for_model(s) for s in sentences]

    # Step 3: bias detection
    print("[i] Scoring sentences for potential bias...")
    detector = BiasDetector(hf_model=args.hf_model)
    scored = detector.score(norm_sentences)

    # Step 4: write bias report
    rows = [{"sentence": s["text"], "bias_score": s["bias_score"], "source": s["source"]} for s in scored]
    if not rows:
        raise SystemExit("No sentences extracted from transcript!")
    write_csv(rows, args.out)
    print(f"[OK] Bias report saved to: {args.out}")

    # Step 5: recommendations
    print("[i] Generating recommendations...")
    rec_rows = []
    for r in rows:
        rec = recommend_for_sentence(r["sentence"], r["bias_score"])
        rec_rows.append({
            "sentence": rec["original"],
            "bias_score": rec["bias_score"],
            "recommendations": " | ".join(rec["recommendations"])
        })
    write_csv(rec_rows, args.recs_out)
    print(f"[OK] Recommendations saved to: {args.recs_out}")

if __name__ == "__main__":
    main()
