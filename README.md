# AI-Enhanced Decision Support System (DSS) with Bias Detection

This repository accompanies the paper **"AI-Enhanced Boardroom Decision Support System with Integrated Bias Detection for Corporate Governance"** and includes a dataset roadmap and reference code.

## Repository Layout
```
DSS-Bias-Decision-Support/
├── code/                      # Reference Python code (pipeline & modules)
├── datasets/                  # Dataset roadmap (CSV/Excel)
├── paper/                     # Research paper (DOCX)
├── samples/                   # Sample transcript
├── reports/                   # Output reports (created on run)
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Quick Start
1. Create & activate a virtualenv
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. Install deps
   ```bash
   pip install -r requirements.txt
   # Install PyTorch CPU wheel for your OS if needed:
   # pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -c "import nltk; import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```
   *Whisper requires FFmpeg.* Install via package manager or from https://ffmpeg.org/

3. Run the end-to-end pipeline (text transcript)
   ```bash
   python code/pipeline.py --transcript samples/sample_transcript.txt --out reports/bias_report.csv
   ```
   Or transcribe audio first (requires FFmpeg):
   ```bash
   python code/pipeline.py --audio path/to/meeting_audio.wav --out reports/bias_report.csv
   ```

4. (Optional) Use a Hugging Face classifier
   ```bash
   python code/pipeline.py --transcript samples/sample_transcript.txt --hf_model unitary/toxic-bert
   ```

## Citation
If you use this repo, please cite the paper in `paper/`.

## License
MIT — see `LICENSE`. External datasets follow their own licenses.
