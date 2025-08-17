from typing import List, Dict, Optional
import re

try:
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# Simple heuristic patterns indicating potentially biased phrasing
_HEURISTIC_PATTERNS = [
    r"\btoo emotional\b",
    r"\bnot a culture fit\b",
    r"\bfor a (woman|man)\b",
    r"\bdespite (being|her|his)\b",
    r"\b(aggressive|bossy) (for|in) (a )?woman\b",
    r"\bpregnant\b",
    r"\bfamily (man|woman)\b",
    r"\bwhere (he|she|they) (comes|come) from\b",
    r"\bstrong because he is assertive\b",
]

def heuristic_bias_score(text: str) -> float:
    """
    Returns a score in [0,1] based on presence of heuristic bias indicators.
    """
    text_l = text.lower()
    hits = 0
    for pat in _HEURISTIC_PATTERNS:
        if re.search(pat, text_l):
            hits += 1
    if hits == 0:
        return 0.15
    # Cap score; more hits => higher score
    return min(0.15 + 0.25 * hits, 0.99)

class BiasDetector:
    def __init__(self, hf_model: Optional[str] = None):
        self.hf_model = hf_model
        self.clf = None
        if hf_model and _HF_AVAILABLE:
            try:
                self.clf = pipeline("text-classification", model=hf_model, truncation=True)
            except Exception:
                self.clf = None

    def score(self, sentences: List[str]) -> List[Dict]:
        results = []
        if self.clf is not None:
            preds = self.clf(sentences)
            for sent, p in zip(sentences, preds):
                # Normalize label/score (assume label 'LABEL_1' is 'biased' if available)
                label = p.get("label", "").upper()
                score = float(p.get("score", 0.5))
                biased_prob = score if ("BIAS" in label or label.endswith("1") or "TOXIC" in label) else (1.0 - score)
                results.append({"text": sent, "bias_score": round(float(biased_prob), 4), "source": "hf"})
            return results

        # Fallback: heuristic
        for sent in sentences:
            results.append({"text": sent, "bias_score": round(heuristic_bias_score(sent), 4), "source": "heuristic"})
        return results
