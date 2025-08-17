import re
from typing import Dict

SUGGESTION_TEMPLATES = [
    ("too emotional", "Focus on observable behavior or metrics instead of subjective traits (e.g., 'evaluate based on conflict-resolution outcomes')."),
    ("not a culture fit", "Replace with explicit, job-relevant competencies (e.g., collaboration, reliability) and evidence."),
    ("for a woman", "Remove gendered comparison; evaluate all candidates with the same criteria."),
    ("pregnant", "Remove references to family or pregnancy; stick to professional qualifications."),
    ("where (he|she|they) come(s)? from", "Avoid references to origin/accent; focus on skills and performance."),
]

def recommend_for_sentence(sentence: str, bias_score: float) -> Dict:
    recs = []
    s_lower = sentence.lower()
    for patt, tip in SUGGESTION_TEMPLATES:
        if re.search(patt, s_lower):
            recs.append(tip)
    # Thresholding
    if bias_score >= 0.7 and not recs:
        recs.append("Rephrase to remove subjective or identity-related language; add evidence (KPIs, milestones).")
    elif bias_score >= 0.4 and not recs:
        recs.append("Consider clarifying the criterion with measurable indicators.")
    return {
        "original": sentence,
        "bias_score": round(bias_score, 4),
        "recommendations": recs or ["No change needed; keep language evidence-based."]
    }
