import re
from typing import List
import nltk

def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.strip()
    # normalize spaces
    text = re.sub(r"\s+", " ", text)
    return text

def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    # Remove very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return sentences

def normalize_for_model(sentence: str) -> str:
    # Light normalization; keep case for readability but remove extra whitespace
    s = re.sub(r"\s+", " ", sentence)
    return s.strip()
