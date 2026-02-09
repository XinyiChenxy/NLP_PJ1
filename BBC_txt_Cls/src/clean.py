import re
from typing import Iterable, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_HTML_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9]+")

def base_clean(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _HTML_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def ml_clean(text: str) -> str:
    t = base_clean(text).lower()
    t = _NON_WORD_RE.sub(" ", t)
    tokens = [w for w in t.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

def batch_clean(texts: Iterable[str]) -> List[str]:
    return [ml_clean(t) for t in texts]
