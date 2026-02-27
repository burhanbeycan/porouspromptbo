from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_SNIPPETS_PATH = Path(__file__).resolve().parents[2] / "data" / "literature_snippets.jsonl"


@dataclass
class Snippet:
    sid: str
    source: str
    text: str


_RANGE_RE = re.compile(
    r"(?P<var>temperature|concentration)\s*(?P<lo>\d+(?:\.\d+)?)\s*(?:-|to)\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit>°C|C|M)",
    flags=re.IGNORECASE,
)


class SnippetCorpus:
    def __init__(self, snippets: List[Snippet]) -> None:
        self.snippets = snippets
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([s.text for s in snippets])

    @staticmethod
    def load(path: Optional[Path] = None) -> "SnippetCorpus":
        p = path or DEFAULT_SNIPPETS_PATH
        snippets: List[Snippet] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                snippets.append(Snippet(sid=obj["id"], source=obj["source"], text=obj["text"]))
        return SnippetCorpus(snippets)

    def search(self, query: str, top_k: int = 4) -> List[Snippet]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:top_k]
        return [self.snippets[i] for i in idx if scores[i] > 0]

    def extract_ranges(self, text: str) -> Dict[str, Tuple[float, float, str]]:
        out: Dict[str, Tuple[float, float, str]] = {}
        for m in _RANGE_RE.finditer(text):
            var = m.group("var").lower()
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
            unit = m.group("unit")
            out[var] = (lo, hi, unit)
        return out
