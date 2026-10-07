"""
Enhanced Semantic Embedder with stopword filtering, subword n-grams,
and domain synonym expansion.
"""
import re
import numpy as np
from typing import List

STOPWORDS = {
    "what", "is", "the", "and", "which", "contains", "this", "information",
    "according", "to", "knowledge", "base", "a", "an", "of", "in", "for",
    "on", "at", "by", "with", "from", "as", "into", "through", "during",
    "including", "until", "against", "among", "throughout", "despite",
    "towards", "upon", "concerning", "to", "in", "for", "on", "by", "about",
    "can", "could", "should", "would", "do", "does", "did", "tell", "me", "show"
}

SYNONYM_GROUPS = [
    {"remote", "telework", "wfh", "home", "work from home", "remotely", "out of the office", "operate"},
    {"refund", "money back", "reimbursement", "return money", "reimburse"},
    {"staff", "employee", "employees", "workers", "team", "personnel", "members"},
    {"purchase", "buy", "buying", "bought", "transaction"},
    {"annual leave", "vacation", "pto", "holiday", "time off", "annual leave allowance", "leave allowance"},
    {"approval", "approving", "sign off", "authorize", "manager", "director"},
    {"sick", "illness", "medical", "hospital", "sick leave"},
    {"jet", "airplane", "aircraft", "flight", "aviation", "private jet"},
    {"transfer", "license", "credentials", "provision", "software license"},
    {"policy", "policies", "rules", "guidelines", "procedure", "sop", "window", "terms"}
]

class SemanticEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim
        np.random.seed(42)
        self.projection = np.random.randn(4096, self.dim).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-z0-9'-]+\b", text.lower())
        meaningful = [w for w in words if w not in STOPWORDS]
        return meaningful if meaningful else words

    def encode(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)

        # 1. Word and Subword N-grams (Typo tolerance)
        for token in tokens:
            idx = abs(hash(token)) % 4096
            vec += self.projection[idx] * 1.5

            if len(token) >= 3:
                for n in (3, 4):
                    for i in range(len(token) - n + 1):
                        ngram = token[i:i+n]
                        ng_idx = abs(hash(ngram)) % 4096
                        vec += self.projection[ng_idx] * 0.5

        # 2. Semantic Synonym Concept Injections
        text_lower = text.lower()
        for group_idx, group in enumerate(SYNONYM_GROUPS):
            for term in group:
                if term in text_lower:
                    concept_idx = (group_idx * 313) % 4096
                    vec += self.projection[concept_idx] * 3.5
                    break

        # 3. L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.zeros(self.dim, dtype=np.float32)

        return vec.tolist()
