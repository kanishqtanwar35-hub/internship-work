"""Hybrid retrieval: BM25 + vector search, merged with Reciprocal Rank Fusion.

Simplified version of the retrieval layer I worked on. The real one used a
proper embedding model and a vector DB. Here TF-IDF vectors stand in for the
embeddings so the whole thing runs offline with no API key.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on", "at", "for",
    "per", "and", "or", "i", "my", "me", "we", "can", "do", "does", "how", "what",
    "when", "where", "which", "who", "many", "much", "with", "by", "it", "this", "that",
}


def tokenize(text):
    # filler words like "is" or "per" would otherwise make unrelated chunks look relevant
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def chunk(text, size=60, overlap=15):
    """Split text into word chunks with some overlap so context isn't cut in half."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


class BM25:
    """Plain BM25, written out so it's easy to follow.

    k1: how fast a repeated word stops adding score
    b:  how much long chunks get penalised
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        # rare words get more weight than common ones
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                norm = 1 - self.b + self.b * len(d) / self.avgdl
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(s)
        return np.array(out)


@dataclass
class Hit:
    chunk_id: int
    text: str
    source: str
    score: float


class HybridRetriever:
    def __init__(self, documents, chunk_size=60, overlap=15):
        """documents: dict of {source_name: full_text}"""
        self.chunks, self.sources = [], []
        for name, text in documents.items():
            for c in chunk(text, chunk_size, overlap):
                self.chunks.append(c)
                self.sources.append(name)
        self.bm25 = BM25(self.chunks)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=sorted(STOPWORDS))
        self.matrix = self.vec.fit_transform(self.chunks)

    def _vector_scores(self, query):
        q = self.vec.transform([query])
        return (self.matrix @ q.T).toarray().ravel()

    @staticmethod
    def rrf(rankings, k=60):
        """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) across the lists.

        It only looks at rank position, so BM25 scores and cosine scores
        don't need to be on the same scale.
        """
        fused = Counter()
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] += 1.0 / (k + rank)
        return fused

    def search(self, query, top_k=3, min_bm25=0.5):
        bm = self.bm25.scores(query)
        vs = self._vector_scores(query)
        bm_rank = [i for i in np.argsort(-bm) if bm[i] > 0]
        vs_rank = [i for i in np.argsort(-vs) if vs[i] > 0]
        fused = self.rrf([bm_rank, vs_rank])
        # if almost nothing in the docs matches the question, don't pretend we found something
        if not fused or bm.max() < min_bm25:
            return []
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [Hit(int(i), self.chunks[i], self.sources[i], round(fused[i], 5)) for i in best]


def answer(retriever, question):
    """Builds the grounded prompt. In the real bot this went to the LLM.
    Here it returns the prompt and sources so you can see what the LLM would get."""
    hits = retriever.search(question)
    if not hits:
        return {"answer": "I don't have that information in the documents.", "sources": []}
    context = "\n\n".join(f"[{h.source}] {h.text}" for h in hits)
    prompt = (
        "Answer only from the context below. Cite the source in brackets. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return {"prompt": prompt, "sources": sorted({h.source for h in hits})}


SAMPLE_DOCS = {
    "leave_policy.txt": "Employees get 24 paid leaves per year. Unused leave up to 10 days can be "
                        "carried forward. Leave requests above 5 days need manager approval two weeks in advance.",
    "travel_policy.txt": "Domestic travel is booked through the travel desk. Hotel limit is Rs 6000 per night "
                         "in metro cities. Expense claims must be filed within 15 days with bills attached.",
    "it_policy.txt": "Laptops must be locked when unattended. VPN is mandatory on public wifi. "
                     "Report lost devices to IT within 24 hours.",
}

if __name__ == "__main__":
    r = HybridRetriever(SAMPLE_DOCS)
    for q in ["How many leaves can I carry forward?", "hotel limit per night", "What is on the cafeteria menu?"]:
        res = answer(r, q)
        print(f"{q:40s} -> {res.get('sources') or res['answer']}")
