"""Document summarizer + Q&A, done the map-reduce way.

The version at MaRS used an LLM for the summaries. This one is extractive
(it picks the most important sentences with TF-IDF) so it runs without an API
key, but the flow is the same: split -> summarize each part -> combine -> answer questions.
"""
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]


def split_sections(text, max_sentences=8):
    sents = sentences(text)
    return [sents[i:i + max_sentences] for i in range(0, len(sents), max_sentences)]


def top_sentences(sents, k):
    if len(sents) <= k:
        return sents
    m = TfidfVectorizer(stop_words="english").fit_transform(sents)
    scores = np.asarray(m.sum(axis=1)).ravel()
    keep = sorted(np.argsort(-scores)[:k])  # keep original order so it reads naturally
    return [sents[i] for i in keep]


def summarize(text, per_section=2, final=4):
    # map: summarize each section on its own (a long document won't fit the model in one go)
    partial = [s for sec in split_sections(text) for s in top_sentences(sec, per_section)]
    # reduce: summarize the summaries
    return " ".join(top_sentences(partial, final))


def ask(text, question, k=2):
    sents = sentences(text)
    vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
    sims = (vec.transform(sents) @ vec.transform([question]).T).toarray().ravel()
    if sims.max() == 0:
        return "The document doesn't seem to cover that."
    return " ".join(sents[i] for i in sorted(np.argsort(-sims)[:k]) if sims[i] > 0)


if __name__ == "__main__":
    doc = (Path(__file__).parent / "sample_report.txt").read_text(encoding="utf-8")
    print("SUMMARY:\n", summarize(doc))
    print("\nQ: What caused the delay?\nA:", ask(doc, "What caused the delay?"))
    print("\nQ: Who won the cricket match?\nA:", ask(doc, "Who won the cricket match?"))
