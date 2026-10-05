# Document summarizer and analyzer chatbot

## What it was
A chatbot where you upload a document, get a short summary, and then ask questions about it. At MaRS this was for long reports that people didn't have time to read end to end.

## How it works
The tricky part is long documents. You can't always fit the whole thing into the model in one go, so the summary is done in two steps:
1. **Map**: split the document into sections and summarize each section separately.
2. **Reduce**: summarize those section summaries into one final summary.

For questions, it finds the parts of the document that match the question and answers from those. If nothing matches, it says the document doesn't cover it instead of guessing.

## What's in this folder
The original used an LLM for the summaries. This version picks the most important sentences with TF-IDF instead, so it runs without an API key, but the map-reduce flow and the question answering are the same. `sample_report.txt` is a made-up site progress report to try it on.

```bash
python summarizer.py
```
