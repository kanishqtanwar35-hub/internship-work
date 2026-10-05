from summarizer import ask, summarize
from train_compare import compare, make_data


def test_ensembles_beat_single_tree():
    scores = compare(make_data(n=800), cv=3)
    assert scores["XGBoost"] > scores["Decision Tree"] - 0.02
    assert all(0.4 < s <= 1 for s in scores.values())


def test_summary_is_shorter_and_answers_question():
    doc = open(__import__("summarizer").__file__.replace("summarizer.py", "sample_report.txt")).read()
    assert len(summarize(doc)) < len(doc) / 2
    assert "steel" in ask(doc, "What caused the delay?").lower()
    assert "doesn't seem to cover" in ask(doc, "Who won the cricket match?")
