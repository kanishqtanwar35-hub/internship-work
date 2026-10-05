from hybrid_retrieval import BM25, SAMPLE_DOCS, HybridRetriever, answer, chunk


def test_finds_right_policy():
    r = HybridRetriever(SAMPLE_DOCS)
    assert answer(r, "How many leaves can I carry forward?")["sources"][0] == "leave_policy.txt"
    assert "travel_policy.txt" in answer(r, "hotel limit per night")["sources"]


def test_declines_when_docs_dont_cover_it():
    r = HybridRetriever(SAMPLE_DOCS)
    assert answer(r, "What is on the cafeteria menu?")["sources"] == []


def test_bm25_prefers_exact_rare_term():
    s = BM25(["the VPN must be on", "the the the laptop"]).scores("VPN")
    assert s[0] > s[1]


def test_chunks_overlap():
    parts = chunk(" ".join(str(i) for i in range(100)), size=40, overlap=10)
    assert parts[0].split()[-10:] == parts[1].split()[:10]


def test_rrf_rewards_agreement():
    fused = HybridRetriever.rrf([[1, 2, 3], [1, 3, 2]])
    assert max(fused, key=fused.get) == 1
