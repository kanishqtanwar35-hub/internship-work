# RAG chatbots for a large enterprise client

## What it was
We built chatbots for a large enterprise client that answer questions from their own internal documents. The LLM is never trusted to answer from memory. It only gets the parts of the documents that are relevant to the question, and it has to answer from those.

## Why hybrid search
I started with plain vector (embedding) search and ran into a problem quickly: it's great at meaning but bad at exact things. Ask about a specific project code, a clause number or an amount and it often brings back something that's about the same topic but not the right document.

BM25 is the opposite. It's an old-school keyword ranking method that's very good at exact terms but doesn't understand that "flat price" and "apartment cost" mean the same thing.

So we used both and merged the results with **Reciprocal Rank Fusion (RRF)**. RRF only looks at where each chunk ranked in each list, not the raw scores, so you don't have to worry that BM25 scores and cosine similarity are on completely different scales. A chunk that ranks well in both lists ends up on top.

## The full flow
1. **Chunking**: documents are split into small pieces with some overlap, so a sentence that sits on a boundary isn't lost.
2. **Indexing**: every chunk goes into a BM25 index and a vector index.
3. **Search**: the question runs against both.
4. **Fusion**: RRF merges the two ranked lists.
5. **Re-ranking**: the top chunks get re-scored against the question to tighten the order.
6. **Answer**: the top chunks go into a prompt that tells the model to answer only from that context, cite the source, and say it doesn't know if the answer isn't there.

That last rule mattered a lot. A chatbot that confidently makes something up about a company policy is worse than one that says "I don't know".

## What's in this folder
`hybrid_retrieval.py` is my rebuild of the retrieval side. BM25 is written out by hand so it's easy to read. TF-IDF vectors stand in for the embedding model so it runs offline. It includes chunking, RRF and the "don't answer if nothing matches" check.

```bash
python hybrid_retrieval.py
```

One bug I hit while building this: the "don't know" check was getting fooled by filler words. A question like "what is on the cafeteria menu" was matching "is" and "on" in the IT policy and returning it as an answer. Filtering stopwords fixed it, and there's a test for it now.
