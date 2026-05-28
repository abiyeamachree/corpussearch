# corpussearch — Assignment 1: Retrieval, Honest Comparison

Public repo: *https://github.com/abiyeamachree/corpussearch*

This project compares three retrieval configurations on the same biomedical corpus and the same 20 labeled queries. The goal is a **defensible claim** about which setup fits this domain, with trade-offs stated explicitly.

---

## Why this assignment

I chose the retrieval-comparison brief because a lot of the projects I’ve worked on have involved finding useful information inside large amounts of messy or technical data. During previous AI engineering work, I built systems around document extraction, classification, and downstream searchability, so comparing retrieval methods felt like a natural extension of problems I’ve already dealt with in practice rather than a purely academic exercise.

I used the SciFact dataset because it contains short scientific claims and technical abstracts where relevant information is often phrased differently from the query itself. That creates a realistic retrieval problem where keyword search can fail on semantic phrasing, while dense retrieval can struggle with exact terminology. This made it a good dataset for evaluating the strengths and weaknesses of BM25, dense retrieval, and hybrid retrieval under the same conditions.

---

## What I built

| Piece | Choice |
|--------|--------|
| **Corpus** | [SciFact](https://huggingface.co/datasets/BeIR/scifact) — 400 real abstracts (subset of 5,183) |
| **Queries** | 20 hand-written queries in [`data/assignment_queries.json`](data/assignment_queries.json), each with gold `relevant_doc_ids` |
| **Hard queries** | 5 marked `hard: true` (paraphrase, colloquial wording, rare terms, ambiguous prognosis, numeric paraphrase) |
| **Configs** | BM25 · Dense (MiniLM + FAISS) · Hybrid (0.5/0.5 score fusion after min–max normalisation) |
| **Metrics** | Recall@5, MRR, p95 query latency (ms) |

The 400-document subset includes every gold document for our 20 queries plus 386 additional abstracts sampled with `seed=42`, so the index size meets the assignment’s 200–500 doc requirement while keeping distractors realistic.

---

## Decisions and alternatives I ruled out

### Corpus domain and size

- **Chosen:** SciFact biomedical abstracts, 400-doc subset.  
- **Ruled out:** Full 5,183-doc index (goes against the “small collection” brief); synthetic corpus (no external validity); Wikipedia/arXiv (less aligned with claim–evidence retrieval).  
- **Why:** SciFact gives real labels, short queries, and a natural lexical-vs-semantic split.

### Query set

- **Chosen:** 20 hand-authored queries with explicit gold IDs (not the full BEIR 300-query test split).  
- **Ruled out:** Using only official BEIR qrels without custom queries (does not satisfy the hand-written + 5 hard requirement); larger query sets (harder to debug failures).  
- **Why:** Small labeled set makes failure analysis inspectable and keeps grading criteria visible.

### Dense model

- **Chosen:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` on L2-normalised embeddings (cosine similarity).  
- **Ruled out:** Larger bi-encoders (e.g. `e5-large`, PubMedBERT) which have better quality but slower cold start and heavier RAM; OpenAI/Cohere APIs which are not reproducible offline.  
- **Why:** MiniLM is a standard CPU-friendly baseline; differences between methods show up clearly.

### Hybrid fusion

- **Chosen:** Min–max normalise BM25 and dense scores over the full 400-doc corpus, then `0.5 × BM25 + 0.5 × dense`.  
- **Ruled out:** Reciprocal rank fusion (RRF) — often better but adds another hyperparameter; cross-encoder reranker (bonus tier) has best quality but much slower per query; BM25-only candidate generation before dense (production pattern) was not implemented to keep the notebook minimal.  
- **Why:** Simple fusion is easy to explain and reproduce; weaknesses are interpretable.

### Evaluation protocol

- **Chosen:** Same 20 queries for all three configs; p95 latency measured over 5 repeats per query.  
- **Ruled out:** Different query sets per method; single-run latency (noisy).  

---

## Headline numbers

Produced by `make run` on a single Windows laptop (Python 3.13, CPU, no GPU). Full table also written to [`results.csv`](results.csv).

| Method | Recall@5 | MRR | p95 latency (ms) |
|--------|----------|-----|------------------|
| **Dense** | **0.90** | **0.842** | 12.4 |
| Hybrid | 0.90 | 0.767 | 17.9 |
| BM25 | 0.80 | 0.738 | 2.2 |

**Latency constraint:** max p95 = **17.9 ms** → **PASS** (well under 1,000 ms).

**Headline claim for this corpus and query set:** On our 400-doc SciFact slice and 20 labeled queries, **dense retrieval is the strongest configuration** for both Recall@5 and MRR. Hybrid matches dense on recall but ranks relevant documents lower on average (lower MRR). BM25 is the fastest baseline and still reaches 0.80 Recall@5 when queries overlap abstract terminology, but it lags on paraphrased hard queries.

Interactive exploration, plots, and per-query failure prints live in [`scifact_retrieval_evaluation.ipynb`](scifact_retrieval_evaluation.ipynb).

---

## Where the best config still loses

**Category: paraphrased or entailment-flipped biomedical claims (hard queries).** Even when dense leads overall, it can still miss or rank poorly on queries such as *“Zero-dimensional scaffold materials promote tissue induction”* (q16): the supporting abstract may state that such biomaterials **lack** inductive properties, while top results discuss biomaterials in a generic positive frame. Retrievers reward topical overlap, not whether the abstract **supports** the claim. Hybrid can make this worse when BM25 and dense disagree and fusion demotes a correct doc sitting at rank 4–5 in the dense list.

---

## One thing I would do differently with another week

**Add a two-stage pipeline:** retrieve top-100 with BM25 and top-100 with dense, fuse only that union, then **rerank top-20 with a small cross-encoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). That would likely fix hard-query ranking without scoring all 400 documents twice per query for hybrid, and it is closer to what I would ship in production. A secondary stretch goal would be a biomedical encoder (e.g. S-PubMedBert) to test whether the “dense wins” conclusion is model-specific.

---

## Reproduce on one machine

**Requirements:** Python 3.10+, 

```bash
pip install -r requirements.txt jupyter
jupyter notebook scifact_retrieval_evaluation.ipynb
```

Run all cells top-to-bottom. Generates `results_comparison.png` in the project root.

---

## Project layout

```
corpussearch/
├── README.md
├── requirements.txt
├── data/
│   └── assignment_queries.json
├── scifact_retrieval_evaluation.ipynb
├── results.csv              
└── results_comparison.png   
```

--

## References

- SciFact / BEIR: [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact)  
- BM25: `rank-bm25`  
- Dense: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- FAISS: `faiss-cpu`
