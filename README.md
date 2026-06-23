# Section B - Retrieval Pipeline

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

The corpus lives at **`data/Wikipedia Entries/`** in the handout. The submitted
repository only needs the prebuilt files under `artifacts/` for query-time
retrieval.

## Method

The pipeline combines two simple retrieval signals:

1. Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
2. Lexical BM25 retrieval over the full page text.

At query time, both methods return candidate pages. Their scores are min-max
normalized per query and blended. The code is organized under `core/` by
pipeline stage: chunking, embedding, indexing, lexical retrieval, fusion, and
query-time service. The optional reranker code is present but disabled in the
submitted config, keeping the final run simple and stable.

## Artifacts

The autograder does not rebuild the index, so these files must be committed:

| Path | Format | Purpose |
| --- | --- | --- |
| `artifacts/page_vectors.npy` | NumPy float32 array | Dense MiniLM page embeddings. |
| `artifacts/page_ids.npy` | NumPy int64 array | Page IDs aligned to vector rows. |
| `artifacts/page_texts.npy` | NumPy object array | Truncated page text used by optional reranking. |
| `artifacts/bm25_index.npz` | NumPy archive | BM25 postings and weights. |
| `artifacts/bm25_index.meta.json` | JSON | BM25 vocabulary and metadata. |
| `artifacts/retrieval_config.json` | JSON | Fusion and optional reranker settings. |

## Build Index

Run once locally to create `artifacts/`. This step is offline and not timed by
the grader.

```bash
python scripts/build_index.py
```

## Public Self-Test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

Current public result after the query-set fix:

```text
public_queries=29
mean_ndcg@10=0.4467
query_phase_time=0.36s
```

## Submit

Submit a public GitHub repository with this code, the required `artifacts/`
files, and the presentation video link.
