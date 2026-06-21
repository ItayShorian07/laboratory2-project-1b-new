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
normalized per query and averaged. This keeps the implementation general: the
public queries are used only for evaluation, not for lookup tables or
query-specific rules.

## Artifacts

The autograder does not rebuild the index, so these files must be committed:

| Path | Format | Purpose |
| --- | --- | --- |
| `artifacts/index_vectors.npy` | NumPy float32 array | Dense MiniLM page embeddings. |
| `artifacts/index_meta.json` | JSON | Page IDs and dense index metadata. |
| `artifacts/lexical_index.pkl.gz` | gzip-compressed pickle | BM25 postings, title postings, and document lengths. |

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
mean_ndcg@10=0.3916
query_phase_time=2.81s
```

## Submit

Submit a public GitHub repository with this code, the required `artifacts/`
files, and the presentation video link.
