"""Inspect reranker behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss
from core.index import load_index
from core.embed import embed_queries
from core.retrieve import _minmax_rows
from eval import ndcg_at_k, mean_ndcg_at_k
from utils import load_public_queries

POOL = 100


def main() -> None:
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [set(r["relevant_page_ids"]) for r in rows]

    idx = load_index()
    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)
    qv = embed_queries(queries)
    n_pages = idx.page_ids.shape[0]
    sims, ids = fa.search(qv, n_pages)
    dense = np.empty((len(queries), n_pages), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims
    lex = idx.bm25.score_batch(queries)
    fused = idx.alpha * _minmax_rows(dense) + (1 - idx.alpha) * _minmax_rows(lex)
    order = np.argsort(-fused, axis=1)[:, :POOL]
    cand_ids = idx.page_ids[order]
    cand_hyb = np.take_along_axis(fused, order, axis=1)

    ce = np.load(ROOT / "dev" / "cache" / "ce_cross-encoder__ms-marco-MiniLM-L-6-v2.npy")

    for k in [10, 20, 50, 100]:
        oracle = []
        for qi in range(len(queries)):
            inpool = [int(p) for p in cand_ids[qi, :k] if int(p) in gold[qi]]
            ranked = inpool + [-1] * 10
            oracle.append(ndcg_at_k(ranked, gold[qi]))
        print(f"oracle@10 within top-{k:>3} pool = {np.mean(oracle):.4f}")
    print()

    k = 20
    print(f"{'#':>2} {'hyb':>5} {'ce':>5} {'nGold':>5} {'inPool':>6}  query")
    hyb_s, ce_s = [], []
    for qi in range(len(queries)):
        ids_k = cand_ids[qi, :k]
        hy = cand_hyb[qi, :k]
        cs = ce[qi, :k]
        hyb_rank = [int(ids_k[i]) for i in np.argsort(-hy)[:10]]
        ce_rank = [int(ids_k[i]) for i in np.argsort(-cs)[:10]]
        nh = ndcg_at_k(hyb_rank, gold[qi])
        nc = ndcg_at_k(ce_rank, gold[qi])
        hyb_s.append(nh); ce_s.append(nc)
        inpool = sum(1 for p in ids_k if int(p) in gold[qi])
        flag = "  <-- CE worse" if nc < nh - 1e-6 else ("  ++ CE better" if nc > nh + 1e-6 else "")
        print(f"{qi:>2} {nh:>5.2f} {nc:>5.2f} {len(gold[qi]):>5} {inpool:>6}  {queries[qi][:50]}{flag}")
    print(f"\nmean hybrid={np.mean(hyb_s):.4f}  mean pureCE={np.mean(ce_s):.4f}")


if __name__ == "__main__":
    main()
