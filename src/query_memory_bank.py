# src/query_memory_bank.py
import os, sys, json, pickle, argparse, datetime
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
DEFAULT_BANK = os.path.join(RESULTS_DIR, "memory_bank.pkl")

def resolve_path(p: str) -> str:
    """Return an existing path by checking common locations."""
    # as-given
    if os.path.exists(p):
        return p
    # relative to repo root
    alt = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(alt):
        return alt
    # results/<basename>
    alt2 = os.path.join(RESULTS_DIR, os.path.basename(p))
    return alt2  # may still not exist; caller will check

def _first_not_none(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _norm(X, eps=1e-12):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def load_bank(path):
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Memory bank not found: {path}")
    with open(path, "rb") as f:
        bank = pickle.load(f)
    if not isinstance(bank, dict):
        raise ValueError("Unsupported memory bank format (expected dict).")

    vectors = _first_not_none(bank, ["vectors", "embeddings", "X", "bank"])
    if vectors is None:
        raise ValueError("No vectors found (looked for keys: vectors/embeddings/X/bank).")
    X = np.asarray(vectors, dtype=np.float32)

    meta = _first_not_none(bank, ["meta", "metadata"])
    if meta is None:
        meta = [{} for _ in range(len(X))]
    else:
        meta = list(meta)
        if len(meta) < len(X):
            meta += [{} for _ in range(len(X) - len(meta))]
        elif len(meta) > len(X):
            meta = meta[:len(X)]

    return X, meta, path

def pick_query(args, X, meta):
    # priority: --vec > --mean-epoch > --id
    if args.vec:
        v = np.load(args.vec).astype(np.float32).reshape(-1)
        return v, None, f"vec:{os.path.basename(args.vec)}"
    if args.mean_epoch is not None:
        ep = int(args.mean_epoch)
        idx = [i for i, m in enumerate(meta) if m.get("epoch") == ep]
        if not idx:
            raise ValueError(f"No entries with epoch={ep}")
        v = X[idx].mean(axis=0)
        return v, None, f"mean_epoch={ep}"
    qid = 0 if args.id is None else int(args.id)
    if qid < 0 or qid >= len(X):
        raise IndexError(f"--id {qid} out of range [0, {len(X)-1}]")
    return X[qid], qid, f"id={qid}"

def filter_candidate_indices(meta, only_epochs):
    if not only_epochs:
        return None
    only_epochs = only_epochs.strip()
    ids = []
    if ":" in only_epochs:
        a, b = only_epochs.split(":")
        lo, hi = int(a), int(b)
        for i, m in enumerate(meta):
            e = m.get("epoch")
            if e is not None and lo <= e <= hi:
                ids.append(i)
    else:
        allowed = set(int(x) for x in only_epochs.split(",") if x.strip())
        for i, m in enumerate(meta):
            e = m.get("epoch")
            if e in allowed:
                ids.append(i)
    return set(ids)

def topk_neighbors(q, X, meta, k=5, exclude_id=None, only_ids=None, normalize=True):
    if normalize:
        q = _norm(q.reshape(1, -1))[0]
        Xn = _norm(X)
    else:
        Xn = X
    sims = Xn @ q
    order = np.argsort(-sims)
    out = []
    for idx in order:
        if exclude_id is not None and idx == exclude_id:
            continue
        if only_ids is not None and idx not in only_ids:
            continue
        out.append((idx, float(sims[idx]), meta[idx]))
        if len(out) >= k:
            break
    return out

def pretty_print(items):
    for i, (idx, cos, m) in enumerate(items, 1):
        print(f"{i:2d}. id={idx:5d}  cos={cos:0.4f}  meta={m}")

def main():
    ap = argparse.ArgumentParser(description="Query a single memory bank by cosine similarity.")
    ap.add_argument("--bank", default=DEFAULT_BANK, help="Path to memory bank (default: results/memory_bank.pkl).")
    ap.add_argument("--id", type=int, help="Use vector at this index as query (default 0).")
    ap.add_argument("--vec", help="Path to .npy vector as query (overrides --id).")
    ap.add_argument("--mean-epoch", type=int, help="Use mean vector of entries with meta['epoch']==N.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only-epochs", help="Restrict search to epochs: '1,3,5' or '2:7'.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable L2-normalization before cosine.")
    ap.add_argument("--save", action="store_true", help="Save results JSON in results/ folder.")
    args = ap.parse_args()

    X, M, resolved = load_bank(args.bank)
    print(f"Loaded {resolved} -> vectors: {X.shape}, meta: {len(M)}")

    q, qid, qsrc = pick_query(args, X, M)
    only_ids = filter_candidate_indices(M, args.only_epochs)
    knn = topk_neighbors(q, X, M, k=args.topk, exclude_id=qid, only_ids=only_ids, normalize=not args.no_normalize)

    print(f"\nQuery source: {qsrc} | dim={q.shape[0]} | normalize={not args.no_normalize}")
    print("\nTop-K neighbors:")
    pretty_print(knn)

    if args.save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {
            "bank_path": resolved,
            "query_source": qsrc,
            "normalize": not args.no_normalize,
            "topk": args.topk,
            "neighbors": [{"id": i, "cos": c, "meta": m} for i,c,m in knn],
            "only_epochs": args.only_epochs or ""
        }
        out_path = os.path.join(RESULTS_DIR, f"query_{ts}.json")
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
        print(f"\n✓ Saved JSON -> {out_path}")

if __name__ == "__main__":
    main()
