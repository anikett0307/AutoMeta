# src/tsne_memory_bank.py
import os, sys, pickle, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

def resolve_path(p):
    # try given path
    if os.path.exists(p):
        return p
    # try relative to repo root
    alt = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(alt):
        return alt
    # try results dir
    alt2 = os.path.join(RESULTS_DIR, os.path.basename(p))
    return alt2

def _first_not_none(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def load_bank(path):
    with open(path, "rb") as f:
        bank = pickle.load(f)

    if not isinstance(bank, dict):
        raise ValueError(f"Unsupported bank format in {path} (expected dict).")

    vectors = _first_not_none(bank, ["vectors", "embeddings", "X", "bank"])
    if vectors is None:
        raise ValueError(f"No vectors found in {path} (looked for keys: vectors/embeddings/X/bank).")
    vectors = np.asarray(vectors, dtype=np.float32)

    meta = _first_not_none(bank, ["meta", "metadata"])
    if meta is None:
        meta = [{} for _ in range(len(vectors))]
    else:
        meta = list(meta)
        # pad/truncate to match vectors length
        if len(meta) < len(vectors):
            meta += [{} for _ in range(len(vectors) - len(meta))]
        elif len(meta) > len(vectors):
            meta = meta[:len(vectors)]

    return vectors, meta

def run_tsne(vectors, seed=42, perplexity=30.0):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
        verbose=1,
    )
    return tsne.fit_transform(vectors)

def main():
    ap = argparse.ArgumentParser(description="t-SNE of task memory bank.")
    ap.add_argument("--bank", default=os.path.join(RESULTS_DIR, "memory_bank.pkl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "tsne_memory_bank.png"))
    args = ap.parse_args()

    bank_path = resolve_path(args.bank)
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Memory bank not found: {bank_path}")

    X, meta = load_bank(bank_path)
    print(f"Loaded {bank_path} | vectors: {X.shape} | meta: {len(meta)}")
    print(f"TSNE params: seed={args.seed}, perplexity={args.perplexity}")

    # Safety: perplexity must be < num_samples
    n = len(X)
    if args.perplexity >= n:
        p = max(5.0, min(50.0, n / 3.0))
        print(f"⚠️  Perplexity {args.perplexity} >= n={n}; using {p:.1f} instead.")
        args.perplexity = p

    emb = run_tsne(X, seed=args.seed, perplexity=args.perplexity)

    # Save arrays + meta
    np.save(os.path.join(RESULTS_DIR, "tsne_memory_bank.npy"), emb)
    with open(os.path.join(RESULTS_DIR, "tsne_meta.json"), "w") as f:
        json.dump(meta, f)

    # Plot colored by epoch (if present)
    epochs = np.array([m.get("epoch", -1) for m in meta])
    plt.figure(figsize=(7,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=epochs, cmap="viridis", s=12, alpha=0.85)
    plt.colorbar(sc, label="Epoch")
    plt.title("t-SNE of Task Memory Bank")
    plt.xlabel("t-SNE dim 1"); plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(f"Saved embeddings  -> {os.path.join(RESULTS_DIR, 'tsne_memory_bank.npy')}")
    print(f"Saved meta        -> {os.path.join(RESULTS_DIR, 'tsne_meta.json')}")
    print(f"Saved plot        -> {args.out}")

if __name__ == "__main__":
    main()
