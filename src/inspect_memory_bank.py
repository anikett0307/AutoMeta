# src/inspect_memory_bank.py
import os, sys, pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

# Try both filenames (use whichever your training wrote)
CANDIDATES = ["memory_bank.pkl", "task_bank.pkl"]

path = None
for name in CANDIDATES:
    p = os.path.join(RESULTS_DIR, name)
    if os.path.isfile(p):
        path = p
        break

if path is None:
    raise FileNotFoundError(f"Could not find any of {CANDIDATES} in {RESULTS_DIR}")

with open(path, "rb") as f:
    data = pickle.load(f)

vectors = data.get("vectors", None)
meta    = data.get("meta", [])

# Vectors may be a list or np.ndarray
if vectors is None:
    count, dim = 0, 0
else:
    vectors = np.asarray(vectors)
    # If it got saved as a list of 1D vectors, reshape
    if vectors.ndim == 1 and len(vectors) > 0 and hasattr(vectors[0], "__len__"):
        vectors = np.vstack(vectors)
    count = int(vectors.shape[0]) if vectors.ndim >= 1 else 0
    dim   = int(vectors.shape[1]) if vectors.ndim == 2 else 0

print("✅ Loaded:", path)
print(f"• vectors shape: {None if vectors is None else vectors.shape}")
print(f"• meta items:    {len(meta)}")
if count > 0:
    # quick sanity: norms and a small sample
    norms = np.linalg.norm(vectors, axis=1)
    print(f"• vector norms:  mean={norms.mean():.3f}, min={norms.min():.3f}, max={norms.max():.3f}")
    print("• first 2 meta:", meta[:2])
else:
    print("ℹ️ No vectors found yet (run a few episodes/epochs first).")
