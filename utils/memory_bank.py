# utils/memory_bank.py
# Minimal, compatible Memory Bank for Step-4
from __future__ import annotations
import os, pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    # Optional speed-up; falls back to brute-force if not present
    from sklearn.neighbors import NearestNeighbors
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


class MemoryBank:
    """
    Simple in-memory vector store with optional ring buffer.
    Compatible signatures:
        MemoryBank(dim=128)
        MemoryBank(128)
        MemoryBank()

    Fields:
        vectors: list[np.ndarray]  (float32, shape [dim])
        meta:    list[dict|Any]    (optional metadata per vector)
    """
    def __init__(self, dim: Optional[int] = 128, max_items: int = 20000):
        if isinstance(dim, bool):  # guard weird calls
            dim = 128
        self.dim: Optional[int] = dim
        self.max_items = int(max_items)
        self.vectors: List[np.ndarray] = []
        self.meta: List[Any] = []

    # allow alternate positional constructor MemoryBank(128)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def _ensure_dim(self, vec: np.ndarray):
        if self.dim is None:
            self.dim = int(vec.shape[0])
        if int(vec.shape[0]) != int(self.dim):
            raise ValueError(f"[MemoryBank] Dim mismatch: got {vec.shape[0]}, expected {self.dim}")

    def add(self, vector: np.ndarray, meta: Optional[Any] = None) -> None:
        """Add a single vector with optional metadata."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        self._ensure_dim(v)
        self.vectors.append(v)
        self.meta.append(meta)
        # ring buffer if over capacity
        if len(self.vectors) > self.max_items:
            overflow = len(self.vectors) - self.max_items
            if overflow > 0:
                self.vectors = self.vectors[overflow:]
                self.meta    = self.meta[overflow:]

    def size(self) -> int:
        return len(self.vectors)

    def as_matrix(self) -> np.ndarray:
        if not self.vectors:
            d = int(self.dim or 0)
            return np.zeros((0, d), dtype=np.float32)
        return np.vstack(self.vectors).astype(np.float32, copy=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"dim": int(self.dim or 0),
                 "vectors": self.as_matrix(),
                 "meta": self.meta},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(path: str) -> "MemoryBank":
        with open(path, "rb") as f:
            data = pickle.load(f)
        mb = MemoryBank(dim=int(data.get("dim", 0)) or None)
        V = np.asarray(data.get("vectors", np.zeros((0, mb.dim or 0), np.float32)), dtype=np.float32)
        mb.dim = int(V.shape[1]) if V.size else (mb.dim or 128)
        mb.vectors = [row.copy() for row in V]
        mb.meta = list(data.get("meta", [None] * len(mb.vectors)))
        return mb


class CosineSimRetriever:
    """
    Lightweight cosine similarity retriever.
    If scikit-learn is available, uses NearestNeighbors(metric='cosine').
    Otherwise, falls back to brute-force cosine similarities.
    """
    def __init__(self, use_sklearn: Optional[bool] = None):
        self.use_sklearn = (_HAVE_SK if use_sklearn is None else bool(use_sklearn))
        self.nn = None
        self.X: Optional[np.ndarray] = None  # [N, D] float32

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        self.X = X
        if self.use_sklearn and X.shape[0] > 0:
            # n_neighbors will be specified at query time (can be <= n_samples)
            self.nn = NearestNeighbors(metric="cosine")
            self.nn.fit(X)
        else:
            self.nn = None  # brute-force path

    def query(self, q: np.ndarray, topk: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            q: [D] vector (float32)
            topk: number of results to return
        Returns:
            idxs: [K] integer indices into fitted matrix rows
            sims: [K] cosine similarities (float32), higher is more similar
        """
        assert self.X is not None, "Retriever not fitted. Call fit(X) first."
        if self.X.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        k = int(min(max(topk, 1), self.X.shape[0]))

        if self.nn is not None:
            dists, idxs = self.nn.kneighbors(q, n_neighbors=k, return_distance=True)
            sims = 1.0 - np.asarray(dists[0], dtype=np.float32)
            return np.asarray(idxs[0], dtype=np.int64), sims

        # brute force cosine
        X = self.X
        qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        Xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        sims_all = (X @ q.T).reshape(-1) / (Xn.reshape(-1) * qn.reshape(-1))
        idxs = np.argpartition(-sims_all, k-1)[:k]
        # sort topk by similarity descending
        order = np.argsort(-sims_all[idxs])
        idxs = idxs[order]
        sims = sims_all[idxs].astype(np.float32)
        return idxs.astype(np.int64), sims


def save_retriever_index(mb: MemoryBank, retriever: Optional[CosineSimRetriever], path: str) -> None:
    """
    Saves the memory bank vectors/meta AND a fitted retriever (if provided) to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "dim": int(mb.dim or 0),
        "vectors": mb.as_matrix(),
        "meta": mb.meta,
    }
    if retriever is not None and retriever.X is not None:
        payload["retriever_use_sklearn"] = bool(retriever.use_sklearn)
        payload["retriever_X_shape"] = tuple(retriever.X.shape)
        # retriever can be re-fit at load time from vectors; we store a flag
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = ["MemoryBank", "CosineSimRetriever", "save_retriever_index"]
