# src/detect_change_points.py
import os, sys, json, pickle, contextlib
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Optional libs
with contextlib.suppress(Exception):
    import ruptures as rpt

# Try BOCPD (Bayesian change-point detection) – make optional/robust
BOCPD_OK = False
try:
    import bayesian_changepoint_detection.offline_changepoint_detection as offcd
    import bayesian_changepoint_detection.offline_likelihoods as offl
    BOCPD_OK = True
except Exception:
    BOCPD_OK = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
METRICS_CSV  = os.path.join(RESULTS_DIR, "metrics_tr.csv")
MEMBANK_PKL  = os.path.join(RESULTS_DIR, "memory_bank.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------
# Utils
# -----------------------
def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if x is None or len(x) == 0 or win <= 1:
        return x
    win = min(win, len(x))
    w = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, w, mode="same")


def load_metrics(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    # Ensure required columns
    must = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    for m in must:
        if m not in df.columns:
            raise ValueError(f"metrics CSV missing column: {m}")
    return {
        "epoch":     df["epoch"].to_numpy(),
        "train_loss":df["train_loss"].to_numpy(dtype=float),
        "train_acc": df["train_acc"].to_numpy(dtype=float),
        "val_loss":  df["val_loss"].to_numpy(dtype=float),
        "val_acc":   df["val_acc"].to_numpy(dtype=float),
    }


def load_memory_bank(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Expected schema from earlier scripts: {"vectors": np.ndarray [T, D], "meta": List[dict]}
    vecs = data.get("vectors", None)
    meta = data.get("meta", None)
    if vecs is None or meta is None:
        return None
    return {"vectors": np.asarray(vecs), "meta": meta}


def epoch_means_from_bank(bank: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """Aggregate memory-bank vectors by epoch -> mean vector per epoch."""
    vecs = np.asarray(bank["vectors"])
    meta = bank["meta"]
    # robust normalize to avoid scale issues
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs = vecs / norms

    # group by meta['epoch']
    groups: Dict[int, list] = {}
    for v, m in zip(vecs, meta):
        e = int(m.get("epoch", -1))
        if e < 0:
            continue
        groups.setdefault(e, []).append(v)

    means = {}
    for e, vs in groups.items():
        arr = np.stack(vs, axis=0)
        mu  = arr.mean(axis=0)
        mu  = mu / (np.linalg.norm(mu) + 1e-8)
        means[e] = mu
    return means


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def embedding_drift_series(epoch_list: np.ndarray,
                           epoch_means: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Construct a drift series aligned to epoch_list.
    drift[t] = 1 - cos(mean[e_t], mean[e_{t-1}]); first epoch -> 0
    """
    out = []
    prev_vec = None
    for e in epoch_list:
        vec = epoch_means.get(int(e), None)
        if vec is None:
            out.append(np.nan)
            continue
        if prev_vec is None:
            out.append(0.0)
        else:
            out.append(1.0 - cosine(prev_vec, vec))
        prev_vec = vec if vec is not None else prev_vec
    # forward-fill then zeros for any gaps
    s = pd.Series(out).ffill().fillna(0.0).to_numpy()
    return s


# -----------------------
# Change-point routines
# -----------------------
def run_ruptures(series: np.ndarray,
                 model: str = "rbf",
                 pen: float = 3.0,
                 min_size: int = 5) -> List[int]:
    """
    Use ruptures PELT to find change points; returns sorted indices (end indices).
    """
    if "ruptures" not in sys.modules:
        return []
    algo = rpt.Pelt(model=model, min_size=min_size).fit(series)
    # pen = penalty strength; tweakable
    cps = algo.predict(pen=pen)
    # 'cps' includes last index (len+1). Remove it; convert to 0-based cut indices
    cps = [int(c) for c in cps if c < len(series)]
    return cps


def run_bocpd(series: np.ndarray,
              hazard: float = 1/100.0) -> List[int]:
    """
    Offline BOCPD using StudentT likelihood (if available).
    Returns list of changepoint indices (0-based).
    """
    if not BOCPD_OK:
        return []
    x = np.asarray(series, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-8)

    try:
        # Likelihood model: Student-T with weak priors
        like = offl.StudentT()  # new API needs no hyperparameters here
        # Posterior over run lengths
        Q, Pcp = offcd.offline_changepoint_detection(
            x,
            prior_func=lambda r: hazard,  # constant hazard
            observation_log_likelihood_function=like.log_likelihood,
            truncate=-40
        )
        # Extract MAP changepoints from Pcp
        # Pcp is (T x T), upper-triangular-ish; pick peaks on the last row
        p_last = np.exp(Pcp).sum(axis=0)  # marginal over run-ends
        # Heuristic: peaks above a percentile
        thresh = np.percentile(p_last, 95)
        cps = np.where(p_last >= thresh)[0].tolist()
        cps = [int(c) for c in cps if 0 < c < len(x)]
        cps = sorted(list(set(cps)))
        return cps
    except Exception:
        # If library signature differs, just skip BOCPD
        return []


# -----------------------
# Main
# -----------------------
def main():
    assert os.path.isfile(METRICS_CSV), f"Missing metrics file: {METRICS_CSV}"
    m = load_metrics(METRICS_CSV)
    epochs    = m["epoch"]
    train_acc = m["train_acc"]
    val_acc   = m["val_acc"]
    vloss     = m["val_loss"]
    tloss     = m["train_loss"]
    E         = len(epochs)
    print(f"Loaded metrics: E={E}, last epoch={int(epochs[-1])}")

    # 1) Smooth loss series (helps change-point stability)
    vloss_s = moving_average(vloss, win=7)
    tloss_s = moving_average(tloss, win=7)

    # 2) Build embedding-drift (optional, if memory bank exists)
    drift_series = None
    bank = load_memory_bank(MEMBANK_PKL)
    if bank is not None:
        means = epoch_means_from_bank(bank)
        drift_raw = embedding_drift_series(epochs, means)
        drift_series = moving_average(drift_raw, win=5)
        print("✓ Loaded embedding drift from memory_bank.pkl")

    # 3) Ruptures on smoothed val_loss
    cps_loss_rup = run_ruptures(vloss_s, model="rbf", pen=3.0, min_size=5)

    # 4) Ruptures on drift (if present)
    cps_drift_rup: List[int] = []
    if drift_series is not None:
        cps_drift_rup = run_ruptures(drift_series, model="rbf", pen=2.5, min_size=5)

    # 5) BOCPD on val_loss (optional)
    cps_loss_bocpd: List[int] = run_bocpd(vloss_s)

    # 6) Segment stats w.r.t val_loss change-points (use ruptures result as default)
    cut_idxs = sorted(set(cps_loss_rup))
    if len(cut_idxs) == 0 or cut_idxs[-1] != (E - 1):
        cut_idxs = cut_idxs + [E - 1]

    segments = []
    start = 0
    for ci in cut_idxs:
        end = int(ci)
        seg = {
            "start_epoch": int(epochs[start]),
            "end_epoch":   int(epochs[end]),
            "len":         int(end - start + 1),
            "mean_val_acc":float(np.mean(val_acc[start:end+1])),
            "mean_val_loss":float(np.mean(vloss[start:end+1])),
            "mean_train_acc":float(np.mean(train_acc[start:end+1])),
            "mean_train_loss":float(np.mean(tloss[start:end+1])),
        }
        segments.append(seg)
        start = end + 1
        if start >= E:
            break

    # 7) Save change-point summary
    out_json = os.path.join(RESULTS_DIR, "change_points.json")
    summary = {
        "ruptures_val_loss": cps_loss_rup,
        "ruptures_drift": cps_drift_rup,
        "bocpd_val_loss": cps_loss_bocpd,
        "epochs": E,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved:", out_json)

    # 8) Save segment stats
    seg_csv = os.path.join(RESULTS_DIR, "segment_stats.csv")
    pd.DataFrame(segments).to_csv(seg_csv, index=False)
    print("✓ Saved:", seg_csv)

    # 9) Save raw series (ALIGN LENGTHS FIRST to avoid mismatch)
    cols = {
        "epoch": epochs,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_error_smooth": vloss_s,
        "train_error_smooth": tloss_s,
    }
    if drift_series is not None:
        cols["drift_smooth"] = drift_series

    L = min(len(v) for v in cols.values())
    for k in list(cols.keys()):
        cols[k] = np.asarray(cols[k])[:L]

    drift_csv = os.path.join(RESULTS_DIR, "drift_series.csv")
    pd.DataFrame(cols).to_csv(drift_csv, index=False)
    print("✓ Saved:", drift_csv)

    print("Done.")


if __name__ == "__main__":
    main()
