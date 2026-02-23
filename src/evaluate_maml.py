# src/evaluate_maml.py
import os, sys, json, time, csv, contextlib, argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Project path + imports
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from task_generator import EpisodeSampler
from maml_model import TaskCondNet, MAML_TE

# ---- AMP: forward-only (match training) ----
AMP_FWD = True
def amp_autocast():
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=AMP_FWD, dtype=torch.float16)
    except Exception:
        try:
            from torch.amp import autocast as generic_autocast
            return generic_autocast(device_type="cuda", enabled=AMP_FWD, dtype=torch.float16)
        except Exception:
            return contextlib.nullcontext()

# ---------------------------
# Utils
# ---------------------------
def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def label_smooth_ce(logits, targets, eps=0.05):
    num_classes = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    y = F.one_hot(targets, num_classes).float()
    y = (1.0 - eps) * y + eps / num_classes
    return -(y * logp).sum(dim=1).mean()

def mean_conf_int(x, conf=0.95):
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    z = 1.96 if abs(conf - 0.95) < 1e-6 else 1.96
    half = z * s / np.sqrt(max(1, len(x)))
    return m, (m - half, m + half)

# ---------------------------
# Main evaluation
# ---------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate trained MAML on unseen tasks")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=12)
    parser.add_argument("--q-query", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=600, help="number of meta-test episodes")
    parser.add_argument("--test-root", default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "test"))
    parser.add_argument("--val-fallback", action="store_true",
                        help="allow using the VAL split if test/ doesn't exist")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--ckpt", default=None, help="path to checkpoint; defaults to best_tr.pth then last_tr.pth")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ----- device
    assert torch.cuda.is_available(), "CUDA not available—please run on GPU."
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    # ----- choose split
    data_root = args.test_root
    if not os.path.isdir(data_root):
        if args.val_fallback:
            data_root = os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "val")
            print(f"⚠️  test/ not found; falling back to VAL split: {data_root}")
        else:
            raise FileNotFoundError(f"Test split not found: {args.test_root}\n"
                                    f"(Tip: pass --val-fallback to use the validation split.)")

    # ----- sampler (no aug at test time)
    sampler = EpisodeSampler(data_root, image_size=224, normalize=True, support_aug=False)

    # ----- build + load model
    net  = TaskCondNet(n_way=args.n_way).to(device)
    maml = MAML_TE(model=net, inner_lr=0.03, inner_steps=12, meta_lr=8e-4, head_only=True).to(device)

    # checkpoint selection
    ckpt_path = args.ckpt
    if ckpt_path is None:
        cand1 = os.path.join(args.results_dir, "best_tr.pth")
        cand2 = os.path.join(args.results_dir, "last_tr.pth")
        ckpt_path = cand1 if os.path.isfile(cand1) else (cand2 if os.path.isfile(cand2) else None)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError("No checkpoint found. Expected results/best_tr.pth or results/last_tr.pth "
                                "or pass --ckpt PATH.")

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    net.load_state_dict(sd, strict=False)
    print(f"✓ Loaded checkpoint: {ckpt_path}")

    # ----- evaluation loop
    acc_ep, loss_ep, adapt_ms = [], [], []
    with torch.no_grad():
        net.eval()
        for ep in range(args.episodes):
            sx, sy, qx, qy = sampler.generate_task(n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            t0 = time.perf_counter()
            adapted, task_e = maml.adapt(sx, sy)
            t_adapt = (time.perf_counter() - t0) * 1000.0  # ms

            with amp_autocast():
                q_feats = adapted.features(qx)
                logits  = adapted.logits(q_feats, task_e)
                loss    = label_smooth_ce(logits, qy, eps=0.05)

            acc = accuracy_from_logits(logits, qy)
            acc_ep.append(acc); loss_ep.append(float(loss.item())); adapt_ms.append(t_adapt)

            if (ep + 1) % 50 == 0:
                print(f"[{ep+1}/{args.episodes}] acc={acc:.2f}%  adapt={t_adapt:.1f}ms")

    # ----- summarize
    acc_mean, acc_ci = mean_conf_int(acc_ep, 0.95)
    t_mean,  t_ci    = mean_conf_int(adapt_ms, 0.95)
    loss_mean = float(np.mean(loss_ep))

    summary = {
        "episodes": args.episodes,
        "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
        "accuracy_mean": round(acc_mean, 3),
        "accuracy_ci95": [round(acc_ci[0], 3), round(acc_ci[1], 3)],
        "loss_mean": round(loss_mean, 4),
        "adapt_ms_mean": round(t_mean, 2),
        "adapt_ms_ci95": [round(t_ci[0], 2), round(t_ci[1], 2)],
        "checkpoint": ckpt_path,
        "split_root": data_root,
    }

    print("\n====== Meta-Test Summary ======")
    print(json.dumps(summary, indent=2))

    # ----- save CSV of per-episode metrics
    csv_path = os.path.join(args.results_dir, "test_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "acc", "loss", "adapt_ms"])
        for i, (a, l, t) in enumerate(zip(acc_ep, loss_ep, adapt_ms)):
            w.writerow([i, a, l, t])
    print(f"• Episode metrics CSV -> {csv_path}")

    # ----- save summary JSON
    jpath = os.path.join(args.results_dir, "test_summary.json")
    with open(jpath, "w") as f: json.dump(summary, f, indent=2)
    print(f"• Summary JSON        -> {jpath}")

    # ----- plots
    df = pd.DataFrame({"acc": acc_ep, "loss": loss_ep, "adapt_ms": adapt_ms})
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.title("Episode Accuracy"); plt.plot(df["acc"]); plt.ylabel("%"); plt.xlabel("episode")
    plt.subplot(1,2,2); plt.title("Adaptation Time (ms)"); plt.plot(df["adapt_ms"]); plt.ylabel("ms"); plt.xlabel("episode")
    plt.tight_layout()
    ppath = os.path.join(args.results_dir, "test_curves.png")
    plt.savefig(ppath, dpi=150); plt.close()
    print(f"• Plots PNG            -> {ppath}")

    # Histogram view
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Acc histogram"); plt.hist(acc_ep, bins=20)
    plt.subplot(1,2,2); plt.title("Adapt(ms) histogram"); plt.hist(adapt_ms, bins=20)
    plt.tight_layout()
    hpath = os.path.join(args.results_dir, "test_hists.png")
    plt.savefig(hpath, dpi=150); plt.close()
    print(f"• Hists PNG            -> {hpath}")

if __name__ == "__main__":
    main()
