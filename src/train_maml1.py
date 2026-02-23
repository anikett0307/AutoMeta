# src/train_maml.py
# ---- Stable MAML+FiLM trainer (~87–90% parallel accuracy curves) ----

import os, sys, csv, time, random, pickle, contextlib, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Project path + imports
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from task_generator import EpisodeSampler
from maml_model import TaskCondNet, MAML_TE  # FiLM + Task Embedding

# ---- AMP: forward-only (mixed precision) ----
AMP_FWD = True
def amp_autocast():
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=AMP_FWD, dtype=torch.float16)
    except Exception:
        return contextlib.nullcontext()

# ---------------------------
# Optional MemoryBank (Step-4)
# ---------------------------
MB_AVAILABLE = False
MemoryBank = CosineSimRetriever = save_retriever_index = None
def _import_memory_bank():
    global MB_AVAILABLE, MemoryBank, CosineSimRetriever, save_retriever_index
    try:
        from utils.memory_bank import MemoryBank, CosineSimRetriever, save_retriever_index
        MB_AVAILABLE = True
    except Exception:
        MB_AVAILABLE = False
_import_memory_bank()

def _make_memory_bank():
    if not MB_AVAILABLE: return None, None
    try:
        mb = MemoryBank(dim=128)
        retr = CosineSimRetriever() if CosineSimRetriever else None
        return mb, retr
    except Exception:
        return None, None

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def label_smooth_ce(logits, targets, eps=0.05):
    num_classes = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    y = F.one_hot(targets, num_classes).float()
    y = (1.0 - eps) * y + eps / num_classes
    return -(y * logp).sum(dim=1).mean()

def append_csv_row(path, row, header=None):
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file and header: w.writerow(header)
        w.writerow(row)

def save_plot(history, path):
    df = pd.DataFrame(history)
    if len(df) == 0: return
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.title("Training & Validation Loss")
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2); plt.title("Training & Validation Accuracy")
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

@torch.no_grad()
def evaluate_on_val(maml, sampler, device, n_way, k_shot, q_query, episodes=300):
    maml.model.eval()
    losses, accs = [], []
    for _ in range(episodes):
        sx, sy, qx, qy = sampler.generate_task(n_way=n_way, k_shot=k_shot, q_query=q_query)
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
        adapted, task_e = maml.adapt(sx, sy)
        with amp_autocast():
            logits = adapted.logits(adapted.features(qx), task_e)
            loss = label_smooth_ce(logits, qy, eps=0.05)
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, qy))
    maml.model.train()
    return np.mean(losses), np.mean(accs)

# ---------------------------
# Main
# ---------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "train"))
    ap.add_argument("--val-root", default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "val"))
    ap.add_argument("--results", default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=12)
    ap.add_argument("--q-query", type=int, default=12)
    ap.add_argument("--inner-lr", type=float, default=0.03)
    ap.add_argument("--inner-steps", type=int, default=12)
    ap.add_argument("--meta-lr", type=float, default=8e-4)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--episodes-per-epoch", type=int, default=180)
    ap.add_argument("--target-acc", type=float, default=87.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--init-supervised", default="best_supervised1.pth")
    return ap

def main():
    args = build_argparser().parse_args()
    RESULTS_DIR = args.results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CSV_PATH = os.path.join(RESULTS_DIR, "metrics_maml.csv")
    PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_maml.png")
    BEST_CKPT = os.path.join(RESULTS_DIR, "best_maml.pth")
    LAST_CKPT = os.path.join(RESULTS_DIR, "last_maml.pth")

    assert torch.cuda.is_available(), "CUDA not available—run on GPU."
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    set_seed(42)

    # ---------------------------
    # Data and Model
    # ---------------------------
    train_sampler = EpisodeSampler(args.train_root, image_size=224, normalize=True)
    val_sampler = EpisodeSampler(args.val_root, image_size=224, normalize=True)
    net = TaskCondNet(n_way=args.n_way, te_dim=128).to(device)
    maml = MAML_TE(model=net, inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                   meta_lr=args.meta_lr, head_only=True).to(device)

    # Warm load pretrained supervised backbone
    INIT_SUP_CKPT = os.path.join(RESULTS_DIR, args.init_supervised)
    if os.path.isfile(INIT_SUP_CKPT):
        ck = torch.load(INIT_SUP_CKPT, map_location="cpu")
        sd = ck.get("model", ck)
        if hasattr(net.backbone, "features"):
            feat_keys = {k.replace("features.", ""): v for k, v in sd.items() if k.startswith("features.")}
            net.backbone.features.load_state_dict(feat_keys, strict=False)
            print("Loaded EfficientNet-B0 backbone from supervised stage.")

    mb, retriever = _make_memory_bank()
    scheduler = CosineAnnealingLR(maml.meta_optimizer, T_max=max(args.epochs//2, 1), eta_min=1e-5)

    # ---------------------------
    # Training loop
    # ---------------------------
    history, best_val, patience = {"epoch":[],"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}, -1, 0

    for ep in range(args.epochs):
        net.train()
        ep_losses, ep_accs = [], []
        pbar = trange(args.episodes_per_epoch, leave=False)

        i = 0
        while i < args.episodes_per_epoch:
            maml.meta_optimizer.zero_grad(set_to_none=True)
            steps_this_batch, meta_loss_accum, meta_accs, adapted_list = 0,0.0,[],[]
            for _ in range(args.meta_batch):
                if i >= args.episodes_per_epoch: break
                i += 1; steps_this_batch += 1
                sx, sy, qx, qy = train_sampler.generate_task(args.n_way, args.k_shot, args.q_query)
                sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
                adapted, task_e = maml.adapt(sx, sy)
                with amp_autocast():
                    logits = adapted.logits(adapted.features(qx), task_e)
                    loss = label_smooth_ce(logits, qy, eps=0.05)
                loss.backward()
                meta_loss_accum += loss.item()
                meta_accs.append(accuracy_from_logits(logits, qy))
                adapted_list.append(adapted)

            # Reptile head update
            w = 1.0 / max(1, len(adapted_list))
            for a in adapted_list:
                maml.accumulate_reptile_head(a, weight=w)

            clip_grad_norm_(maml.model.parameters(), args.grad_clip)
            maml.step()

            ep_losses.append(meta_loss_accum / max(1, steps_this_batch))
            ep_accs.append(float(np.mean(meta_accs)))
            pbar.update(steps_this_batch)

        scheduler.step()
        train_loss, train_acc = np.mean(ep_losses), np.mean(ep_accs)
        val_loss, val_acc = evaluate_on_val(maml, val_sampler, device, args.n_way, args.k_shot, args.q_query)

        if val_acc > best_val:
            best_val, patience = val_acc, 0
            torch.save({"model": net.state_dict(), "epoch": ep, "val_acc": best_val}, BEST_CKPT)
        else:
            patience += 1

        torch.save({"model": net.state_dict(), "opt": maml.meta_optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_val": best_val, "epoch": ep}, LAST_CKPT)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        append_csv_row(CSV_PATH, [ep, train_loss, train_acc, val_loss, val_acc],
                       ["epoch","train_loss","train_acc","val_loss","val_acc"] if ep==0 else None)
        save_plot(history, PLOT_PATH)

        print(f"Epoch {ep:02d} | Train {train_acc:.2f}% | Val {val_acc:.2f}% | Best {best_val:.2f}%")

        if val_acc >= args.target_acc:
            print(f"🎯 Target {args.target_acc:.1f}% reached. Stopping.")
            break
        if patience >= args.patience:
            print(f"⏹️ Early stop: no improvement for {args.patience} epochs.")
            break

    print("✅ MAML training complete.")
    print(f"• Metrics CSV:   {CSV_PATH}")
    print(f"• Metrics Plot:  {PLOT_PATH}")
    print(f"• Best Checkpoint: {BEST_CKPT}")
    print(f"• Last Checkpoint: {LAST_CKPT}")

if __name__ == "__main__":
    main()
