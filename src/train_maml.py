# src/train_maml.py
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

# ---- AMP: forward-only (no GradScaler) ----
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
# Optional MemoryBank (step-4 storage)
# ---------------------------
MB_AVAILABLE = False
MemoryBank = CosineSimRetriever = save_retriever_index = None

def _import_memory_bank():
    global MB_AVAILABLE, MemoryBank, CosineSimRetriever, save_retriever_index
    try:
        from utils.memory_bank import MemoryBank, CosineSimRetriever, save_retriever_index
        MB_AVAILABLE = True; return
    except Exception:
        try:
            import importlib.util
            mb_path = os.path.join(PROJECT_ROOT, "utils", "memory_bank.py")
            spec = importlib.util.spec_from_file_location("memory_bank_fallback", mb_path)
            mod = importlib.util.module_from_spec(spec); assert spec and spec.loader
            spec.loader.exec_module(mod)
            MemoryBank = mod.MemoryBank
            CosineSimRetriever = getattr(mod, "CosineSimRetriever", None)
            save_retriever_index = getattr(mod, "save_retriever_index", None)
            MB_AVAILABLE = True
            print("✅ Loaded MemoryBank via path fallback.")
        except Exception as e:
            print(f"⚠️  MemoryBank import failed ({e}). Continuing without Step-4 storage.")
            MB_AVAILABLE = False

def _make_memory_bank():
    if not MB_AVAILABLE: return None, None
    for ctor in (lambda: MemoryBank(dim=128),
                 lambda: MemoryBank(128),
                 lambda: MemoryBank()):
        try:
            mb = ctor()
            retr = CosineSimRetriever() if CosineSimRetriever else None
            return mb, retr
        except TypeError:
            continue
        except Exception as e:
            print(f"⚠️  MemoryBank creation attempt failed: {e}")
            continue
    print("⚠️  Could not construct MemoryBank with any known signature. Disabling.")
    return None, None

_import_memory_bank()

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
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
        if new_file and header:
            w.writerow(header)
        w.writerow(row)

def save_plot(history, path):
    df = pd.DataFrame(history)
    if len(df) == 0: return
    plt.figure(figsize=(12,5))
    # loss
    plt.subplot(1,2,1); plt.title("Loss over Epochs")
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    if df["val_loss"].notna().any(): plt.plot(df["epoch"], df["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    # acc
    plt.subplot(1,2,2); plt.title("Accuracy over Epochs")
    plt.plot(df["epoch"], df["train_acc"], label="Train")
    if df["val_acc"].notna().any(): plt.plot(df["epoch"], df["val_acc"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

@torch.no_grad()
def evaluate_on_val(maml, sampler, device, n_way, k_shot, q_query, episodes=200):
    """Episodic validation with adaptation-time measurement."""
    maml.model.eval()
    losses, accs, adapt_ms = [], [], []
    for _ in range(episodes):
        sx, sy, qx, qy = sampler.generate_task(n_way=n_way, k_shot=k_shot, q_query=q_query)
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

        # measure adaptation time on GPU precisely
        start = time.perf_counter()
        adapted, task_e = maml.adapt(sx, sy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        adapt_ms.append((time.perf_counter() - start) * 1000.0)

        with amp_autocast():
            q_feats = adapted.features(qx)
            logits  = adapted.logits(q_feats, task_e)
            loss    = label_smooth_ce(logits, qy, eps=0.05)
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, qy))
    maml.model.train()
    return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(adapt_ms))

# ---------------------------
# Main
# ---------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Meta-training (MAML+FiLM) with episodic validation.")
    # data / device
    ap.add_argument("--train-root", default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "train"))
    ap.add_argument("--val-root",   default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "val"))
    ap.add_argument("--results",    default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    # episodic params
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=12)
    ap.add_argument("--q-query", type=int, default=12)
    ap.add_argument("--episodes-per-epoch", type=int, default=180)
    ap.add_argument("--val-episodes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    # inner/meta
    ap.add_argument("--inner-lr", type=float, default=0.03)
    ap.add_argument("--inner-steps", type=int, default=12)
    ap.add_argument("--meta-lr", type=float, default=8e-4)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--target-acc", type=float, default=87.0)
    ap.add_argument("--patience", type=int, default=5, help="early stop patience (epochs w/o val improvement)")
    # checkpoints
    ap.add_argument("--resume", action="store_true", help="resume from last_tr.pth if exists")
    ap.add_argument("--init-supervised", default="best_supervised.pth")
    return ap

def main():
    args = build_argparser().parse_args()

    # ---------------------------
    # Setup
    # ---------------------------
    RESULTS_DIR = args.results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CSV_PATH  = os.path.join(RESULTS_DIR, "metrics_tz.csv")
    PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_tz.png")
    BEST_CKPT = os.path.join(RESULTS_DIR, "best_tz.pth")
    LAST_CKPT = os.path.join(RESULTS_DIR, "last_tz.pth")
    INIT_SUP_CKPT = os.path.join(RESULTS_DIR, args.init_supervised)

    assert torch.cuda.is_available(), "CUDA not available—please run on GPU."
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))

    set_seed(args.seed, deterministic=args.deterministic)

    # Samplers
    train_sampler = EpisodeSampler(args.train_root, image_size=224, normalize=True)
    val_sampler   = EpisodeSampler(args.val_root,   image_size=224, normalize=True)

    # Build model
    net  = TaskCondNet(n_way=args.n_way, te_dim=128).to(device)
    maml = MAML_TE(model=net, inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                   meta_lr=args.meta_lr, head_only=True).to(device)

    # Warm-load supervised EfficientNet features if available
    if os.path.isfile(INIT_SUP_CKPT):
        ck = torch.load(INIT_SUP_CKPT, map_location="cpu")
        sd = ck.get("model", ck)
        if hasattr(net.backbone, "features"):
            feat_keys = {k.replace("features.", ""): v for k, v in sd.items() if k.startswith("features.")}
            missing, unexpected = net.backbone.features.load_state_dict(feat_keys, strict=False)
            print(f"Loaded backbone features from {INIT_SUP_CKPT}\n  missing: {list(missing)}\n  unexpected: {list(unexpected)}")
        else:
            print("⚠️ Could not match EfficientNet features; using ImageNet weights.")

    # Optional memory bank
    mb, retriever = _make_memory_bank()
    if mb is None:
        print("ℹ️  Task memory disabled (no compatible MemoryBank found).")

    # Scheduler
    scheduler = CosineAnnealingLR(maml.meta_optimizer, T_max=max(args.epochs//2, 1), eta_min=1e-5)

    print(
        f"🎯 MAML+FiLM (EffB0) | {args.n_way}-way {args.k_shot}-shot (Q={args.q_query}) | "
        f"inner {args.inner_steps}@{args.inner_lr} | meta-batch {args.meta_batch} | AMP fwd {AMP_FWD}"
    )

    # Resume (optional)
    start_epoch, best_val = 0, -1.0
    if args.resume and os.path.isfile(LAST_CKPT):
        ck = torch.load(LAST_CKPT, map_location="cpu")
        try:
            net.load_state_dict(ck["model"])
            maml.meta_optimizer.load_state_dict(ck["opt"])
            scheduler.load_state_dict(ck["sched"])
            best_val = ck.get("best_val", best_val)
            start_epoch = ck.get("epoch", -1) + 1
            print(f"🔁 Resumed from epoch {start_epoch}, best_val={best_val:.2f}%")
        except Exception as e:
            print(f"⚠️  Resume failed ({e}). Starting fresh.")

    history = {"epoch":[],"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[],"val_adapt_ms":[]}
    patience_since_best = 0

    header = ["epoch","train_loss","train_acc","val_loss","val_acc","val_adapt_ms"]
    if start_epoch == 0 and os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)  # fresh run → fresh CSV
    # ---------------------------
    # Training
    # ---------------------------
    for ep in range(start_epoch, args.epochs):
        net.train()
        ep_losses, ep_accs = [], []
        pbar = trange(args.episodes_per_epoch, leave=False)

        i = 0
        while i < args.episodes_per_epoch:
            maml.meta_optimizer.zero_grad(set_to_none=True)
            steps_this_batch = 0
            meta_loss_accum  = 0.0
            meta_accs        = []
            adapted_list     = []  # for reptile head updates

            for _ in range(args.meta_batch):
                if i >= args.episodes_per_epoch: break
                i += 1; steps_this_batch += 1

                sx, sy, qx, qy = train_sampler.generate_task(n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
                sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

                # inner: head+task encoder only; backbone shared (no duplicate VRAM)
                adapted, task_e = maml.adapt(sx, sy)

                # outer: real grads for backbone from query loss
                with amp_autocast():
                    q_feats = adapted.features(qx)
                    logits  = adapted.logits(q_feats, task_e)
                    loss    = label_smooth_ce(logits, qy, eps=0.05)

                loss.backward()  # accumulates real grads on SHARED backbone
                meta_loss_accum += loss.item()
                meta_accs.append(accuracy_from_logits(logits, qy))
                adapted_list.append(adapted)

                # memory bank (store task embedding sparsely to limit growth)
                if mb is not None and (i % 3 == 0):
                    try:
                        te = F.normalize(task_e.squeeze(0), dim=0)
                        mb.add(te.detach().float().cpu().numpy(), meta={"epoch": ep})
                    except Exception:
                        pass

            # synthetic grads for head + task encoder (Reptile-style)
            w = 1.0 / max(1, len(adapted_list))
            for a in adapted_list:
                maml.accumulate_reptile_head(a, weight=w)

            # step outer
            clip_grad_norm_(maml.model.parameters(), args.grad_clip)
            maml.step()

            ep_losses.append(meta_loss_accum / max(1, steps_this_batch))
            ep_accs.append(float(np.mean(meta_accs)))
            pbar.update(steps_this_batch)

        scheduler.step()
        train_loss, train_acc = float(np.mean(ep_losses)), float(np.mean(ep_accs))
        val_loss, val_acc, val_adapt_ms = evaluate_on_val(
            maml, val_sampler, device, args.n_way, args.k_shot, args.q_query, episodes=args.val_episodes
        )

        # save ckpts
        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            patience_since_best = 0
            torch.save({"model": net.state_dict(), "epoch": ep, "val_acc": best_val}, BEST_CKPT)
        else:
            patience_since_best += 1

        torch.save({"model": net.state_dict(),
                    "opt": maml.meta_optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "best_val": best_val,
                    "epoch": ep}, LAST_CKPT)

        # save memory bank periodically
        if mb is not None and (ep % 2 == 0 or ep == args.epochs-1):
            out = os.path.join(RESULTS_DIR, "memory_bank.pkl")
            try:
                if save_retriever_index:
                    save_retriever_index(mb, retriever, out)
                else:
                    with open(out, "wb") as f:
                        pickle.dump({"vectors": getattr(mb, "vectors", None),
                                     "meta": getattr(mb, "meta", None)}, f)
            except Exception as e:
                print(f"⚠️  Could not save memory bank ({e}).")

        # log/plot
        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_adapt_ms"].append(val_adapt_ms)
        append_csv_row(CSV_PATH, [ep, train_loss, train_acc, val_loss, val_acc, val_adapt_ms], header if ep==0 and start_epoch==0 else None)

        save_plot(history, PLOT_PATH)

        print(f"epoch {ep:02d} | Train {train_acc:.2f}% | Val {val_acc:.2f}% "
              f"| Best {best_val:.2f}% | Val Adapt {val_adapt_ms:.1f} ms")

        # early stop / target reached
        if val_acc >= args.target_acc:
            print(f"🎉 Target {args.target_acc:.1f}% reached at epoch {ep}. Stopping.")
            break
        if patience_since_best >= args.patience:
            print(f"⏹️  Early stopping: no val improvement for {args.patience} epochs.")
            break

    print("✅ MAML training complete.")
    print(f"• Metrics CSV:   {CSV_PATH}")
    print(f"• Metrics Plot:  {PLOT_PATH}")
    print(f"• Best Checkpoint: {BEST_CKPT}")
    print(f"• Last Checkpoint: {LAST_CKPT}")

if __name__ == "__main__":
    main()
