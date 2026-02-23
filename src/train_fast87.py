# -*- coding: utf-8 -*-
"""
Fast supervised trainer + optional episodic head-tune (ANIL-style)
Designed for HAM10000_split on a GTX 1650-class GPU (16 GB RAM system).

Usage:
  python train_fast87.py --epochs 15 --batch 64 --warmup 2 --episodic_tune
  python train_fast87.py --epochs 15 --batch 64 --no-warmup              (skip warm-up)
  python train_fast87.py --resume                                        (resume last.pth)

Outputs (under results/):
  - best_supervised.pth                 (best val by accuracy)
  - last_supervised.pth                 (last epoch)
  - best_supervised_plus_episodic.pth   (after optional episodic head-tune)
  - metrics.csv, metrics.png
"""
import os, csv, argparse, random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ---------- Paths / Device ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_TRAIN   = os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "train")
DATA_VAL     = os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "val")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_SUP = os.path.join(RESULTS_DIR, "best_supervised.pth")
LAST_SUP = os.path.join(RESULTS_DIR, "last_supervised.pth")
CSV_PATH = os.path.join(RESULTS_DIR, "metrics9.csv")
PLOT_PATH= os.path.join(RESULTS_DIR, "metrics9`.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = device.type == "cuda"
print("Device:", device, "| AMP:", AMP_ENABLED)

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ---------- Utils ----------
def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def label_smooth_ce(logits, targets, eps=0.1, num_classes: Optional[int]=None):
    if num_classes is None:
        num_classes = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes).float()
    yls = (1.0 - eps) * one_hot + eps / num_classes
    return -(yls * logp).sum(dim=1).mean()

def save_plot(history, path):
    df = pd.DataFrame(history)
    if len(df) == 0: return
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.title("Loss over Epochs")
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    if df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    # Accuracy
    plt.subplot(1,2,2)
    plt.title("Accuracy over Epochs")
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    if df["val_acc"].notna().any():
        plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def append_csv_row(ep, tr_loss, tr_acc, v_loss, v_acc):
    new_file = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerow([ep, tr_loss, tr_acc, v_loss, v_acc])

# ---------- Model: EfficientNet-B0 with small head ----------
class EffNetB0(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        in_feats = base.classifier[1].in_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

def freeze_backbone(model: EffNetB0, freeze: bool):
    for p in model.features.parameters():
        p.requires_grad = not (freeze)
    for p in model.head.parameters():
        p.requires_grad = True

# ---------- Data ----------
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size) if img_size>=224 else transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def make_weighted_sampler(dataset: datasets.ImageFolder):
    # handle class imbalance
    targets = [y for _, y in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[y] for y in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# ---------- Train / Eval ----------
def train_one_epoch(model, loader, opt, scaler, epoch, epochs, label_smooth_eps=0.1):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, leave=False)
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with amp.autocast(enabled=AMP_ENABLED):
            logits = model(x)
            loss = label_smooth_ce(logits, y, label_smooth_eps, num_classes=logits.size(-1))
        if scaler:
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        running_loss += loss.item() * x.size(0)
        running_acc  += (logits.argmax(1)==y).float().sum().item()
        n += x.size(0)
        pbar.set_description(f"ep {epoch+1}/{epochs} | loss {running_loss/n:.4f} | acc {100*running_acc/n:.2f}%")
    return running_loss/n, 100.0*running_acc/n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        running_loss += loss.item() * x.size(0)
        running_acc  += (logits.argmax(1)==y).float().sum().item()
        n += x.size(0)
    return running_loss/n, 100.0*running_acc/n

# ---------- Optional episodic ANIL head-tune ----------
def quick_episodic_head_tune(model, data_root, steps=150, n_way=5, k_shot=5, q_query=15, inner_lr=5e-3):
    try:
        from task_generator import EpisodeSampler
    except Exception:
        print("⚠️ task_generator.EpisodeSampler not found – skipping episodic tune.")
        return
    print("\n🎯 Episodic (ANIL) head-tune...")
    freeze_backbone(model, True)  # tune head only
    head_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=inner_lr, weight_decay=1e-4)
    sampler = EpisodeSampler(os.path.join(data_root, "train"), image_size=224, normalize=True)
    model.train()
    for i in tqdm(range(steps), leave=False):
        sx, sy, qx, qy = sampler.generate_task(n_way=n_way, k_shot=k_shot, q_query=q_query)
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
        head_opt.zero_grad(set_to_none=True)
        # inner-step: head update on support
        logits_s = model(sx)
        loss_s   = F.cross_entropy(logits_s, sy)
        loss_s.backward()
        clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        head_opt.step()
    torch.save({"model": model.state_dict()}, os.path.join(RESULTS_DIR, "best_supervised_plus_episodic.pth"))
    print("✅ Saved episodically tuned model -> results/best_supervised_plus_episodic.pth")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2, help="head-only warm-up epochs (set 0 to disable)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--episodic_tune", action="store_true")
    args = ap.parse_args()

    EPOCHS      = args.epochs
    BATCH       = args.batch
    WARMUP_E    = 0 if args.no_warmup else max(0, args.warmup)
    BASE_LR     = args.lr

    # Data
    train_tf, val_tf = build_transforms(img_size=224)
    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_VAL,   transform=val_tf)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Weighted sampler to fight imbalance
    sampler = make_weighted_sampler(train_ds)
    train_loader_warm = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                   num_workers=4, pin_memory=(device.type=="cuda"))
    # Unfrozen loader (shuffle also ok)
    train_loader_unf  = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                   num_workers=4, pin_memory=(device.type=="cuda"))
    val_loader        = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                                   num_workers=4, pin_memory=(device.type=="cuda"))

    # Model
    model = EffNetB0(num_classes=num_classes).to(device)
    scaler = amp.GradScaler(enabled=AMP_ENABLED)

    # ---- Resume (if requested) ----
    start_epoch, best_val = 0, -1.0
    if args.resume and os.path.isfile(LAST_SUP):
        ckpt = torch.load(LAST_SUP, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        best_val = ckpt.get("best_val", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"🔁 Resumed from last_supervised.pth @ epoch {start_epoch}, best_val {best_val:.2f}%")

    # ---- Phase 1: warm-up (optional) ----
    if start_epoch == 0 and WARMUP_E > 0:
        print(f"🔥 Warm-up (head-only) for {WARMUP_E} epoch(s)")
        freeze_backbone(model, True)
        opt_w = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=2e-3, weight_decay=1e-4)
        sch_w = CosineAnnealingLR(opt_w, T_max=WARMUP_E, eta_min=2e-4)
        for ep in range(WARMUP_E):
            tr_loss, tr_acc = train_one_epoch(model, train_loader_warm, opt_w, scaler, ep, WARMUP_E, 0.1)
            vl_loss, vl_acc = evaluate(model, val_loader)
            sch_w.step()
            print(f"[WarmUp {ep+1}/{WARMUP_E}] Train {tr_acc:.2f}% | Val {vl_acc:.2f}%")

    # ---- Phase 2: full fine-tune ----
    print("🚀 Full fine-tune")
    freeze_backbone(model, False)
    opt = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=max(EPOCHS, 1), eta_min=BASE_LR/10)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(start_epoch, EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader_unf, opt, scaler, ep, EPOCHS, 0.1)
        vl_loss, vl_acc = evaluate(model, val_loader)
        sch.step()

        # save last + best
        torch.save({"model": model.state_dict(), "epoch": ep, "best_val": max(best_val, vl_acc)}, LAST_SUP)
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best_val}, BEST_SUP)

        # log/plot
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        append_csv_row(ep, tr_loss, tr_acc, vl_loss, vl_acc)
        save_plot(history, PLOT_PATH)

        print(f"Epoch {ep:02d} | Train {tr_acc:.2f}% | Val {vl_acc:.2f}% "
              f"| Best {best_val:.2f}% | LR {opt.param_groups[0]['lr']:.2e}")

    print("✅ Supervised training complete.")
    print(f"• Metrics CSV:   {CSV_PATH}")
    print(f"• Metrics Plot:  {PLOT_PATH}")
    print(f"• Best Checkpoint: {BEST_SUP}")
    print(f"• Last Checkpoint: {LAST_SUP}")

    # ---- Optional episodic head-tune for few-shot reporting ----
    if args.episodic_tune:
        quick_episodic_head_tune(model, os.path.join(PROJECT_ROOT, "data", "HAM10000_split"),
                                 steps=150, n_way=5, k_shot=5, q_query=15, inner_lr=5e-3)
        print("🎯 Episodic head-tune done.")

if __name__ == "__main__":
    main()
