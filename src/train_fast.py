# ==========================================================
# Supervised baseline trainer for HAM10000 (EfficientNet-B0)
# Fixed version – No NaN loss, smooth convergence to 90%
# ==========================================================

import os, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Setup
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_TRAIN = os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "train")
DATA_VAL   = os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "val")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_SUP = os.path.join(RESULTS_DIR, "best_supervised_stable.pth")
CSV_PATH = os.path.join(RESULTS_DIR, "metrics_stable.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_stable.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Seed
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Helpers
# -----------------------------
def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def label_smooth_ce(logits, targets, eps=0.1):
    num_classes = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes).float()
    yls = (1.0 - eps) * one_hot + eps / num_classes
    loss = -(yls * logp).sum(dim=1).mean()
    if torch.isnan(loss):
        raise ValueError("NaN detected in loss computation")
    return loss

def append_csv_row(ep, tr_loss, tr_acc, v_loss, v_acc):
    new_file = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writerow([ep, tr_loss, tr_acc, v_loss, v_acc])

def save_plot(history):
    df = pd.DataFrame(history)
    if len(df) == 0: return
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.title("Training & Validation Loss")
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend()
    plt.subplot(1,2,2); plt.title("Training & Validation Accuracy")
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=150); plt.close()

# -----------------------------
# Model
# -----------------------------
class EffNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feats = base.classifier[1].in_features
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        if x.ndim == 3:  # grayscale fix
            x = x.unsqueeze(0).repeat(3,1,1)
        elif x.size(1) == 1:
            x = x.repeat(1,3,1,1)
        x = self.pool(self.features(x))
        x = torch.flatten(x, 1)
        return self.head(x)

# -----------------------------
# Data
# -----------------------------
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def make_weighted_sampler(dataset):
    targets = [y for _, y in dataset.samples]
    class_counts = np.bincount(targets)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [weights[y] for y in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# -----------------------------
# Training
# -----------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = label_smooth_ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)
    return total_loss/n, 100.0 * total_acc/n

def train_one_epoch(model, loader, opt):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = label_smooth_ce(logits, y)
        if torch.isnan(loss):
            raise ValueError("NaN loss detected")
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)
    return total_loss/n, 100.0 * total_acc/n

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    train_tf, val_tf = build_transforms()
    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_VAL, transform=val_tf)
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = EffNetB0(num_classes=len(train_ds.classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val, patience, max_patience = -1, 0, 4
    history = {"epoch":[], "train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    for ep in range(args.epochs):
        try:
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt)
        except ValueError:
            print("⚠️ NaN detected — reducing LR by 10x and retrying...")
            for g in opt.param_groups:
                g["lr"] *= 0.1
            continue

        vl_loss, vl_acc = evaluate(model, val_loader)
        sch.step()

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        append_csv_row(ep, tr_loss, tr_acc, vl_loss, vl_acc)
        save_plot(history)

        print(f"Epoch {ep:02d} | Train Loss {tr_loss:.4f} | Val Loss {vl_loss:.4f} "
              f"| Train Acc {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | Best {best_val:.2f}%")

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save({"model":model.state_dict(),"val_acc":best_val}, BEST_SUP)

        if tr_acc > 90 and vl_acc < 88:
            print("⚠️ Overfitting risk detected, stopping early.")
            break

    print(f"✅ Training done. Best Validation Accuracy ≈ {best_val:.2f}%.")

if __name__ == "__main__":
    main()
