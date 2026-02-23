import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# === Import model from your supervised trainer ===
from train_fast87 import EffNetB0

# ---------- CONFIG ----------
CHECKPOINT_PATH = r"E:\Major\AutoMeta\results\best_supervised_stable.pth"  # or best_supervised.pth
VAL_PATH = r"E:\Major\AutoMeta\data\HAM10000_split\val"
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ---------- LOAD MODEL ----------
model = EffNetB0(num_classes=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)

clean_state_dict = {}

# Automatically handle prefix mismatches (backbone./features./head.)
for k, v in state_dict.items():
    if k.startswith("backbone.features."):
        new_key = k.replace("backbone.", "")
        clean_state_dict[new_key] = v
    elif k.startswith("features.") or k.startswith("head."):
        clean_state_dict[k] = v
    elif "backbone" not in k and "task_enc" not in k and "film" not in k:
        clean_state_dict[k] = v

missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
print(f"\n✅ Checkpoint loaded successfully!")
print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}\n")

# ---------- EVALUATE ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_ds = datasets.ImageFolder(VAL_PATH, transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Evaluating"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

try:
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
except:
    auc = None

print("\n📊 Evaluation Metrics:")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
if auc:
    print(f"AUC-ROC:   {auc:.2f}")
