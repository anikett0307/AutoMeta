# AutoMeta/src/split.py
import os
import shutil
import random
from pathlib import Path

# ---------- Resolve project paths robustly ----------
THIS_DIR     = Path(__file__).resolve().parent          # .../AutoMeta/src
PROJECT_ROOT = THIS_DIR.parent                          # .../AutoMeta
DATA_DIR     = PROJECT_ROOT / "data" / "HAM10000"       # <- your classes live here
OUTPUT_DIR   = PROJECT_ROOT / "data" / "HAM10000_split" # <- will be created

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
random.seed(42)

# ---------- Sanity checks ----------
if not DATA_DIR.exists():
    print("❌ Could not find the dataset folder.")
    print(f"  Expected: {DATA_DIR}")
    print("  Tip: Open Finder/Explorer and confirm the exact folder name under AutoMeta/data/")
    raise SystemExit(1)

classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
if not classes:
    print(f"❌ No class subfolders found in: {DATA_DIR}")
    raise SystemExit(1)

print("✅ Using dataset at:", DATA_DIR)
print("✅ Found classes:", classes)

# ---------- Prepare output directories ----------
for split in SPLITS.keys():
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# ---------- Perform split per class ----------
for cls in classes:
    cls_src = DATA_DIR / cls
    images = [f for f in cls_src.iterdir() if f.is_file()]
    random.shuffle(images)

    n = len(images)
    n_train = int(SPLITS["train"] * n)
    n_val   = int(SPLITS["val"] * n)

    split_map = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, files in split_map.items():
        dst_dir = OUTPUT_DIR / split / cls
        for src_path in files:
            shutil.copy2(src_path, dst_dir / src_path.name)

print(f"✅ Dataset split complete.\n   Output: {OUTPUT_DIR}")