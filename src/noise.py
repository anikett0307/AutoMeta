import os
from collections import Counter

# Adjust paths if different
train_dir = r"E:\Major\AutoMeta\data\HAM10000_split\train"
val_dir   = r"E:\Major\AutoMeta\data\HAM10000_split\val"


def count_images(folder):
    counts = {}
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    return counts

train_counts = count_images(train_dir)
val_counts   = count_images(val_dir)

print("=== TRAIN SPLIT ===")
total_train = sum(train_counts.values())
for k,v in train_counts.items():
    print(f"{k}: {v}")
print(f"Total train: {total_train}")

print("\n=== VAL SPLIT ===")
total_val = sum(val_counts.values())
for k,v in val_counts.items():
    print(f"{k}: {v}")
print(f"Total val: {total_val}")
