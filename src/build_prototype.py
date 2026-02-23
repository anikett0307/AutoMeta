# src/build_prototypes.py
import os, sys, argparse, pickle
from typing import List
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maml_model import TaskCondNet  # EfficientNet-B0 backbone inside

def list_images(root: str, exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")) -> List[str]:
    return [os.path.join(root, f) for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(exts)]

def load_image(path, tfm):
    img = Image.open(path).convert("RGB")
    return tfm(img)

def load_backbone_safely(net: TaskCondNet, ckpt_path: str):
    """
    Load ONLY backbone weights from a training checkpoint, robust to
    different key prefixes (e.g., 'backbone.features.*' or 'features.*').
    Avoids classifier head shape mismatches (5-way vs 7-way).
    """
    if not os.path.isfile(ckpt_path):
        print(f"⚠️  Checkpoint not found: {ckpt_path} (continuing with ImageNet weights).")
        return

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)  # handle both dict formats

    # Try common layouts in saved state_dict
    loaded = False

    # Case A: keys like 'backbone.features.X...'
    feat_keys = {k: v for k, v in sd.items() if k.startswith("backbone.features.")}
    if feat_keys:
        # strip 'backbone.features.' -> load into net.backbone.features
        trimmed = {k.replace("backbone.features.", ""): v for k, v in feat_keys.items()}
        missing, unexpected = net.backbone.features.load_state_dict(trimmed, strict=False)
        print(f"✓ Loaded backbone.features from {ckpt_path}")
        print("  missing:", list(missing))
        print("  unexpected:", list(unexpected))
        loaded = True

    # Case B: keys like 'features.X...' (e.g., supervised pretrain file)
    if not loaded:
        feat_keys = {k: v for k, v in sd.items() if k.startswith("features.")}
        if feat_keys:
            trimmed = {k.replace("features.", ""): v for k, v in feat_keys.items()}
            missing, unexpected = net.backbone.features.load_state_dict(trimmed, strict=False)
            print(f"✓ Loaded backbone.features from {ckpt_path} (features.* layout)")
            print("  missing:", list(missing))
            print("  unexpected:", list(unexpected))
            loaded = True

    if not loaded:
        # Last resort: try loading the whole backbone module if it exists in sd
        try:
            bb_keys = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
            if bb_keys:
                missing, unexpected = net.backbone.load_state_dict(bb_keys, strict=False)
                print(f"✓ Loaded backbone (broad) from {ckpt_path}")
                print("  missing:", list(missing))
                print("  unexpected:", list(unexpected))
                loaded = True
        except Exception as e:
            print(f"⚠️  Broad backbone load failed: {e}")

    if not loaded:
        print("⚠️  Could not find backbone keys in checkpoint. Using ImageNet weights.")

def main():
    ap = argparse.ArgumentParser(description="Build global class prototypes from the training split.")
    ap.add_argument("--source", default=os.path.join(PROJECT_ROOT, "data", "HAM10000_split", "train"),
                    help="Training folder with class subfolders.")
    ap.add_argument("--ckpt",   default=os.path.join(PROJECT_ROOT, "results", "best_supervised.pth"),
                    help="Path to trained checkpoint (used only for backbone).")
    ap.add_argument("--out",    default=os.path.join(PROJECT_ROOT, "results", "prototypes.pkl"),
                    help="Output path for prototypes pickle.")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch",  type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU (CUDA not available).")

    # transforms (match training/eval query transforms)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # discover classes
    class_names = sorted([d for d in os.listdir(args.source)
                          if os.path.isdir(os.path.join(args.source, d))])
    assert class_names, f"No classes found in {args.source}"
    print(f"Found {len(class_names)} classes:", class_names)

    # Build model with n_way=len(classes). We'll only use .features()
    net = TaskCondNet(n_way=len(class_names)).to(device)
    load_backbone_safely(net, args.ckpt)
    net.eval()

    # compute mean feature per class
    protos = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=torch.float16):
        for cname in class_names:
            cdir = os.path.join(args.source, cname)
            paths = list_images(cdir)
            assert paths, f"No images in {cdir}"

            feats = []
            B = max(1, args.batch)
            for i in range(0, len(paths), B):
                batch_paths = paths[i:i+B]
                xs = torch.stack([load_image(p, tfm) for p in batch_paths], 0).to(device)
                f = net.features(xs)            # [B, 1280], L2-normalized inside EffB0Backbone
                feats.append(f.float())
            feats = torch.cat(feats, 0)        # [N_c, 1280]
            mu = F.normalize(feats.mean(0, keepdim=True), p=2, dim=1)  # [1,1280]
            protos.append(mu.squeeze(0).cpu().numpy())

    protos = np.stack(protos, 0)  # [C, 1280]

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"class_names": class_names, "prototypes": protos}, f)
    print(f"✓ Saved prototypes -> {args.out}")

if __name__ == "__main__":
    main()
