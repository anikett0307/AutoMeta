# src/streamlit_app.py
import os, sys, io
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# ------------------------------------------
# Paths / imports
# ------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from src.maml_model import TaskCondNet  # when project root is on sys.path
except ModuleNotFoundError:
    from maml_model import TaskCondNet      # when running from inside src/

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
# >>> IMPORTANT: default to meta-trained backbone (matches prototypes)
DEFAULT_CKPT  = os.path.join(RESULTS_DIR, "best_supervised.pth")
DEFAULT_PROTO = os.path.join(RESULTS_DIR, "prototypes.pkl")
CACHE_DIR     = os.path.join(PROJECT_ROOT, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------
# Transforms
# ------------------------------------------
BASE_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

FIVECROP_PRE = transforms.Resize(256)
FIVECROP = transforms.FiveCrop(224)
TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def device_select():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------------------------
# Cached loaders
# ------------------------------------------
@st.cache_resource(show_spinner=False)
def load_prototypes(proto_path: str):
    """Load prototypes saved either with torch.save or pickle.dump.

    Returns (class_names, prototypes_np)
    """
    data = None
    # Try torch first (robust to various storage formats)
    try:
        data = torch.load(proto_path, map_location="cpu")
    except Exception:
        pass
    if data is None:
        import pickle
        with open(proto_path, "rb") as f:
            data = pickle.load(f)

    # Normalize common structures
    if isinstance(data, dict):
        if "class_names" in data and "prototypes" in data:
            class_names = data["class_names"]
            protos = data["prototypes"]
        elif "classes" in data and "protos" in data:
            class_names = data["classes"]
            protos = data["protos"]
        else:
            raise ValueError("Unrecognized prototype dict keys. Expected 'class_names' and 'prototypes'.")
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        class_names, protos = data
    else:
        raise ValueError("Unsupported prototype file format.")

    protos_np = np.asarray(protos, dtype=np.float32)
    return class_names, protos_np

@st.cache_resource(show_spinner=False)
def load_backbone(ckpt_path: str, n_classes: int, device: torch.device):
    net = TaskCondNet(n_way=n_classes).to(device)
    net.eval()
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("model", ck)
        if hasattr(net.backbone, "features"):
            feat_keys = {k.replace("features.", ""): v for k,v in sd.items()
                         if k.startswith("features.")}
            net.backbone.features.load_state_dict(feat_keys, strict=False)
        else:
            print("⚠️ EfficientNet features not found; using ImageNet weights.")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}; using ImageNet weights.")
    return net

# ------------------------------------------
# TTA builder
# ------------------------------------------
def make_tta_batch(img: Image.Image, mode: str):
    """Return a list of PIL images according to the TTA mode."""
    imgs = [img]
    if mode == "None":
        return imgs
    if mode in ("Flips", "Flips + Rotations"):
        imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        imgs.append(img.transpose(Image.FLIP_TOP_BOTTOM))
    if mode == "Flips + Rotations":
        imgs.extend([img.rotate(90, expand=True),
                     img.rotate(180, expand=True),
                     img.rotate(270, expand=True)])
    if mode == "Five-Crop (256→224)":
        big = FIVECROP_PRE(img)
        crops = FIVECROP(big)  # tuple of 5 PIL Images
        imgs = list(crops)
    return imgs

def images_to_tensor_batch(pil_list):
    # Always resize to 224x224 and normalize (even for Five-Crop/rotations).
    tensors = [BASE_TFM(p) for p in pil_list]   # BASE_TFM includes Resize(224,224)
    return torch.stack(tensors, 0)              # [T, 3, 224, 224]


# ------------------------------------------
# Inference
# ------------------------------------------
@torch.inference_mode()
def predict_image_tta(
    net: TaskCondNet,
    protos: torch.Tensor,           # [C,1280], L2-normalized
    img: Image.Image,
    temperature: float,
    tta_mode: str,
    device: torch.device
):
    views = make_tta_batch(img, tta_mode)
    x = images_to_tensor_batch(views).to(device)  # [T,3,224,224]

    use_amp = (device.type == "cuda")
    dtype   = torch.float16 if use_amp else torch.float32

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        feats = net.features(x)                          # [T,1280], L2-normalized in backbone
        feats = F.normalize(feats.mean(0, keepdim=True), # average then re-normalize
                            p=2, dim=1)                  # [1,1280]
        logits = feats @ protos.T                        # [1,C] cosine
        logits = logits * float(temperature)
        probs  = torch.softmax(logits, dim=1)[0]         # [C]
    return probs

def topk_from_probs(probs: torch.Tensor, class_names, k=5):
    probs_np = probs.detach().float().cpu().numpy()
    order = np.argsort(-probs_np)[:k]
    return [(class_names[i], float(probs_np[i])) for i in order]

# ------------------------------------------
# UI
# ------------------------------------------
st.set_page_config(page_title="Skin Lesion Few-Shot Classifier", layout="centered")
st.title("🩺 Skin Lesion Few-Shot Classifier")

with st.sidebar:
    st.header("Settings")
    ckpt_path  = st.text_input("Checkpoint (.pth)", DEFAULT_CKPT)
    proto_path = st.text_input("Prototypes (.pkl)", DEFAULT_PROTO)

    # Normalize to absolute paths if user input is relative
    ckpt_path  = os.path.normpath(ckpt_path)
    proto_path = os.path.normpath(proto_path)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
    if not os.path.isabs(proto_path):
        proto_path = os.path.join(PROJECT_ROOT, proto_path)

    # Optional: upload files instead of typing paths
    up_proto = st.file_uploader("Or upload prototypes file", type=["pkl","pt","pth"], key="proto_upload")
    if up_proto is not None:
        proto_cache = os.path.join(CACHE_DIR, "prototypes_uploaded.pkl")
        with open(proto_cache, "wb") as f:
            f.write(up_proto.getbuffer())
        proto_path = proto_cache
        st.caption(f"Using uploaded prototypes → {proto_path}")

    up_ckpt = st.file_uploader("Or upload checkpoint (.pth)", type=["pth","pt"], key="ckpt_upload")
    if up_ckpt is not None:
        ckpt_cache = os.path.join(CACHE_DIR, "checkpoint_uploaded.pth")
        with open(ckpt_cache, "wb") as f:
            f.write(up_ckpt.getbuffer())
        ckpt_path = ckpt_cache
        st.caption(f"Using uploaded checkpoint → {ckpt_path}")

    temperature = st.slider("Temperature (softmax scale)", 1.0, 30.0, 8.0, 0.5)
    tta_mode    = st.selectbox("Test-time augmentation", 
                               ["None", "Flips", "Flips + Rotations", "Five-Crop (256→224)"],
                               index=2)
    uncertain_thr = st.slider("Mark as 'Uncertain' below", 0.0, 0.9, 0.50, 0.01)

    device = device_select()
    st.write(f"**Device:** `{device.type}`")

    # load (non-blocking): show uploader even if this fails
    net = None
    protos = None
    class_names = None
    try:
        if os.path.exists(proto_path) and os.path.exists(ckpt_path):
            class_names, protos_np = load_prototypes(proto_path)
            net = load_backbone(ckpt_path, n_classes=len(class_names), device=device)
            protos = F.normalize(torch.from_numpy(protos_np).to(device), p=2, dim=1)
            st.success(f"✅ Loaded {len(class_names)} classes.")
            st.caption("Make sure prototypes and checkpoint come from the same training run.")
        else:
            if not os.path.exists(proto_path):
                st.error(f"Prototypes not found: {proto_path}")
            if not os.path.exists(ckpt_path):
                st.error(f"Checkpoint not found: {ckpt_path}")
    except Exception as e:
        st.error(f"Failed to load model/prototypes: {e}")

st.markdown("### 📸 Upload a Dermoscopic Image")
uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1, col2 = st.columns([1,1])
    with col1: st.image(img, caption="Uploaded image", use_container_width=True)

    # Check if filename contains "ISIC"
    if "ISIC" not in uploaded.name:
        st.markdown("---")
        st.markdown("### 🎯 Classification Results")
        st.error("The uploaded image is not a valid skin lesion or does not have the specification settings of HAM10000. Hence cannot classify the uploaded image.")
    else:
        if net is None or protos is None or class_names is None:
            st.error("Model is not loaded. Fix paths in the sidebar to run inference.")
            st.stop()
        else:
            with st.spinner("🔍 Analyzing..."):
                probs = predict_image_tta(
                    net=net, protos=protos, img=img,
                    temperature=temperature, tta_mode=tta_mode, device=device
                )
                top5 = topk_from_probs(probs, class_names, k=min(5, len(class_names)))

        pred_name, pred_prob = top5[0]
        st.markdown("---")
        st.markdown("### 🎯 Classification Results")

        st.markdown(f"**Predicted Disease:** `{pred_name.replace('_',' ').title()}`")

    # Class probabilities removed per user request

    # Removed Top-5 breakdown section per user request

else:
    st.info("👆 Upload a dermoscopic image to begin.")
