# src/task_generator.py
import os, random
from collections import defaultdict, OrderedDict
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms


class _LRUCache(OrderedDict):
    """Simple path->tensor cache to reduce disk I/O."""
    def __init__(self, max_items: int = 512):
        super().__init__()
        self.max_items = max_items

    def get(self, key):
        if key in self:
            val = super().pop(key)
            super().__setitem__(key, val)
            return val
        return None

    def put(self, key, val):
        if key in self:
            super().pop(key)
        super().__setitem__(key, val)
        while len(self) > self.max_items:
            self.popitem(last=False)


class EpisodeSampler:
    """
    Balanced episodic sampler for N-way K-shot, Q-query tasks.

    - Uniformly samples N distinct classes.
    - Draws (K+Q) images per selected class *without overlap* between support/query.
    - Falls back to with-replacement if a class has too few images.
    - Optional light augmentation on support only.
    - Tiny LRU cache to avoid re-decoding images every episode.
    """
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        normalize: bool = True,
        support_aug: bool = True,
        cache_items: int = 512,
    ):
        self.root = root_dir
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        if len(self.class_names) == 0:
            raise RuntimeError(f"No class folders found in: {root_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Build per-class file index
        self.files_by_class = defaultdict(list)
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for cname in self.class_names:
            cpath = os.path.join(root_dir, cname)
            files = [
                os.path.join(cpath, f) for f in os.listdir(cpath)
                if f.lower().endswith(exts)
            ]
            if not files:
                raise RuntimeError(f"No images in class folder: {cpath}")
            self.files_by_class[self.class_to_idx[cname]] = files

        # Base transforms (no heavy aug by default)
        base = [transforms.Resize((image_size, image_size)),
                transforms.ToTensor()]
        if normalize:
            base.append(transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]))
        self.base_transform = transforms.Compose(base)

        # Optional *light* support augmentation
        self.support_aug = support_aug
        self.support_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]) if support_aug else self.base_transform

        # Small LRU cache for decoded tensors
        self.cache = _LRUCache(max_items=cache_items)

    def _load_tensor(self, path: str, transform: transforms.Compose) -> torch.Tensor:
        cached = self.cache.get(path)
        if cached is not None and transform is self.base_transform:
            # only reuse when transforms are identical
            return cached
        img = Image.open(path).convert("RGB")
        x = transform(img)
        if transform is self.base_transform:
            self.cache.put(path, x)
        return x

    def _sample_unique(self, pool: List[str], n: int) -> List[str]:
        """Sample n unique items if possible; fallback to with-replacement."""
        if len(pool) >= n:
            return random.sample(pool, n)
        # Not enough images: sample with replacement, but ensure no S/Q overlap later
        return [random.choice(pool) for _ in range(n)]

    def generate_task(
        self,
        n_way: int,
        k_shot: int,
        q_query: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_way > len(self.class_names):
            raise ValueError(f"Requested n_way={n_way} but only {len(self.class_names)} classes in {self.root}")

        # Choose N distinct classes uniformly
        chosen_cls = random.sample(list(self.class_to_idx.values()), n_way)

        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []

        per_class_total = k_shot + q_query
        for new_label, cls_idx in enumerate(chosen_cls):
            pool = self.files_by_class[cls_idx]
            picks = self._sample_unique(pool, per_class_total)

            # If with-replacement was used (pool smaller than needed), ensure
            # support and query subsets are *different* when possible
            if len(set(picks)) < per_class_total and len(pool) >= 2:
                # swap duplicates in the query slice when we can
                seen = set()
                uniq = []
                for p in picks:
                    if p in seen and len(seen) < len(pool):
                        # try to replace with something unseen
                        alt = random.choice(pool)
                        tries = 0
                        while alt in seen and tries < 8:
                            alt = random.choice(pool); tries += 1
                        uniq.append(alt)
                        seen.add(alt)
                    else:
                        uniq.append(p); seen.add(p)
                picks = uniq

            support_files = picks[:k_shot]
            query_files   = picks[k_shot:k_shot+q_query]

            # Load tensors (light aug for support, clean for query)
            for fp in support_files:
                s_imgs.append(self._load_tensor(fp, self.support_transform))
                s_lbls.append(new_label)
            for fp in query_files:
                q_imgs.append(self._load_tensor(fp, self.base_transform))
                q_lbls.append(new_label)

        sx = torch.stack(s_imgs, 0)
        qx = torch.stack(q_imgs, 0)
        sy = torch.tensor(s_lbls, dtype=torch.long)
        qy = torch.tensor(q_lbls, dtype=torch.long)
        return sx, sy, qx, qy
