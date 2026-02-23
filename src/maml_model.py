# src/maml_model.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torchvision import models

GRAD_CLIP = 1.0  # conservative when we combine real + synthetic grads


# -------------------------
# EfficientNet-B0 backbone -> L2-normalized 1280-D features
# -------------------------
class EffB0Backbone(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.drop     = nn.Dropout(dropout_rate)
        self.out_dim  = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)      # [B, 1280]
        x = self.drop(x)
        return x

    def freeze_all(self, freeze=True):
        for p in self.parameters():
            p.requires_grad = not (freeze)

    def unfreeze_last_block(self):
        last = list(self.features.children())[-1]
        for p in last.parameters():
            p.requires_grad = True


# -------------------------
# Task Embedding via DeepSets
# -------------------------
class TaskSetEncoder(nn.Module):
    def __init__(self, feat_dim=1280, hidden=512, te_dim=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),   nn.ReLU(inplace=True),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, te_dim),
        )
        self.te_dim = te_dim

    def forward(self, feats, labels, n_way):
        # feats: [S, F], labels: [S]
        class_means = []
        for c in range(n_way):
            m = (labels == c)
            if m.any():
                class_means.append(self.phi(feats[m]).mean(0))
        if len(class_means) == 0:
            pooled = self.phi(feats).mean(0, keepdim=True)
        else:
            pooled = torch.stack(class_means, 0).mean(0, keepdim=True)
        return self.rho(pooled)  # [1, te_dim]


# -------------------------
# FiLM conditioning (gamma/beta from task embedding)
# -------------------------
class FiLM(nn.Module):
    def __init__(self, te_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(te_dim, feat_dim)
        self.beta  = nn.Linear(te_dim, feat_dim)

    def forward(self, z, te):
        g = self.gamma(te)
        b = self.beta(te)
        if g.shape[0] == 1 and z.shape[0] > 1:
            g = g.expand(z.size(0), -1)
            b = b.expand(z.size(0), -1)
        return z * (1 + g) + b


# -------------------------
# Task-conditioned head (FiLM + MLP classifier)
# -------------------------
class TaskCondHead(nn.Module):
    def __init__(self, feat_dim=1280, te_dim=128, hidden=512, n_way=5, p_drop=0.1):
        super().__init__()
        self.film = FiLM(te_dim, feat_dim)
        self.mlp  = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_way),
        )

    def forward(self, feats, task_embed):
        z = self.film(feats, task_embed)
        return self.mlp(z)


# -------------------------
# Full task-conditioned network
# -------------------------
class TaskCondNet(nn.Module):
    def __init__(self, n_way=5, te_dim=128):
        super().__init__()
        self.backbone = EffB0Backbone(dropout_rate=0.0)
        self.task_enc = TaskSetEncoder(feat_dim=self.backbone.out_dim, te_dim=te_dim)
        self.head     = TaskCondHead(feat_dim=self.backbone.out_dim, te_dim=te_dim, n_way=n_way)
        self.n_way    = n_way

        self.backbone_params = list(self.backbone.parameters())
        self.task_params     = list(self.task_enc.parameters())
        self.head_params     = list(self.head.parameters())

    def features(self, x):
        return self.backbone(x)

    def task_embed(self, support_feats, support_labels):
        return self.task_enc(support_feats, support_labels, self.n_way)

    def logits(self, feats, task_embed):
        return self.head(feats, task_embed)

    def freeze_backbone(self, freeze=True):
        for p in self.backbone_params:
            p.requires_grad = not (freeze)
        for p in self.task_params + self.head_params:
            p.requires_grad = True


# -------------------------
# MAML wrapper (Head + TaskEnc inner loop) + Reptile head update
# Backbone is SHARED into the adapted copy so query loss gives real grads.
# -------------------------
class MAML_TE(nn.Module):
    def __init__(self, model: TaskCondNet, inner_lr=0.03, inner_steps=12, meta_lr=8e-4, head_only=True):
        super().__init__()
        self.model        = model
        self.inner_lr     = inner_lr
        self.inner_steps  = inner_steps
        self.head_only    = head_only
        self.loss_fn      = nn.CrossEntropyLoss()
        self.meta_optimizer = optim.AdamW(self.model.parameters(), lr=meta_lr, weight_decay=2e-4)

    @torch.enable_grad()
    def adapt(self, sx, sy):
        """
        Returns an adapted copy:
          - backbone is SHARED with the base model (no duplication, real grads later)
          - inner SGD updates ONLY head + task encoder (head_only=True)
        """
        adapted = copy.deepcopy(self.model)

        # share the big conv tower to save VRAM and enable true outer grads
        adapted.backbone = self.model.backbone

        # disable backbone grads during inner steps (fast & stable)
        for p in adapted.backbone.parameters():
            p.requires_grad = False

        inner_params = list(adapted.head.parameters()) + list(adapted.task_enc.parameters())
        inner_opt = optim.SGD(inner_params, lr=self.inner_lr, momentum=0.9, nesterov=True)

        # inner loop on support
        for _ in range(self.inner_steps):
            s_feats = adapted.features(sx).detach()  # no backbone grads in inner loop
            task_e  = adapted.task_embed(s_feats, sy)
            s_logits = adapted.logits(s_feats, task_e)
            loss = self.loss_fn(s_logits, sy)
            inner_opt.zero_grad()
            loss.backward()
            clip_grad_norm_(inner_params, GRAD_CLIP)
            inner_opt.step()

        # re-enable backbone grads for outer step (query loss)
        for p in adapted.backbone.parameters():
            p.requires_grad = True

        # compute final task embedding with the updated head/task_enc
        with torch.no_grad():
            s_feats = adapted.features(sx)
            task_e  = adapted.task_embed(s_feats, sy)

        return adapted, task_e

    def accumulate_reptile_head(self, adapted, weight=1.0):
        """
        Synthetic grads for head + task encoder:
        grad := (theta - theta_adapted)  so step() moves base -> adapted
        """
        # head params
        for p, p_a in zip(self.model.head.parameters(), adapted.head.parameters()):
            if p.grad is None:
                p.grad = (p.data - p_a.data) * weight
            else:
                p.grad.data.add_(p.data - p_a.data, alpha=weight)
        # task encoder params
        for p, p_a in zip(self.model.task_enc.parameters(), adapted.task_enc.parameters()):
            if p.grad is None:
                p.grad = (p.data - p_a.data) * weight
            else:
                p.grad.data.add_(p.data - p_a.data, alpha=weight)

    def step(self):
        clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad(set_to_none=True)
