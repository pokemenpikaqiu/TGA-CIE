"""GPU RotatE pretraining and seed-supervised orthogonal alignment."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatE(nn.Module):
    def __init__(self, num_ent: int, num_rel: int, dim: int, gamma: float = 12.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotatE dim must be even")
        self.dim = dim
        self.gamma = gamma
        self.epsilon = 2.0
        self.ent = nn.Embedding(num_ent, dim)
        self.rel = nn.Embedding(num_rel, dim // 2)
        nn.init.uniform_(self.ent.weight, -gamma / dim, gamma / dim)
        nn.init.uniform_(self.rel.weight, -gamma / dim, gamma / dim)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        phase = r / (self.gamma / self.epsilon + self.dim / 2)
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)
        re_rot = re_h * re_r - im_h * im_r
        im_rot = re_h * im_r + im_h * re_r
        dist = torch.stack([re_rot - re_t, im_rot - im_t], dim=0).norm(p=2, dim=0).sum(dim=-1)
        return self.gamma - dist

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(self.ent(h), self.rel(r), self.ent(t))


def _remap_triples(triples: np.ndarray) -> Tuple[np.ndarray, dict, dict]:
    ents = np.unique(triples[:, [0, 2]])
    rels = np.unique(triples[:, 1])
    e_map = {int(e): i for i, e in enumerate(ents)}
    r_map = {int(r): i for i, r in enumerate(rels)}
    e_idx = np.full(int(ents.max()) + 1, -1, dtype=np.int64)
    r_idx = np.full(int(rels.max()) + 1, -1, dtype=np.int64)
    e_idx[ents] = np.arange(len(ents), dtype=np.int64)
    r_idx[rels] = np.arange(len(rels), dtype=np.int64)
    mapped = np.stack(
        [e_idx[triples[:, 0]], r_idx[triples[:, 1]], e_idx[triples[:, 2]]],
        axis=1,
    ).astype(np.int64)
    return mapped, e_map, r_map


@torch.no_grad()
def _export(model: RotatE, e_map: dict, num_ent: int, dim: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(num_ent, dim, device=device)
    local = torch.tensor(list(e_map.values()), device=device, dtype=torch.long)
    glob = torch.tensor(list(e_map.keys()), device=device, dtype=torch.long)
    out[glob] = model.ent(local)
    return out


def pretrain_rotate(
    triples: np.ndarray,
    num_ent_global: int,
    dim: int,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 4096,
    lr: float = 5e-4,
    neg_size: int = 16,
) -> torch.Tensor:
    mapped, e_map, _r_map = _remap_triples(triples)
    n_ent, n_rel = len(e_map), len(_r_map)
    model = RotatE(n_ent, n_rel, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(mapped).to(device)
    n = data.size(0)
    steps = max(n // batch_size, 1)

    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            if idx.numel() == 0:
                continue
            pos = data[idx]
            h, r, t = pos[:, 0], pos[:, 1], pos[:, 2]
            b = h.size(0)
            neg_t = torch.randint(0, n_ent, (b, neg_size), device=device)
            pos_s = model(h, r, t)
            neg_s = model(h.unsqueeze(1).expand_as(neg_t), r.unsqueeze(1).expand_as(neg_t), neg_t)
            loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [RotatE] epoch {epoch:03d}/{epochs} loss={total / steps:.4f}")

    model.eval()
    return _export(model, e_map, num_ent_global, dim, device).detach().cpu()


def ridge_map(src: torch.Tensor, tgt: torch.Tensor, lam: float = 5.0) -> torch.Tensor:
    """Least-squares map tgt -> src with ridge: W = (Y^T Y + λI)^{-1} Y^T X."""
    d = tgt.size(1)
    eye = torch.eye(d, device=tgt.device, dtype=tgt.dtype)
    return torch.linalg.solve(tgt.t() @ tgt + lam * eye, tgt.t() @ src)


def align_priors(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    pairs: torch.Tensor,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    ridge_lam: float = 25.0,
) -> torch.Tensor:
    xs = src_emb[pairs[:, 0]]
    yt = tgt_emb[pairs[:, 1]]
    w = ridge_map(xs, yt, lam=ridge_lam)
    prior = torch.zeros_like(src_emb)
    prior[src_ids] = src_emb[src_ids]
    prior[tgt_ids] = tgt_emb[tgt_ids] @ w
    with torch.no_grad():
        aligned = prior[pairs[:, 1]]
        gold = prior[pairs[:, 0]]
        cos = F.cosine_similarity(aligned, gold, dim=-1).mean().item()
    print(f"[rotate] ridge(lam={ridge_lam:g}) seed cosine={cos:.4f}")
    return prior


def build_global_prior(
    triples_src: np.ndarray,
    triples_tgt: np.ndarray,
    num_ent: int,
    dim: int,
    device: torch.device,
    cache_path: str,
    train_pairs: torch.Tensor,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    epochs: int = 40,
    ridge_lam: float = 25.0,
) -> torch.Tensor:
    split_path = cache_path.replace(".pt", "_split.pt")
    if os.path.isfile(split_path):
        print(f"[rotate] load split {split_path}")
        blob = torch.load(split_path, map_location="cpu", weights_only=True)
        src_emb, tgt_emb = blob["src"], blob["tgt"]
    else:
        print("[rotate] pretrain source KG on GPU")
        src_emb = pretrain_rotate(triples_src, num_ent, dim, device, epochs=epochs, neg_size=32)
        print("[rotate] pretrain target KG on GPU")
        tgt_emb = pretrain_rotate(triples_tgt, num_ent, dim, device, epochs=epochs, neg_size=32)
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        torch.save({"src": src_emb, "tgt": tgt_emb}, split_path)
        print(f"[rotate] saved {split_path}")

    pairs = train_pairs.cpu()
    return align_priors(src_emb, tgt_emb, pairs, src_ids.cpu(), tgt_ids.cpu(), ridge_lam=ridge_lam)
