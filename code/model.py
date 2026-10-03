"""GPU TEA-MMGR with aligned priors, event-interval CfC, and attention structure."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import AlignmentData


class IntervalCfCCell(nn.Module):
    def __init__(self, d_m: int, d_h: int, interval_gate: bool = False, gate_bias: float = 0.0):
        super().__init__()
        self.interval_gate = interval_gate
        self.w_f = nn.Linear(d_m + 1, d_h)
        self.u_f = nn.Linear(d_h, d_h, bias=False)
        self.w_o = nn.Linear(d_m, d_h)
        self.u_o = nn.Linear(d_h, d_h, bias=False)
        self.w_alpha = nn.Linear(d_m + d_h + 1, 1) if interval_gate else None
        if self.w_alpha is not None:
            nn.init.constant_(self.w_alpha.bias, gate_bias)

    def forward(self, m: torch.Tensor, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if self.w_alpha is not None:
            alpha = torch.sigmoid(self.w_alpha(torch.cat([m, h, dt], dim=-1)))
            dt = alpha * dt
        f = torch.sigmoid(self.w_f(torch.cat([m, dt], dim=-1)) + self.u_f(h))
        o = torch.tanh(self.w_o(m) + self.u_o(h))
        return f * h + (1.0 - f) * o


class RelAttnStruct(nn.Module):
    """Relation-aware attention over recency-sampled in/out neighbors."""

    def __init__(self, d_e: int, d_r: int, d_t: int, d_s: int):
        super().__init__()
        self.w_q = nn.Linear(d_e, d_s, bias=False)
        self.w_k = nn.Linear(d_e + d_r + d_t, d_s, bias=False)
        self.w_v = nn.Linear(d_e + d_r + d_t, d_s, bias=False)
        self.out = nn.Linear(d_s, d_s)
        self.scale = d_s ** 0.5

    def forward(
        self,
        center: torch.Tensor,
        nb: torch.Tensor,
        rel: torch.Tensor,
        time_pe: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat([nb, rel, time_pe], dim=-1)
        q = self.w_q(center)
        k = self.w_k(feat)
        v = self.w_v(feat)
        logits = (q.unsqueeze(1) * k).sum(-1) / self.scale
        logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        valid = (mask > 0).any(dim=-1, keepdim=True)
        attn = torch.where(valid, attn, torch.zeros_like(attn))
        attn = attn * (mask > 0).to(attn.dtype)
        pooled = (attn.unsqueeze(-1) * v).sum(1)
        return self.out(pooled)


class TEAMMGR(nn.Module):
    def __init__(
        self,
        data: AlignmentData,
        d_pre: int,
        d_glob: int = 256,
        d_e: int = 128,
        d_r: int = 64,
        d_s: int = 256,
        d_t: int = 64,
        d_m: int = 256,
        d_h: int = 128,
        d_v: int = 256,
        align_head: str = "siamese",
        siamese_hidden: tuple[int, int] | None = None,
        ablate: str = "none",
        interval_gate: bool = False,
        gate_bias: float = 0.0,
    ):
        super().__init__()
        if align_head not in {"siamese", "cosine"}:
            raise ValueError(f"unknown align_head={align_head}")
        allowed = {"none", "no_glob", "no_struct", "concat", "avg", "static", "gru", "mean"}
        if ablate not in allowed:
            raise ValueError(f"unknown ablate={ablate}")
        self.align_head = align_head
        self.ablate = ablate
        self.interval_gate = interval_gate
        self.gate_bias = gate_bias
        self.d_h = d_h
        self.d_s = d_s
        self.seq_len = data.seq_len

        self.ent_loc = nn.Embedding(data.num_ent, d_e)
        self.rel_emb = nn.Embedding(data.num_rel, d_r)
        nn.init.xavier_uniform_(self.ent_loc.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.w_glob = nn.Linear(d_pre, d_glob, bias=False)
        if d_pre == d_glob:
            nn.init.eye_(self.w_glob.weight)
        else:
            nn.init.xavier_uniform_(self.w_glob.weight)
        self.graph_bias = nn.Embedding(2, d_glob)
        nn.init.zeros_(self.graph_bias.weight)
        self.out_mix = nn.Parameter(torch.tensor(0.0))
        self.prior_head = nn.Linear(d_glob, d_v, bias=False)
        with torch.no_grad():
            w = torch.zeros(d_v, d_glob)
            dim = min(d_v, d_glob)
            w[:dim, :dim] = torch.eye(dim)
            self.prior_head.weight.copy_(w)
        self.w_tau = nn.Linear(d_t, d_t)
        self.w_event = nn.Linear(d_e + d_r + d_t, d_s)
        self.struct = RelAttnStruct(d_e, d_r, d_t, d_s)
        self.w_local = nn.Linear(2 * d_s, d_s)

        self.proj_glob = nn.Linear(d_glob, d_m)
        self.proj_struct = nn.Linear(d_s, d_m)
        self.proj_time = nn.Linear(d_t, d_m)
        self.gate = nn.Linear(d_glob + d_s + d_t + d_h, 3)
        self.fuse_norm = nn.LayerNorm(d_m)
        self.cell = IntervalCfCCell(d_m, d_h, interval_gate=interval_gate, gate_bias=gate_bias)
        self.gru = nn.GRUCell(d_m, d_h) if ablate == "gru" else None
        self.fuse_cat = nn.Linear(3 * d_m, d_m) if ablate == "concat" else None
        self.static_w = nn.Parameter(torch.zeros(3)) if ablate == "static" else None
        self.mean_proj = nn.Linear(d_m, d_h) if ablate == "mean" else None

        self.pool_q = nn.Linear(d_h, 1)
        seq_in = d_h + d_s
        self.seq_proj = nn.Linear(seq_in, d_v, bias=False)
        if seq_in == d_v:
            nn.init.eye_(self.seq_proj.weight)
        else:
            nn.init.xavier_uniform_(self.seq_proj.weight)
        self.siamese = None
        if align_head == "siamese":
            if siamese_hidden is None:
                hid1, hid2 = max(256, d_v), max(128, d_v // 2)
            else:
                hid1, hid2 = siamese_hidden
            self.siamese = nn.Sequential(
                nn.Linear(seq_in, hid1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hid1, hid2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hid2, d_v),
            )
        self.cos_gate = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("event_nb", data.event_nb, persistent=False)
        self.register_buffer("event_rel", data.event_rel, persistent=False)
        self.register_buffer("event_time", data.event_time, persistent=False)
        self.register_buffer("event_dt", data.event_dt, persistent=False)
        self.register_buffer("event_mask", data.event_mask, persistent=False)
        self.register_buffer("event_pe", data.event_pe, persistent=False)
        self.register_buffer("neigh_id", data.neigh_id, persistent=False)
        self.register_buffer("neigh_rel", data.neigh_rel, persistent=False)
        self.register_buffer("neigh_time", data.neigh_time, persistent=False)
        self.register_buffer("neigh_mask", data.neigh_mask, persistent=False)
        self.register_buffer("is_src", data.is_src, persistent=False)
        self.register_buffer("e_pre", torch.zeros(data.num_ent, d_pre), persistent=True)

    def set_global_prior(self, prior: torch.Tensor) -> None:
        if prior.shape != self.e_pre.shape:
            raise ValueError(f"prior {tuple(prior.shape)} != {tuple(self.e_pre.shape)}")
        self.e_pre.copy_(prior)

    def set_prior_trainable(self, trainable: bool) -> None:
        for module in (self.w_glob, self.prior_head, self.graph_bias):
            for p in module.parameters():
                p.requires_grad_(trainable)

    def _time_pe(self, times: torch.Tensor) -> torch.Tensor:
        dim = self.event_pe.size(-1)
        pos = times.clamp(0, 1) * 1000.0
        device, dtype = times.device, times.dtype
        div = torch.pow(
            torch.tensor(10000.0, device=device, dtype=dtype),
            torch.arange(0, dim, 2, device=device, dtype=dtype) / dim,
        )
        pe = torch.zeros(*times.shape, dim, device=device, dtype=dtype)
        pe[..., 0::2] = torch.sin(pos.unsqueeze(-1) / div)
        pe[..., 1::2] = torch.cos(pos.unsqueeze(-1) / div)
        return self.w_tau(pe)

    def _struct_step(self, ids: torch.Tensor, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
        center = self.ent_loc(ids)
        nb_id = self.neigh_id[ids]
        nb = self.ent_loc(nb_id)
        rel = self.rel_emb(self.neigh_rel[ids])
        ntime = self.neigh_time[ids]
        nmask = self.neigh_mask[ids]
        cur_t = self.event_time[ids, tau].unsqueeze(-1)
        step_mask = nmask * (ntime <= (cur_t + 1e-6)).to(nmask.dtype)
        step_mask = step_mask * self.event_mask[ids, tau].unsqueeze(-1)
        npe = self._time_pe(ntime)
        ctx = self.struct(center, nb, rel, npe, step_mask)

        ev_nb = self.ent_loc(self.event_nb[ids, tau])
        ev_rel = self.rel_emb(self.event_rel[ids, tau])
        ev_pe = self.w_tau(self.event_pe[ids, tau])
        ev = self.w_event(torch.cat([ev_nb, ev_rel, ev_pe], dim=-1))
        local = self.w_local(torch.cat([ctx, ev], dim=-1))
        return local, ctx

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = ids.size(0)
        h = torch.zeros(b, self.d_h, device=ids.device, dtype=self.ent_loc.weight.dtype)
        graph_id = (self.is_src[ids] < 0.5).long()
        u_glob = self.w_glob(self.e_pre[ids]) + self.graph_bias(graph_id)
        hs = []
        ctx_last = torch.zeros(b, self.d_s, device=ids.device, dtype=h.dtype)

        fused_steps = []
        for tau in range(self.seq_len):
            u_struct, ctx = self._struct_step(ids, tau)
            u_time = self.w_tau(self.event_pe[ids, tau])
            u_g = torch.zeros_like(u_glob) if self.ablate == "no_glob" else u_glob
            if self.ablate == "no_struct":
                u_struct = torch.zeros_like(u_struct)
            pg, ps, pt = self.proj_glob(u_g), self.proj_struct(u_struct), self.proj_time(u_time)
            if self.ablate == "concat":
                fused = self.fuse_norm(self.fuse_cat(torch.cat([pg, ps, pt], dim=-1)))
            elif self.ablate == "avg":
                fused = self.fuse_norm((pg + ps + pt) / 3.0)
            elif self.ablate == "static":
                sw = torch.softmax(self.static_w, dim=0)
                fused = self.fuse_norm(sw[0] * pg + sw[1] * ps + sw[2] * pt)
            else:
                gate = torch.sigmoid(self.gate(torch.cat([u_g, u_struct, u_time, h], dim=-1)))
                fused = self.fuse_norm(
                    gate[:, 0:1] * pg + gate[:, 1:2] * ps + gate[:, 2:3] * pt
                )
            dt = self.event_dt[ids, tau].unsqueeze(-1).to(dtype=fused.dtype)
            if self.ablate == "gru":
                h_new = self.gru(fused, h)
            else:
                h_new = self.cell(fused, h, dt)
            step = self.event_mask[ids, tau].unsqueeze(-1)
            h = torch.where(step.bool(), h_new, h)
            ctx_last = torch.where(step.bool(), ctx, ctx_last)
            hs.append(h)
            fused_steps.append(fused)

        hidden = torch.stack(hs, dim=1)
        mask = self.event_mask[ids]
        score = self.pool_q(hidden).squeeze(-1)
        score = score.masked_fill(mask <= 0, torch.finfo(score.dtype).min)
        alpha = torch.softmax(score, dim=-1) * mask
        denom = alpha.sum(-1, keepdim=True).clamp(min=1e-6)
        z = (alpha.unsqueeze(-1) * hidden).sum(1) / denom
        empty = mask.sum(-1, keepdim=True) <= 0
        z = torch.where(empty, torch.zeros_like(z), z)
        if self.ablate == "mean":
            fused_h = torch.stack(fused_steps, dim=1)
            z = self.mean_proj((fused_h * mask.unsqueeze(-1)).sum(1) / denom)
            z = torch.where(empty, torch.zeros_like(z), z)

        v_g = F.normalize(self.prior_head(u_glob), p=2, dim=-1)
        feat = torch.cat([z, ctx_last], dim=-1)
        if self.siamese is not None:
            v_s = F.normalize(self.siamese(feat), p=2, dim=-1)
        else:
            v_s = F.normalize(self.seq_proj(feat), p=2, dim=-1)
        return v_g, v_s, hidden, mask
