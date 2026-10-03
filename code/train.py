"""Train / evaluate with CSLS, bidirectional check, bootstrap, and early stop."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_dataset, build_time_hist
from .model import TEAMMGR
from .rotate import align_priors, build_global_prior


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_ids(model: TEAMMGR, ids: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    gs, ss = [], []
    for start in range(0, ids.size(0), chunk):
        sl = ids[start : start + chunk]
        vg, vs, _, _ = model(sl)
        gs.append(vg.float())
        ss.append(vs.float())
    return torch.cat(gs, dim=0), torch.cat(ss, dim=0)


@torch.no_grad()
def encode_all(
    model: TEAMMGR, num_ent: int, device: torch.device, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    ids = torch.arange(num_ent, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return encode_ids(model, ids, chunk)


_TIME_SIM = ["idf"]


def _idf_unit(hist: torch.Tensor, idf: torch.Tensor) -> torch.Tensor:
    w = hist * idf
    return w / w.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _binary_jaccard(hist_a: torch.Tensor, hist_b: torch.Tensor) -> torch.Tensor:
    inter = hist_a @ hist_b.t()
    sa = hist_a.sum(-1, keepdim=True)
    sb = hist_b.sum(-1, keepdim=True).t()
    return inter / (sa + sb - inter).clamp(min=1.0)


def time_jaccard(hist_a: torch.Tensor, hist_b: torch.Tensor, idf: torch.Tensor) -> torch.Tensor:
    if _TIME_SIM[0] == "jaccard":
        return _binary_jaccard(hist_a, hist_b)
    return _idf_unit(hist_a, idf) @ _idf_unit(hist_b, idf).t()


def time_pair(hist: torch.Tensor, idf: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _TIME_SIM[0] == "jaccard":
        ha, hb = hist[a], hist[b]
        inter = (ha * hb).sum(-1)
        return inter / (ha.sum(-1) + hb.sum(-1) - inter).clamp(min=1.0)
    return (_idf_unit(hist[a], idf) * _idf_unit(hist[b], idf)).sum(-1)


def time_rand(hist: torch.Tensor, idf: torch.Tensor, src: torch.Tensor, rand: torch.Tensor) -> torch.Tensor:
    if _TIME_SIM[0] == "jaccard":
        ha = hist[src]
        hb = hist[rand]
        inter = torch.einsum("bd,bnd->bn", ha, hb)
        union = ha.sum(-1, keepdim=True) + hb.sum(-1) - inter
        return inter / union.clamp(min=1.0)
    wa = _idf_unit(hist[src], idf)
    wb = _idf_unit(hist[rand], idf)
    return (wa.unsqueeze(1) * wb).sum(-1)


def fuse_sim(emb: torch.Tensor, time_sim: torch.Tensor, w_time: float) -> torch.Tensor:
    if w_time <= 0:
        return emb
    return (1.0 - w_time) * emb + w_time * time_sim


def _fused_block(
    glob: torch.Tensor,
    seq: torch.Tensor,
    hist: torch.Tensor,
    idf: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    w_glob: float,
    w_time: float,
) -> torch.Tensor:
    emb = dual_sim(glob[src], seq[src], glob[tgt], seq[tgt], w_glob=w_glob)
    if w_time <= 0:
        return emb
    tj = time_jaccard(hist[src], hist[tgt], idf)
    return fuse_sim(emb, tj, w_time)


def _csls_rowcol(
    glob: torch.Tensor,
    seq: torch.Tensor,
    hist: torch.Tensor,
    idf: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    w_glob: float,
    w_time: float,
    k: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_s, n_t = int(src.numel()), int(tgt.numel())
    kk = min(k, n_t)
    r_s = torch.zeros(n_s, device=src.device, dtype=torch.float32)
    col_k = torch.full((n_t, kk), -1e9, device=src.device)
    for i in range(0, n_s, chunk):
        sl = src[i : i + chunk]
        sim = _fused_block(glob, seq, hist, idf, sl, tgt, w_glob, w_time)
        r_s[i : i + sl.numel()] = sim.topk(kk, dim=1).values.mean(dim=1)
        merged = torch.cat([col_k, sim.t()], dim=1)
        col_k = merged.topk(kk, dim=1).values
        del sim, merged
    return r_s, col_k.mean(dim=1)


def _metrics_chunked(
    glob: torch.Tensor,
    seq: torch.Tensor,
    hist: torch.Tensor,
    idf: torch.Tensor,
    src: torch.Tensor,
    tgt: torch.Tensor,
    w_glob: float,
    w_time: float,
    chunk: int,
    r_s: torch.Tensor | None = None,
    r_t: torch.Tensor | None = None,
) -> dict:
    n = int(src.numel())
    rank_sum = inv_sum = h1 = h10 = 0.0
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        sim = _fused_block(glob, seq, hist, idf, src[i:end], tgt, w_glob, w_time)
        if r_s is not None:
            sim = 2.0 * sim - r_s[i:end].unsqueeze(1) - r_t.unsqueeze(0)
        gold = torch.arange(i, end, device=sim.device)
        gold_score = sim[torch.arange(end - i, device=sim.device), gold]
        ranks = (sim >= gold_score.unsqueeze(1)).sum(dim=1).float()
        rank_sum += float(ranks.sum().item())
        inv_sum += float((1.0 / ranks).sum().item())
        h1 += float((ranks <= 1).sum().item())
        h10 += float((ranks <= 10).sum().item())
        del sim
    return {"mr": rank_sum / n, "mrr": inv_sum / n, "hits1": h1 / n, "hits10": h10 / n}


def dual_sim(
    src_g: torch.Tensor,
    src_s: torch.Tensor,
    tgt_g: torch.Tensor,
    tgt_s: torch.Tensor,
    w_glob: float = 0.5,
) -> torch.Tensor:
    """Weighted sum of prior cosine and sequential cosine."""
    src_g = F.normalize(src_g, dim=-1)
    src_s = F.normalize(src_s, dim=-1)
    tgt_g = F.normalize(tgt_g, dim=-1)
    tgt_s = F.normalize(tgt_s, dim=-1)
    return w_glob * (src_g @ tgt_g.t()) + (1.0 - w_glob) * (src_s @ tgt_s.t())


def csls(sim: torch.Tensor, k: int = 10) -> torch.Tensor:
    kk = min(k, sim.size(1), sim.size(0))
    r_s = torch.topk(sim, kk, dim=1).values.mean(dim=1, keepdim=True)
    r_t = torch.topk(sim, min(k, sim.size(0)), dim=0).values.mean(dim=0, keepdim=True)
    return 2.0 * sim - r_s - r_t


def metrics_from_ranks(ranks: torch.Tensor) -> dict:
    ranks = ranks.float()
    return {
        "mr": float(ranks.mean().item()),
        "mrr": float((1.0 / ranks).mean().item()),
        "hits1": float((ranks <= 1).float().mean().item()),
        "hits10": float((ranks <= 10).float().mean().item()),
    }


def _ranks_from_sim(sim: torch.Tensor) -> torch.Tensor:
    gold = torch.arange(sim.size(0), device=sim.device)
    gold_score = sim.gather(1, gold.view(-1, 1))
    return (sim >= gold_score).sum(dim=1)


def _both_sides(sim: torch.Tensor) -> tuple[dict, dict]:
    left = metrics_from_ranks(_ranks_from_sim(sim))
    right = metrics_from_ranks(_ranks_from_sim(sim.t()))
    return left, right


@torch.no_grad()
def evaluate(
    model: TEAMMGR,
    data,
    device: torch.device,
    chunk: int,
    csls_k: int,
    w_glob: float = 0.5,
    w_time: float = 0.0,
) -> dict:
    model.eval()
    glob, seq = encode_all(model, data.num_ent, device, chunk)
    src, tgt = data.test_pairs[:, 0], data.test_pairs[:, 1]
    n = int(src.numel())
    extra = n <= 12000
    hist, idf = data.time_hist, data.time_idf
    if extra:
        emb = dual_sim(glob[src], seq[src], glob[tgt], seq[tgt], w_glob=w_glob)
        tj = time_jaccard(hist[src], hist[tgt], idf)
        sim_cos = fuse_sim(emb, tj, w_time)
        del emb
        left_t, right_t = _both_sides(tj)
        del tj
        left_cos, right_cos = _both_sides(sim_cos)
        sim_csls = csls(sim_cos, k=csls_k)
        del sim_cos
        left_csls, right_csls = _both_sides(sim_csls)
        del sim_csls
        sim_g = F.normalize(glob[src], dim=-1) @ F.normalize(glob[tgt], dim=-1).t()
        left_g, right_g = _both_sides(sim_g)
        del sim_g
        sim_s = F.normalize(seq[src], dim=-1) @ F.normalize(seq[tgt], dim=-1).t()
        left_s, right_s = _both_sides(sim_s)
        del sim_s
        mrr_glob = 0.5 * (left_g["mrr"] + right_g["mrr"])
        mrr_seq = 0.5 * (left_s["mrr"] + right_s["mrr"])
        mrr_time = 0.5 * (left_t["mrr"] + right_t["mrr"])
    else:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        r_s, r_t = _csls_rowcol(glob, seq, hist, idf, src, tgt, w_glob, w_time, csls_k, chunk)
        left_cos = _metrics_chunked(glob, seq, hist, idf, src, tgt, w_glob, w_time, chunk)
        right_cos = _metrics_chunked(glob, seq, hist, idf, tgt, src, w_glob, w_time, chunk)
        left_csls = _metrics_chunked(glob, seq, hist, idf, src, tgt, w_glob, w_time, chunk, r_s, r_t)
        right_csls = _metrics_chunked(glob, seq, hist, idf, tgt, src, w_glob, w_time, chunk, r_t, r_s)
        mrr_glob = mrr_seq = mrr_time = 0.0
    del glob, seq
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pick_left = left_csls if left_csls["mrr"] >= left_cos["mrr"] else left_cos
    pick_right = right_csls if right_csls["mrr"] >= right_cos["mrr"] else right_cos
    return {
        "mrr": 0.5 * (pick_left["mrr"] + pick_right["mrr"]),
        "hits1": 0.5 * (pick_left["hits1"] + pick_right["hits1"]),
        "hits10": 0.5 * (pick_left["hits10"] + pick_right["hits10"]),
        "mr": 0.5 * (pick_left["mr"] + pick_right["mr"]),
        "mrr_l2r": pick_left["mrr"],
        "hits1_l2r": pick_left["hits1"],
        "hits10_l2r": pick_left["hits10"],
        "mrr_cos": 0.5 * (left_cos["mrr"] + right_cos["mrr"]),
        "mrr_csls": 0.5 * (left_csls["mrr"] + right_csls["mrr"]),
        "mrr_glob": mrr_glob,
        "mrr_seq": mrr_seq,
        "mrr_time": mrr_time,
    }


@torch.no_grad()
def bootstrap_pairs(
    glob: torch.Tensor,
    seq: torch.Tensor,
    data,
    train_pairs: torch.Tensor,
    csls_k: int,
    max_add: int,
    min_score: float,
    min_margin: float,
    w_glob: float = 0.5,
    w_time: float = 0.0,
    chunk: int = 2048,
) -> torch.Tensor:
    src_ids, tgt_ids = data.src_ids, data.tgt_ids
    n_s, n_t = int(src_ids.numel()), int(tgt_ids.numel())
    used_s = torch.zeros(n_s, dtype=torch.bool, device=glob.device)
    used_t = torch.zeros(n_t, dtype=torch.bool, device=glob.device)
    local_s = {int(g): i for i, g in enumerate(src_ids.tolist())}
    local_t = {int(g): i for i, g in enumerate(tgt_ids.tolist())}
    for a, b in train_pairs.tolist():
        if a in local_s:
            used_s[local_s[a]] = True
        if b in local_t:
            used_t[local_t[b]] = True
    hist, idf = data.time_hist, data.time_idf
    r_s, r_t = _csls_rowcol(
        glob, seq, hist, idf, src_ids, tgt_ids, w_glob, w_time, csls_k, chunk
    )
    pred_t = torch.zeros(n_s, dtype=torch.long, device=glob.device)
    top1 = torch.full((n_s,), -1e9, device=glob.device)
    top2 = torch.full((n_s,), -1e9, device=glob.device)
    col_best = torch.full((n_t,), -1e9, device=glob.device)
    col_arg = torch.zeros(n_t, dtype=torch.long, device=glob.device)
    for i in range(0, n_s, chunk):
        end = min(i + chunk, n_s)
        sim = _fused_block(glob, seq, hist, idf, src_ids[i:end], tgt_ids, w_glob, w_time)
        sim = 2.0 * sim - r_s[i:end].unsqueeze(1) - r_t.unsqueeze(0)
        sim[used_s[i:end]] = -1e9
        sim[:, used_t] = -1e9
        vals, inds = sim.topk(k=min(2, n_t), dim=1)
        pred_t[i:end] = inds[:, 0]
        top1[i:end] = vals[:, 0]
        top2[i:end] = vals[:, 1]
        chunk_best, chunk_arg = sim.max(dim=0)
        better = chunk_best > col_best
        col_best = torch.where(better, chunk_best, col_best)
        col_arg = torch.where(better, chunk_arg + i, col_arg)
        del sim, vals, inds
    src_idx = torch.arange(n_s, device=glob.device)
    mutual = (col_arg[pred_t] == src_idx) & (~used_s)
    keep = mutual & (top1 > min_score) & ((top1 - top2) > min_margin)
    if keep.sum() == 0:
        print("  [bootstrap] no high-confidence pairs")
        return train_pairs
    order = torch.argsort(top1[keep], descending=True)
    sel_s = src_idx[keep][order][:max_add]
    sel_t = pred_t[keep][order][:max_add]
    extra = torch.stack([src_ids[sel_s], tgt_ids[sel_t]], dim=1)
    sc = top1[keep][order][:max_add]
    print(
        f"  [bootstrap] added {extra.size(0)} pairs  "
        f"score[{float(sc.min()):.3f},{float(sc.max()):.3f}]  "
        f"candidates={int(keep.sum())}"
    )
    return torch.cat([train_pairs, extra], dim=0)


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"[env] {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    data_dir = os.path.join(args.data_root, args.dataset)
    cache_dir = os.path.join(args.work_dir, "cache")
    os.makedirs(args.work_dir, exist_ok=True)

    data = build_dataset(
        data_dir,
        n_seed=args.n_seed,
        seq_len=args.seq_len,
        n_max=args.n_max,
        d_time=args.d_t,
        cache_dir=cache_dir,
        event_sample=args.event_sample,
    ).to(device)
    _TIME_SIM[0] = args.time_sim
    th, tsz, tidf = build_time_hist(data.triples_src, data.triples_tgt, data.num_ent)
    data.time_hist = th.to(device)
    data.time_size = tsz.to(device)
    data.time_idf = tidf.to(device)

    prior_path = os.path.join(cache_dir, f"{args.dataset}_rotate_d{args.d_pre}.pt")
    prior = build_global_prior(
        data.triples_src,
        data.triples_tgt,
        data.num_ent,
        args.d_pre,
        device,
        prior_path,
        data.train_pairs,
        data.src_ids,
        data.tgt_ids,
        epochs=args.rotate_epochs,
        ridge_lam=args.ridge_lam,
    ).to(device)
    split_path = prior_path.replace(".pt", "_split.pt")
    rot_blob = torch.load(split_path, map_location="cpu", weights_only=True)
    src_emb, tgt_emb = rot_blob["src"], rot_blob["tgt"]

    def refresh_prior(pairs: torch.Tensor) -> None:
        new_prior = align_priors(
            src_emb,
            tgt_emb,
            pairs.cpu(),
            data.src_ids.cpu(),
            data.tgt_ids.cpu(),
            ridge_lam=args.ridge_lam,
        )
        model.set_global_prior(new_prior.to(device))

    model = TEAMMGR(
        data,
        d_pre=args.d_pre,
        d_glob=args.d_glob,
        d_e=args.d_e,
        d_r=args.d_r,
        d_s=args.d_s,
        d_t=args.d_t,
        d_m=args.d_m,
        d_h=args.d_h,
        d_v=args.d_v,
        align_head=args.align_head,
        ablate=args.ablate,
        interval_gate=args.interval_gate,
        gate_bias=args.gate_bias,
    ).to(device)
    model.set_global_prior(prior)
    model.e_pre.requires_grad_(False)
    if args.freeze_prior > 0:
        model.set_prior_trainable(False)

    def make_optimizer() -> torch.optim.Adam:
        prior_names = ("w_glob", "graph_bias", "prior_head")
        prior_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in prior_names)
        ]
        rest = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(k in n for k in prior_names)
        ]
        groups = [{"params": rest, "lr": args.lr * lr_scale}]
        if prior_params:
            groups.insert(0, {"params": prior_params, "lr": args.lr * args.prior_lr_mult * lr_scale})
        return torch.optim.Adam(groups, weight_decay=1e-5)

    lr_scale = 1.0
    opt = make_optimizer()

    train_pairs = data.train_pairs.clone()
    n_tgt = data.tgt_ids.size(0)
    bank_g = torch.zeros(n_tgt, args.d_v, device=device)
    bank_s = torch.zeros(n_tgt, args.d_v, device=device)
    local = torch.full((data.num_ent,), -1, device=device, dtype=torch.long)
    local[data.tgt_ids] = torch.arange(n_tgt, device=device)

    best = {"mrr": -1.0}
    bad = 0
    tag = args.tag or f"seed{args.n_seed}"
    ckpt = os.path.join(args.work_dir, f"{args.dataset}_{tag}_best.pt")
    log_path = os.path.join(args.work_dir, f"{args.dataset}_{tag}.log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(
        f"train pairs={train_pairs.size(0)} test pairs={data.test_pairs.size(0)}  "
        f"d_pre={args.d_pre} d_v={args.d_v} L={args.seq_len} K={args.n_max}  "
        f"head={args.align_head} ablate={args.ablate} interval_gate={int(args.interval_gate)} "
        f"gate_bias={args.gate_bias:g} "
        f"score=dual+{args.time_sim}-time sample={args.event_sample} "
        f"w_glob={args.w_glob:g} w_time={args.w_time:g} csls_k={args.csls_k} "
        f"freeze_prior={args.freeze_prior} "
        f"boot_min={args.boot_min} boot_margin={args.boot_margin} "
        f"ridge_lam={args.ridge_lam} tag={tag}"
    )
    log("init target memory bank")
    with torch.no_grad():
        g0, s0 = encode_all(model, data.num_ent, device, args.chunk)
        bank_g.copy_(F.normalize(g0[data.tgt_ids], dim=-1))
        bank_s.copy_(F.normalize(s0[data.tgt_ids], dim=-1))
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.freeze_prior > 0 and epoch == args.freeze_prior + 1:
            model.set_prior_trainable(True)
            opt = make_optimizer()
            log(f"  unfroze prior adapters at epoch {epoch}, prior_lr={args.lr * args.prior_lr_mult:g}")
        model.train()
        n_train = train_pairs.size(0)
        steps = max((n_train + args.batch_size - 1) // args.batch_size, 1)
        perm = torch.randperm(n_train, device=device)
        ep_align = 0.0
        ep_temp = 0.0
        for s in range(steps):
            idx = perm[s * args.batch_size : (s + 1) * args.batch_size]
            pairs = train_pairs[idx]
            src, pos = pairs[:, 0], pairs[:, 1]
            b = src.size(0)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                sg, ss, _, _ = model(src)
                emb = dual_sim(sg.float(), ss.float(), bank_g, bank_s, w_glob=args.w_glob)
                tj = time_jaccard(
                    data.time_hist[src],
                    data.time_hist[data.tgt_ids],
                    data.time_idf,
                )
                sim = csls(fuse_sim(emb, tj, args.w_time), k=args.csls_k)
                pos_local = local[pos]
                sim[torch.arange(b, device=device), pos_local] = torch.finfo(sim.dtype).min
                hard = data.tgt_ids[sim.argmax(dim=1)]
            rand = data.tgt_ids[torch.randint(0, n_tgt, (b, args.n_neg), device=device)]
            ids = torch.cat([src, pos, hard, rand.reshape(-1)], dim=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vg, vs, hidden, mask = model(ids)
                def split(x: torch.Tensor):
                    return x[:b], x[b : 2 * b], x[2 * b : 3 * b], x[3 * b :].view(b, args.n_neg, -1)

                gs, gp, gh, gr = split(vg)
                ss, sp, sh, sr = split(vs)
                wg, ws, wt = args.w_glob, 1.0 - args.w_glob, args.w_time
                emb_pos = wg * (gs * gp).sum(-1) + ws * (ss * sp).sum(-1)
                emb_hard = wg * (gs * gh).sum(-1) + ws * (ss * sh).sum(-1)
                emb_rand = wg * torch.einsum("bd,bnd->bn", gs, gr) + ws * torch.einsum("bd,bnd->bn", ss, sr)
                t_pos = time_pair(data.time_hist, data.time_idf, src, pos)
                t_hard = time_pair(data.time_hist, data.time_idf, src, hard)
                t_rand = time_rand(data.time_hist, data.time_idf, src, rand)
                s_pos = fuse_sim(emb_pos, t_pos, wt).unsqueeze(-1)
                s_hard = fuse_sim(emb_hard, t_hard, wt).unsqueeze(-1)
                s_rand = fuse_sim(emb_rand, t_rand, wt)
                logits = torch.cat([s_pos, s_hard, s_rand], dim=1) / args.tau
                loss_align = F.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=device))
                dh = hidden[: 2 * b, 1:] - hidden[: 2 * b, :-1]
                trans = mask[: 2 * b, 1:]
                loss_temp = (dh.pow(2).mean(-1) * trans).sum() / trans.sum().clamp(min=1.0)
                loss = loss_align + args.lam * loss_temp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                bank_g[local[pos]] = F.normalize(0.9 * bank_g[local[pos]] + 0.1 * gp.float(), dim=-1)
                bank_s[local[pos]] = F.normalize(0.9 * bank_s[local[pos]] + 0.1 * sp.float(), dim=-1)
            ep_align += float(loss_align.item())
            ep_temp += float(loss_temp.item())

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            res = evaluate(
                model, data, device, args.chunk, args.csls_k, w_glob=args.w_glob, w_time=args.w_time
            )
            msg = (
                f"epoch {epoch:03d}/{args.epochs}  pairs={train_pairs.size(0)}  "
                f"align={ep_align/steps:.4f} temp={ep_temp/steps:.4f}  "
                f"MRR={res['mrr']:.4f} H@1={res['hits1']:.4f} H@10={res['hits10']:.4f}  "
                f"cos={res.get('mrr_cos', 0):.4f} csls={res.get('mrr_csls', 0):.4f}  "
                f"glob={res.get('mrr_glob', 0):.4f} seq={res.get('mrr_seq', 0):.4f} "
                f"time={res.get('mrr_time', 0):.4f}  "
                f"L2R H@1={res['hits1_l2r']:.4f}  elapsed={time.time()-t0:.1f}s"
            )
            log(msg)
            if res["mrr"] > best["mrr"] + 1e-5:
                best = res
                bad = 0
                torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": res}, ckpt)
                log(f"  saved best -> {ckpt}")
            else:
                bad += 1
                if bad >= args.patience:
                    log(f"early stop at epoch {epoch}, best MRR={best['mrr']:.4f}")
                    break
            if epoch >= args.boot_start and res["mrr"] >= args.boot_mrr:
                n_before = train_pairs.size(0)
                glob, seqv = encode_all(model, data.num_ent, device, args.chunk)
                train_pairs = bootstrap_pairs(
                    glob,
                    seqv,
                    data,
                    train_pairs,
                    args.csls_k,
                    args.boot_max,
                    args.boot_min,
                    args.boot_margin,
                    args.w_glob,
                    args.w_time,
                    args.chunk,
                )
                if train_pairs.size(0) > n_before:
                    refresh_prior(train_pairs)
                    log(f"  re-aligned ridge prior with {train_pairs.size(0)} pairs")
                    if lr_scale > args.boot_lr_mult + 1e-12:
                        lr_scale = args.boot_lr_mult
                        opt = make_optimizer()
                        log(f"  lowered lr x{args.boot_lr_mult:g} after bootstrap")
                    post = evaluate(
                        model, data, device, args.chunk, args.csls_k, w_glob=args.w_glob, w_time=args.w_time
                    )
                    log(
                        f"  after bootstrap  MRR={post['mrr']:.4f} H@1={post['hits1']:.4f} "
                        f"H@10={post['hits10']:.4f} glob={post.get('mrr_glob', 0):.4f} "
                        f"seq={post.get('mrr_seq', 0):.4f} time={post.get('mrr_time', 0):.4f}"
                    )
                    if post["mrr"] > best["mrr"] + 1e-5:
                        best = post
                        bad = 0
                        torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": post}, ckpt)
                        log(f"  saved best -> {ckpt}")
                    g1, s1 = encode_all(model, data.num_ent, device, args.chunk)
                    bank_g.copy_(F.normalize(g1[data.tgt_ids], dim=-1))
                    bank_s.copy_(F.normalize(s1[data.tgt_ids], dim=-1))
                else:
                    bank_g.copy_(F.normalize(glob[data.tgt_ids], dim=-1))
                    bank_s.copy_(F.normalize(seqv[data.tgt_ids], dim=-1))
        else:
            log(
                f"epoch {epoch:03d}/{args.epochs}  pairs={train_pairs.size(0)}  "
                f"align={ep_align/steps:.4f} temp={ep_temp/steps:.4f}"
            )

    log(
        f"best MRR={best['mrr']:.4f} H@1={best.get('hits1', 0):.4f} "
        f"H@10={best.get('hits10', 0):.4f} L2R H@1={best.get('hits1_l2r', 0):.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/autodl-tmp/TEA-MMGR/data")
    p.add_argument("--work_dir", default="/root/autodl-tmp/TEA-MMGR")
    p.add_argument("--dataset", default="ICEWS05-15", choices=["ICEWS05-15", "YAGO-WIKI50K"])
    p.add_argument("--n_seed", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--n_max", type=int, default=32)
    p.add_argument("--d_pre", type=int, default=256)
    p.add_argument("--d_glob", type=int, default=256)
    p.add_argument("--d_e", type=int, default=128)
    p.add_argument("--d_r", type=int, default=64)
    p.add_argument("--d_s", type=int, default=256)
    p.add_argument("--d_t", type=int, default=64)
    p.add_argument("--d_m", type=int, default=256)
    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--d_v", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--rotate_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--n_neg", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--boot_start", type=int, default=20)
    p.add_argument("--boot_max", type=int, default=100)
    p.add_argument("--boot_min", type=float, default=0.15)
    p.add_argument("--boot_margin", type=float, default=0.04)
    p.add_argument("--boot_mrr", type=float, default=0.40)
    p.add_argument("--boot_lr_mult", type=float, default=0.4)
    p.add_argument("--freeze_prior", type=int, default=0)
    p.add_argument("--prior_lr_mult", type=float, default=0.15)
    p.add_argument("--ridge_lam", type=float, default=25.0)
    p.add_argument("--w_glob", type=float, default=0.5)
    p.add_argument("--w_time", type=float, default=0.3)
    p.add_argument("--event_sample", default="strat", choices=["strat", "linspace"])
    p.add_argument("--time_sim", default="idf", choices=["idf", "jaccard"])
    p.add_argument("--align_head", default="siamese", choices=["siamese", "cosine"])
    p.add_argument(
        "--ablate",
        default="none",
        choices=["none", "no_glob", "no_struct", "concat", "avg", "static", "gru", "mean"],
    )
    p.add_argument("--interval_gate", action="store_true")
    p.add_argument("--gate_bias", type=float, default=0.0)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
