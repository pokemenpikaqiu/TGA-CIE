"""Event-sequence TKG tensors with real intervals and recency-sampled neighbors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


def _read_id_file(path: str) -> dict:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if parts[0].lstrip("-").isdigit():
                idx, name = int(parts[0]), parts[1]
            else:
                name, idx = parts[0], int(parts[1])
            mapping[idx] = name
    return mapping


def _read_pairs(path: str) -> np.ndarray:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            pairs.append((int(a), int(b)))
    return np.asarray(pairs, dtype=np.int64)


def _read_triples(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                h, r, t, ts, te = map(int, parts[:5])
                rows.append((h, r, t, ts, te))
            elif len(parts) == 4:
                h, r, t, tm = map(int, parts)
                rows.append((h, r, t, tm, tm))
    return np.asarray(rows, dtype=np.int64)


def sinusoidal_pe(times: np.ndarray, dim: int) -> np.ndarray:
    times = np.clip(times.astype(np.float32), 0.0, 1.0)
    pe = np.zeros(times.shape + (dim,), dtype=np.float32)
    position = times * 1000.0
    div = np.power(10000.0, np.arange(0, dim, 2, dtype=np.float32) / dim)
    pe[..., 0::2] = np.sin(position[..., None] / div)
    pe[..., 1::2] = np.cos(position[..., None] / div[: pe.shape[-1] // 2])
    return pe


def _split_seeds(sup: np.ndarray, ref: np.ndarray, n_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_seed >= len(sup):
        return sup, ref
    return sup[:n_seed], np.concatenate([sup[n_seed:], ref], axis=0)


def build_time_hist(
    triples_src: np.ndarray, triples_tgt: np.ndarray, num_ent: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary timestamp occupancy plus IDF weights for time-set matching."""
    all_t = np.concatenate([triples_src, triples_tgt], axis=0)
    stamps = np.unique(np.concatenate([all_t[:, 3], all_t[:, 4]], axis=0)).astype(np.int64)
    tmap = {int(t): i for i, t in enumerate(stamps)}
    n_time = int(stamps.size)
    hist = np.zeros((num_ent, n_time), dtype=np.float32)
    for h, _r, t, ts, te in all_t:
        hh, tt = int(h), int(t)
        for tm in (int(ts), int(te)):
            j = tmap[tm]
            hist[hh, j] = 1.0
            hist[tt, j] = 1.0
    size = hist.sum(axis=1)
    df = np.clip(hist.sum(axis=0), 1.0, None)
    idf = np.log((float(num_ent) + 1.0) / df).astype(np.float32)
    print(
        f"[data] time hist |T|={n_time} nonzero={int((size > 0).sum())} "
        f"idf[min={idf.min():.3f}, max={idf.max():.3f}]"
    )
    return torch.from_numpy(hist), torch.from_numpy(size), torch.from_numpy(idf)


def _select_linspace(n: int, seq_len: int) -> np.ndarray:
    if n <= seq_len:
        return np.arange(n, dtype=np.int64)
    idx = np.unique(np.round(np.linspace(0, n - 1, seq_len)).astype(np.int64))
    if idx.size < seq_len:
        taken = set(int(i) for i in idx)
        fill = np.asarray([i for i in range(n) if i not in taken], dtype=np.int64)
        idx = np.unique(np.concatenate([idx, fill[: seq_len - idx.size]]))
    return np.sort(idx[:seq_len])


def _select_indices(n: int, seq_len: int, mode: str = "strat") -> np.ndarray:
    """Keep the most recent half of the budget; stratify the earlier span."""
    if mode == "linspace":
        return _select_linspace(n, seq_len)
    if n <= seq_len:
        return np.arange(n, dtype=np.int64)
    n_recent = max(seq_len // 2, 1)
    n_span = seq_len - n_recent
    recent = np.arange(n - n_recent, n, dtype=np.int64)
    n_early = n - n_recent
    if n_span <= 0:
        return recent
    if n_early <= n_span:
        early = np.arange(n_early, dtype=np.int64)
    else:
        edges = np.linspace(0, n_early, n_span + 1)
        early = np.empty(n_span, dtype=np.int64)
        for i in range(n_span):
            lo = int(np.floor(edges[i]))
            hi = int(np.ceil(edges[i + 1]))
            hi = min(max(hi, lo + 1), n_early)
            early[i] = (lo + hi - 1) // 2
    idx = np.unique(np.concatenate([early, recent]))
    if idx.size < seq_len:
        taken = set(int(i) for i in idx)
        fill = [i for i in range(n - 1, -1, -1) if i not in taken]
        extra = np.asarray(fill[: seq_len - idx.size], dtype=np.int64)
        idx = np.unique(np.concatenate([idx, extra]))
    return np.sort(idx[:seq_len])


@dataclass
class AlignmentData:
    num_ent: int
    num_rel: int
    seq_len: int
    n_max: int
    src_ids: torch.Tensor
    tgt_ids: torch.Tensor
    is_src: torch.Tensor
    train_pairs: torch.Tensor
    test_pairs: torch.Tensor
    event_nb: torch.Tensor
    event_rel: torch.Tensor
    event_time: torch.Tensor
    event_dt: torch.Tensor
    event_mask: torch.Tensor
    event_pe: torch.Tensor
    neigh_id: torch.Tensor
    neigh_rel: torch.Tensor
    neigh_time: torch.Tensor
    neigh_mask: torch.Tensor
    triples_src: np.ndarray
    triples_tgt: np.ndarray

    def to(self, device: torch.device) -> "AlignmentData":
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device, non_blocking=True))
        return self


def build_dataset(
    data_dir: str,
    n_seed: int,
    seq_len: int = 16,
    n_max: int = 32,
    d_time: int = 64,
    cache_dir: str | None = None,
    event_sample: str = "strat",
) -> AlignmentData:
    name = os.path.basename(os.path.normpath(data_dir))
    suffix = "seq_strat" if event_sample == "strat" else "seq"
    cache_key = f"{name}_seed{n_seed}_L{seq_len}_K{n_max}_{suffix}.pt"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_key)
        if os.path.isfile(cache_path):
            print(f"[data] load cache {cache_path}")
            return torch.load(cache_path, weights_only=False)

    ent1 = _read_id_file(os.path.join(data_dir, "ent_ids_1"))
    ent2 = _read_id_file(os.path.join(data_dir, "ent_ids_2"))
    src_ids = np.asarray(sorted(ent1), dtype=np.int64)
    tgt_ids = np.asarray(sorted(ent2), dtype=np.int64)
    triples1 = _read_triples(os.path.join(data_dir, "triples_1"))
    triples2 = _read_triples(os.path.join(data_dir, "triples_2"))
    sup = _read_pairs(os.path.join(data_dir, "sup_pairs"))
    ref = _read_pairs(os.path.join(data_dir, "ref_pairs"))
    train_pairs, test_pairs = _split_seeds(sup, ref, n_seed)

    all_triples = np.concatenate([triples1, triples2], axis=0)
    num_ent = int(max(src_ids.max(), tgt_ids.max(), all_triples[:, [0, 2]].max()) + 1)
    num_rel = int(all_triples[:, 1].max() + 1)
    times = all_triples[:, 3].astype(np.float32)
    t_min, t_max = float(times.min()), float(times.max())
    span = max(t_max - t_min, 1.0)

    events: List[List[Tuple[int, int, float]]] = [[] for _ in range(num_ent)]
    for h, r, t, tm, _te in all_triples:
        tn = (float(tm) - t_min) / span
        hh, rr, tt = int(h), int(r), int(t)
        events[hh].append((tt, rr, tn))
        events[tt].append((hh, rr + num_rel, tn))

    event_nb = np.zeros((num_ent, seq_len), dtype=np.int64)
    event_rel = np.zeros((num_ent, seq_len), dtype=np.int64)
    event_time = np.zeros((num_ent, seq_len), dtype=np.float32)
    event_dt = np.zeros((num_ent, seq_len), dtype=np.float32)
    event_mask = np.zeros((num_ent, seq_len), dtype=np.float32)
    event_pe = np.zeros((num_ent, seq_len, d_time), dtype=np.float32)
    neigh_id = np.zeros((num_ent, n_max), dtype=np.int64)
    neigh_rel = np.zeros((num_ent, n_max), dtype=np.int64)
    neigh_time = np.zeros((num_ent, n_max), dtype=np.float32)
    neigh_mask = np.zeros((num_ent, n_max), dtype=np.float32)

    nonempty = 0
    for e in range(num_ent):
        ev = events[e]
        if not ev:
            continue
        nonempty += 1
        ev.sort(key=lambda x: x[2])
        idx = _select_indices(len(ev), seq_len, event_sample)
        chosen = [ev[i] for i in idx]
        start = seq_len - len(chosen)
        prev_t = None
        for i, (nb, rel, tn) in enumerate(chosen):
            p = start + i
            event_nb[e, p] = nb
            event_rel[e, p] = rel
            event_time[e, p] = tn
            event_mask[e, p] = 1.0
            event_dt[e, p] = 0.0 if prev_t is None else max(tn - prev_t, 0.0)
            prev_t = tn
        event_pe[e] = sinusoidal_pe(event_time[e], d_time)

        seen = {}
        for nb, rel, tn in reversed(ev):
            if nb not in seen:
                seen[nb] = (rel, tn)
            if len(seen) >= n_max:
                break
        for i, (nb, (rel, tn)) in enumerate(seen.items()):
            neigh_id[e, i] = nb
            neigh_rel[e, i] = rel
            neigh_time[e, i] = tn
            neigh_mask[e, i] = 1.0

    is_src = np.zeros((num_ent,), dtype=np.float32)
    is_src[src_ids] = 1.0
    data = AlignmentData(
        num_ent=num_ent,
        num_rel=num_rel * 2,
        seq_len=seq_len,
        n_max=n_max,
        src_ids=torch.from_numpy(src_ids),
        tgt_ids=torch.from_numpy(tgt_ids),
        is_src=torch.from_numpy(is_src),
        train_pairs=torch.from_numpy(train_pairs),
        test_pairs=torch.from_numpy(test_pairs),
        event_nb=torch.from_numpy(event_nb),
        event_rel=torch.from_numpy(event_rel),
        event_time=torch.from_numpy(event_time),
        event_dt=torch.from_numpy(event_dt),
        event_mask=torch.from_numpy(event_mask),
        event_pe=torch.from_numpy(event_pe),
        neigh_id=torch.from_numpy(neigh_id),
        neigh_rel=torch.from_numpy(neigh_rel),
        neigh_time=torch.from_numpy(neigh_time),
        neigh_mask=torch.from_numpy(neigh_mask),
        triples_src=triples1,
        triples_tgt=triples2,
    )
    if cache_dir:
        torch.save(data, cache_path)
        print(f"[data] saved cache {cache_path}")
    print(
        f"[data] {name}: |E|={num_ent} |R|={num_rel} L={seq_len} K={n_max} "
        f"active={nonempty} train={len(train_pairs)} test={len(test_pairs)}"
    )
    return data
