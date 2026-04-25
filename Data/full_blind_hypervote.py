#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
KEYS = np.arange(256, dtype=np.uint8)


@dataclass(frozen=True)
class Config:
    model: str
    w0: int
    w1: int


def preprocess(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def tc_stats(tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    return tc, tstd


def model_hyp(pbyte: np.ndarray, model: str) -> np.ndarray:
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    s = SBOX[x]
    if model == "hw_sbox":
        h = HW[s]
    elif model == "hd_ps":
        h = HW[np.bitwise_xor(s, pbyte[:, None])]
    elif model == "hw_in":
        h = HW[x]
    elif model == "hd_0sbox":
        h = HW[s]
    elif model.startswith("mix_"):
        a = float(model.split("_")[1])
        h = a * HW[s] + (1.0 - a) * HW[np.bitwise_xor(s, pbyte[:, None])]
    elif model.startswith("bit"):
        b = int(model[3:])
        h = ((s >> b) & 1).astype(np.float64)
    else:
        raise ValueError(model)
    return h.T.astype(np.float64, copy=False)


def cpa_scores(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray]:
    h = model_hyp(pbyte, model)
    hc = h - h.mean(axis=1, keepdims=True)
    hs = h.std(axis=1, ddof=1, keepdims=True) + 1e-15
    corr = (hc @ tc) / ((tc.shape[0] - 1) * hs * tstd[None, :])
    ac = np.abs(corr)
    s1 = ac.max(axis=1)
    poi = ac.argmax(axis=1)
    return s1, poi


def rank_with_margin(scores: np.ndarray) -> Tuple[int, int, float]:
    o = np.argsort(scores)[::-1]
    k1, k2 = int(o[0]), int(o[1])
    return k1, k2, float(scores[k1] - scores[k2])


def evaluate_config(traces: np.ndarray, plains: np.ndarray, cfg: Config, byte_idx: int) -> Tuple[int, int, float, int]:
    tw = traces[:, cfg.w0:cfg.w1]
    tc, tstd = tc_stats(tw)
    scores, poi = cpa_scores(tc, tstd, plains[:, byte_idx], cfg.model)
    k1, k2, m = rank_with_margin(scores)
    return k1, k2, m, int(poi[k1] + cfg.w0)


def split_indices(n: int, seed: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        h = n // 2
        out.append((p[:h], p[h:]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-blind hyper-vote CPA")
    ap.add_argument("--calib", default="unknown_B_calib5k_nokey.npz")
    ap.add_argument("--attacks", nargs="+", default=["unknown_B_attack15k_nokey.npz", "unknown_B_attack18k_nokey.npz", "unknown_B_20k_aligned_nokey.npz", "unknown_B_20k_aligned_nokey_shiftp8.npz"])
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--topk-configs", type=int, default=4)
    ap.add_argument("--out", default="full_blind_hypervote_summary.json")
    args = ap.parse_args()

    d = np.load(args.calib)
    tr_c = preprocess(d["traces"])
    pt_c = d["plaintexts"].astype(np.uint8)

    # Windows guided by previous experiments + broad coverage.
    windows = [
        (0, 1200), (200, 1600), (400, 2000), (600, 2200),
        (800, 2400), (1000, 2600), (1200, 3000), (1400, 3200), (0, 3968)
    ]
    models = [
        "hw_sbox", "hd_ps", "hw_in", "hd_0sbox",
        "mix_0.25", "mix_0.50", "mix_0.75",
        "bit0", "bit1", "bit2", "bit3", "bit4", "bit5", "bit6", "bit7",
    ]
    cfgs = [Config(m, w0, w1) for m in models for (w0, w1) in windows]

    splits = split_indices(tr_c.shape[0], args.seed, args.n_splits)

    # 1) Blind model selection on calibration: stability + margin.
    selected: Dict[int, List[Tuple[Config, float]]] = {}
    calib_rows = []

    for b in range(16):
        cfg_scores = []
        for cfg in cfgs:
            votes = []
            margins = []
            for ia, ib in splits:
                k1a, _, ma, _ = evaluate_config(tr_c[ia], pt_c[ia], cfg, b)
                k1b, _, mb, _ = evaluate_config(tr_c[ib], pt_c[ib], cfg, b)
                votes.extend([k1a, k1b])
                margins.extend([ma, mb])
            vc = np.bincount(np.array(votes, dtype=np.int32), minlength=256)
            top = int(vc.argmax())
            stability = float(vc[top]) / float(len(votes))
            med_margin = float(np.median(margins))
            score = stability * max(0.0, med_margin)
            cfg_scores.append((cfg, score, top, stability, med_margin))

        cfg_scores.sort(key=lambda x: x[1], reverse=True)
        best = cfg_scores[: args.topk_configs]
        selected[b] = [(c, s) for c, s, _, _, _ in best]
        calib_rows.append({
            "byte": b,
            "top_configs": [
                {
                    "model": c.model,
                    "window": [c.w0, c.w1],
                    "selection_score": s,
                    "calib_key_vote": k,
                    "stability": st,
                    "median_margin": mm,
                }
                for c, s, k, st, mm in best
            ],
        })

    # 2) Attack datasets: weighted voting using selected configs.
    attack_data = []
    for p in args.attacks:
        dd = np.load(p)
        attack_data.append((p, preprocess(dd["traces"]), dd["plaintexts"].astype(np.uint8)))

    byte_rows = []
    recovered = []

    for b in range(16):
        wsum = np.zeros(256, dtype=np.float64)
        evidence = []

        for p, tr_a, pt_a in attack_data:
            for cfg, wcfg in selected[b]:
                k1, k2, m, poi = evaluate_config(tr_a, pt_a, cfg, b)
                w = max(0.0, m) * max(1e-6, wcfg)
                wsum[k1] += w
                evidence.append({
                    "dataset": p,
                    "model": cfg.model,
                    "window": [cfg.w0, cfg.w1],
                    "k1": k1,
                    "k2": k2,
                    "margin": m,
                    "poi": poi,
                    "weight_added": w,
                })

        ordk = np.argsort(wsum)[::-1]
        kf, k2 = int(ordk[0]), int(ordk[1])
        conf = float((wsum[kf] - wsum[k2]) / (wsum[kf] + 1e-12))
        vote_share = float(wsum[kf] / (wsum.sum() + 1e-12))

        recovered.append(kf)
        byte_rows.append({
            "byte": b,
            "recovered_key": kf,
            "recovered_key_hex": f"{kf:02X}",
            "runner_up": k2,
            "runner_up_hex": f"{k2:02X}",
            "confidence_margin_ratio": conf,
            "vote_share": vote_share,
            "top3": [
                {"key": int(k), "key_hex": f"{int(k):02X}", "score": float(wsum[k])}
                for k in ordk[:3]
            ],
            "evidence_count": len(evidence),
            "evidence": evidence,
        })

    key_hex = "".join(f"{k:02X}" for k in recovered)
    out = {
        "calib": args.calib,
        "attacks": args.attacks,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "topk_configs": args.topk_configs,
        "recovered_key": recovered,
        "recovered_key_hex": key_hex,
        "bytes": byte_rows,
        "calibration_selection": calib_rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Recovered key (full blind):", key_hex)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
