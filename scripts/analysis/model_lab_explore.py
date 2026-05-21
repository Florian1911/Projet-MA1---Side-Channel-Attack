#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
KEYS = np.arange(256, dtype=np.uint8)


def preprocess(traces: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return traces.astype(np.float64, copy=False)
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    return t - np.outer((t @ x) / np.dot(x, x), x)


def build_windows(tlen: int) -> list[tuple[int, int]]:
    wins = {(0, tlen)}
    for w, st in [(160, 120), (240, 120), (320, 140), (480, 160), (640, 160), (800, 200)]:
        if w >= tlen:
            continue
        for s in range(0, tlen - w + 1, st):
            wins.add((s, s + w))
        wins.add((tlen - w, tlen))
    return sorted(wins)


def tc_tstd(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    return tc, tstd


def hyp_matrix(pbyte: np.ndarray, model: str) -> np.ndarray:
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    y = SBOX[x]
    if model == "hw_sbox":
        h = HW[y]
    elif model == "hd_pt":
        h = HW[np.bitwise_xor(y, pbyte[:, None])]
    elif model == "hd_inout":
        h = HW[np.bitwise_xor(y, x)]
    elif model == "hw_in":
        h = HW[x]
    else:
        raise ValueError(model)
    return h.astype(np.float64, copy=False)


def corr_scores(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, model: str) -> np.ndarray:
    n = tc.shape[0]
    h = hyp_matrix(pbyte, model)
    hc = h - h.mean(axis=0, keepdims=True)
    hstd = h.std(axis=0, ddof=1) + 1e-15
    c = (hc.T @ tc) / ((n - 1) * hstd[:, None] * tstd[None, :])
    return np.abs(c)


def rank_from_scores(scores: np.ndarray, tk: int) -> int:
    order = np.argsort(scores)[::-1]
    return int(np.where(order == int(tk))[0][0]) + 1


def evaluate_strategy(
    corr_cal: dict[str, np.ndarray],
    corr_eval: dict[str, np.ndarray],
    key_byte: int,
    windows: list[tuple[int, int]],
    strategy: str,
) -> int:
    if strategy == "fixed_hw":
        m = "hw_sbox"
        s = np.max(corr_eval[m][:, :], axis=1)
        return rank_from_scores(s, key_byte)
    if strategy == "fixed_hdpt":
        m = "hd_pt"
        s = np.max(corr_eval[m][:, :], axis=1)
        return rank_from_scores(s, key_byte)

    if strategy == "byte_best_model_window":
        best = (-1.0, None, None)
        for m, cc in corr_cal.items():
            for w0, w1 in windows:
                v = float(np.max(cc[key_byte, w0:w1]))
                if v > best[0]:
                    best = (v, m, (w0, w1))
        _, m, (w0, w1) = best
        s = np.max(corr_eval[m][:, w0:w1], axis=1)
        return rank_from_scores(s, key_byte)

    if strategy == "byte_top2_model_fusion":
        cand = []
        for m, cc in corr_cal.items():
            best_v = -1.0
            best_w = (0, cc.shape[1])
            for w0, w1 in windows:
                v = float(np.max(cc[key_byte, w0:w1]))
                if v > best_v:
                    best_v = v
                    best_w = (w0, w1)
            cand.append((best_v, m, best_w))
        cand.sort(reverse=True, key=lambda x: x[0])
        (_, m1, (a0, a1)), (_, m2, (b0, b1)) = cand[:2]
        s1 = np.max(corr_eval[m1][:, a0:a1], axis=1)
        s2 = np.max(corr_eval[m2][:, b0:b1], axis=1)
        z1 = (s1 - s1.mean()) / (s1.std() + 1e-12)
        z2 = (s2 - s2.mean()) / (s2.std() + 1e-12)
        s = z1 + z2
        return rank_from_scores(s, key_byte)

    raise ValueError(strategy)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory model lab with innovative strategies.")
    ap.add_argument("--npz", default="dataset_aes_sca_highside.npz")
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--calib-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="model_lab_explore_summary.json")
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d["traces"].astype(np.float32)
    ta = d["traces_a"].astype(np.float32) if "traces_a" in d.files else None
    tb = d["traces_b"].astype(np.float32) if "traces_b" in d.files else None
    pt = d["plaintexts"][:, :16].astype(np.uint8)
    key = d["key"][:16].astype(np.uint8)

    sources = {"traces": traces}
    if ta is not None and tb is not None:
        sources["traces_a"] = ta
        sources["traces_b"] = tb
        sources["diff_ab"] = ta - tb

    models = ["hw_sbox", "hd_pt", "hd_inout", "hw_in"]
    preprocs = ["none", "center_detrend"]
    strategies = ["fixed_hw", "fixed_hdpt", "byte_best_model_window", "byte_top2_model_fusion"]
    wins = build_windows(traces.shape[1])

    rng = np.random.default_rng(args.seed)
    n = traces.shape[0]
    n_cal = int(max(128, min(n - 128, round(args.calib_ratio * n))))
    rows = []
    print(f"n={n}, calib={n_cal}, eval={n-n_cal}, windows={len(wins)}")

    for src_name, src in sources.items():
        for prep in preprocs:
            x = preprocess(src, prep)
            split_ranks = {s: [] for s in strategies}
            for _ in range(args.splits):
                perm = rng.permutation(n)
                ic = perm[:n_cal]
                ie = perm[n_cal:]
                tc_c, ts_c = tc_tstd(x[ic])
                tc_e, ts_e = tc_tstd(x[ie])

                for b in range(16):
                    corr_c = {m: corr_scores(tc_c, ts_c, pt[ic, b], m) for m in models}
                    corr_e = {m: corr_scores(tc_e, ts_e, pt[ie, b], m) for m in models}
                    tk = int(key[b])
                    for s in strategies:
                        r = evaluate_strategy(corr_c, corr_e, tk, wins, s)
                        split_ranks[s].append(r)

            for s in strategies:
                rr = split_ranks[s]
                row = {
                    "source": src_name,
                    "preproc": prep,
                    "strategy": s,
                    "rank1_mean": float(np.mean([r == 1 for r in rr]) * 16.0),
                    "rank5_mean": float(np.mean([r <= 5 for r in rr]) * 16.0),
                    "mean_rank": float(np.mean(rr)),
                }
                rows.append(row)
                print(
                    f"[{src_name:8s} {prep:13s} {s:24s}] "
                    f"r1={row['rank1_mean']:.2f}/16 r5={row['rank5_mean']:.2f}/16 mean={row['mean_rank']:.2f}"
                )

    best = max(rows, key=lambda r: (r["rank1_mean"], r["rank5_mean"], -r["mean_rank"]))
    out = {
        "npz": args.npz,
        "splits": int(args.splits),
        "calib_ratio": float(args.calib_ratio),
        "rows": rows,
        "best": best,
        "notes": "rank means are averaged over bytes and splits (scaled to /16).",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nBest:", best)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

