#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
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
    else:
        raise ValueError(model)
    return h.astype(np.float64, copy=False)


def recover_on_window(tc: np.ndarray, tstd: np.ndarray, pt: np.ndarray, model: str, w0: int, w1: int) -> tuple[str, list[float]]:
    keys = []
    margins = []
    tcw = tc[:, w0:w1]
    tsw = tstd[w0:w1]
    n = tc.shape[0]
    for b in range(16):
        h = hyp_matrix(pt[:, b], model)
        hc = h - h.mean(axis=0, keepdims=True)
        hs = h.std(axis=0, ddof=1) + 1e-15
        c = (hc.T @ tcw) / ((n - 1) * hs[:, None] * tsw[None, :])
        s = np.max(np.abs(c), axis=1)
        order = np.argsort(s)[::-1]
        keys.append(int(order[0]))
        margins.append(float(s[order[0]] - s[order[1]]))
    return "".join(f"{k:02X}" for k in keys), margins


def consensus_score(keys_hex: list[str]) -> tuple[int, int]:
    consensus_sum = 0
    unanimous = 0
    for b in range(16):
        votes = [k[2 * b : 2 * b + 2] for k in keys_hex]
        counts = defaultdict(int)
        for v in votes:
            counts[v] += 1
        best = max(counts.values())
        consensus_sum += best
        if best == len(keys_hex):
            unanimous += 1
    return consensus_sum, unanimous


def weighted_vote(keys_hex: list[str], margins_list: list[list[float]]) -> str:
    out = []
    for b in range(16):
        w = defaultdict(float)
        for ki, key_hex in enumerate(keys_hex):
            kb = key_hex[2 * b : 2 * b + 2]
            w[kb] += float(max(0.0, margins_list[ki][b]))
        out.append(max(w.items(), key=lambda kv: kv[1])[0])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch blind_082 part vote across models/windows.")
    ap.add_argument("--parts", default="blind_082_50k_part0.npz,blind_082_50k_part1.npz,blind_082_50k_part2.npz,blind_082_50k_part3.npz,blind_082_50k_part4.npz")
    ap.add_argument("--n-traces", type=int, default=5000)
    ap.add_argument("--models", default="hw_sbox,hd_pt,hd_inout")
    ap.add_argument("--preproc", default="none,center_detrend")
    ap.add_argument("--windows", default="0:3968,800:1800,1200:2600,1400:3000")
    ap.add_argument("--out", default="blind_082_batch_vote_summary.json")
    args = ap.parse_args()

    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    preprocs = [x.strip() for x in args.preproc.split(",") if x.strip()]
    windows = []
    for tok in args.windows.split(","):
        a, b = tok.strip().split(":")
        windows.append((int(a), int(b)))

    cached = {}
    for p in parts:
        d = np.load(p)
        traces = d["traces"][: args.n_traces].astype(np.float32)
        plains = d["plaintexts"][: args.n_traces, :16].astype(np.uint8)
        cached[p] = {"traces": traces, "plains": plains}

    rows = []
    for prep in preprocs:
        for model in models:
            for w0, w1 in windows:
                keys_by_part = []
                margins_by_part = []
                part_rows = []
                for p in parts:
                    tr = preprocess(cached[p]["traces"], prep)
                    tc, tstd = tc_tstd(tr)
                    key_hex, margins = recover_on_window(tc, tstd, cached[p]["plains"], model, w0, w1)
                    keys_by_part.append(key_hex)
                    margins_by_part.append(margins)
                    part_rows.append(
                        {
                            "part": p,
                            "key_hex": key_hex,
                            "avg_margin12": float(np.mean(margins)),
                        }
                    )
                consensus_sum, unanimous = consensus_score(keys_by_part)
                voted = weighted_vote(keys_by_part, margins_by_part)
                row = {
                    "model": model,
                    "preproc": prep,
                    "window": [int(w0), int(w1)],
                    "window_len": int(w1 - w0),
                    "consensus_sum": int(consensus_sum),
                    "unanimous_bytes": int(unanimous),
                    "mean_part_margin": float(np.mean([r["avg_margin12"] for r in part_rows])),
                    "voted_key_hex": voted,
                    "parts": part_rows,
                }
                rows.append(row)
                print(
                    f"[{model:8s} {prep:13s} win=[{w0:4d},{w1:4d})] "
                    f"consensus={consensus_sum:2d}/80 unanimous={unanimous:2d}/16 margin={row['mean_part_margin']:.6f}"
                )

    best = max(rows, key=lambda r: (r["consensus_sum"], r["unanimous_bytes"], r["mean_part_margin"]))
    best_by_model = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        best_by_model[m] = max(rs, key=lambda r: (r["consensus_sum"], r["unanimous_bytes"], r["mean_part_margin"]))

    out = {
        "parts": parts,
        "n_traces_per_part": int(args.n_traces),
        "rows": rows,
        "best_overall": best,
        "best_by_model": best_by_model,
        "selection_rule": "maximize consensus_sum, then unanimous_bytes, then mean_part_margin",
        "note": "Blind evaluation: no true key used.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nBest overall:", best)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

