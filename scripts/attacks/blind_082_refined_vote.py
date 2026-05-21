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


def preprocess(traces: np.ndarray) -> np.ndarray:
    return traces.astype(np.float64, copy=False)


def tc_tstd(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    return tc, tstd


def hyp_matrix(pbyte: np.ndarray, model: str) -> np.ndarray:
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    y = SBOX[x]
    if model == "hd_pt":
        h = HW[np.bitwise_xor(y, pbyte[:, None])]
    elif model == "hd_inout":
        h = HW[np.bitwise_xor(y, x)]
    else:
        raise ValueError(model)
    return h.astype(np.float64, copy=False)


def scores_and_poi(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, model: str, w0: int, w1: int) -> tuple[np.ndarray, np.ndarray]:
    n = tc.shape[0]
    h = hyp_matrix(pbyte, model)
    hc = h - h.mean(axis=0, keepdims=True)
    hs = h.std(axis=0, ddof=1) + 1e-15
    c = (hc.T @ tc[:, w0:w1]) / ((n - 1) * hs[:, None] * tstd[w0:w1][None, :])
    a = np.abs(c)
    poi = np.argmax(a, axis=1) + w0
    sc = a[np.arange(256), poi - w0]
    return sc, poi


def weighted_vote_byte(keys: list[int], weights: list[float]) -> int:
    d = defaultdict(float)
    for k, w in zip(keys, weights):
        d[int(k)] += float(max(0.0, w))
    return max(d.items(), key=lambda kv: kv[1])[0]


def run_combo(parts_data, model: str, base_window: tuple[int, int], refine_radius: int) -> dict:
    w0, w1 = base_window
    per_part = []
    for part_name, tr, pt in parts_data:
        tc, tstd = tc_tstd(preprocess(tr))
        part_rows = []
        for b in range(16):
            s, p = scores_and_poi(tc, tstd, pt[:, b], model, w0, w1)
            o = np.argsort(s)[::-1]
            part_rows.append(
                {
                    "byte": b,
                    "top1": int(o[0]),
                    "top2": int(o[1]),
                    "margin": float(s[o[0]] - s[o[1]]),
                    "poi_top1": int(p[o[0]]),
                }
            )
        per_part.append({"part": part_name, "bytes": part_rows})

    # First vote
    voted1 = []
    for b in range(16):
        ks = [pp["bytes"][b]["top1"] for pp in per_part]
        ws = [pp["bytes"][b]["margin"] for pp in per_part]
        voted1.append(weighted_vote_byte(ks, ws))

    # Refine per-byte windows around POIs supporting voted key
    byte_windows = []
    for b in range(16):
        pois = []
        for pp in per_part:
            row = pp["bytes"][b]
            if int(row["top1"]) == int(voted1[b]):
                pois.append(int(row["poi_top1"]))
        if not pois:
            pois = [int(pp["bytes"][b]["poi_top1"]) for pp in per_part]
        c = int(np.median(pois))
        bw0 = max(w0, c - refine_radius)
        bw1 = min(w1, c + refine_radius + 1)
        if bw1 <= bw0:
            bw0, bw1 = w0, w1
        byte_windows.append((bw0, bw1))

    # Second pass on refined windows
    per_part2 = []
    for part_name, tr, pt in parts_data:
        tc, tstd = tc_tstd(preprocess(tr))
        part_rows = []
        for b in range(16):
            bw0, bw1 = byte_windows[b]
            s, p = scores_and_poi(tc, tstd, pt[:, b], model, bw0, bw1)
            o = np.argsort(s)[::-1]
            part_rows.append(
                {
                    "byte": b,
                    "top1": int(o[0]),
                    "top2": int(o[1]),
                    "margin": float(s[o[0]] - s[o[1]]),
                    "poi_top1": int(p[o[0]]),
                }
            )
        per_part2.append({"part": part_name, "bytes": part_rows})

    voted2 = []
    unanimous = 0
    consensus_sum = 0
    for b in range(16):
        ks = [pp["bytes"][b]["top1"] for pp in per_part2]
        ws = [pp["bytes"][b]["margin"] for pp in per_part2]
        voted2.append(weighted_vote_byte(ks, ws))
        counts = defaultdict(int)
        for k in ks:
            counts[int(k)] += 1
        m = max(counts.values())
        consensus_sum += m
        if m == len(parts_data):
            unanimous += 1

    mean_margin = float(np.mean([pp["bytes"][b]["margin"] for pp in per_part2 for b in range(16)]))
    return {
        "model": model,
        "base_window": [int(w0), int(w1)],
        "refine_radius": int(refine_radius),
        "byte_windows": [[int(a), int(b)] for (a, b) in byte_windows],
        "voted_key_hex": "".join(f"{k:02X}" for k in voted2),
        "consensus_sum": int(consensus_sum),
        "unanimous_bytes": int(unanimous),
        "mean_margin": mean_margin,
        "parts_pass2": per_part2,
    }


def run_fusion(parts_data, base_window, refine_radius):
    w0, w1 = base_window
    models = ["hd_pt", "hd_inout"]
    per_part = []
    for part_name, tr, pt in parts_data:
        tc, tstd = tc_tstd(preprocess(tr))
        part_rows = []
        for b in range(16):
            s_all = []
            p_all = []
            for m in models:
                s, p = scores_and_poi(tc, tstd, pt[:, b], m, w0, w1)
                z = (s - s.mean()) / (s.std() + 1e-12)
                s_all.append(z)
                p_all.append(p)
            sf = s_all[0] + s_all[1]
            o = np.argsort(sf)[::-1]
            k1 = int(o[0])
            # pick poi from model with larger z-score at k1
            if s_all[0][k1] >= s_all[1][k1]:
                poi1 = int(p_all[0][k1])
            else:
                poi1 = int(p_all[1][k1])
            part_rows.append(
                {
                    "byte": b,
                    "top1": k1,
                    "top2": int(o[1]),
                    "margin": float(sf[o[0]] - sf[o[1]]),
                    "poi_top1": poi1,
                }
            )
        per_part.append({"part": part_name, "bytes": part_rows})

    voted = []
    unanimous = 0
    consensus_sum = 0
    for b in range(16):
        ks = [pp["bytes"][b]["top1"] for pp in per_part]
        ws = [pp["bytes"][b]["margin"] for pp in per_part]
        voted.append(weighted_vote_byte(ks, ws))
        counts = defaultdict(int)
        for k in ks:
            counts[int(k)] += 1
        m = max(counts.values())
        consensus_sum += m
        if m == len(parts_data):
            unanimous += 1
    mean_margin = float(np.mean([pp["bytes"][b]["margin"] for pp in per_part for b in range(16)]))
    return {
        "model": "fusion_hdpt_hdinout",
        "base_window": [int(w0), int(w1)],
        "refine_radius": int(refine_radius),
        "voted_key_hex": "".join(f"{k:02X}" for k in voted),
        "consensus_sum": int(consensus_sum),
        "unanimous_bytes": int(unanimous),
        "mean_margin": mean_margin,
        "parts": per_part,
    }


def main():
    ap = argparse.ArgumentParser(description="Refined blind vote with per-byte POI recentering.")
    ap.add_argument("--parts", default="blind_082_50k_part0.npz,blind_082_50k_part1.npz,blind_082_50k_part2.npz,blind_082_50k_part3.npz,blind_082_50k_part4.npz")
    ap.add_argument("--n-traces", type=int, default=5000)
    ap.add_argument("--windows", default="1200:2600,1400:3000,800:1800")
    ap.add_argument("--refine-radius", type=int, default=120)
    ap.add_argument("--out", default="blind_082_refined_vote_summary.json")
    args = ap.parse_args()

    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    windows = []
    for t in args.windows.split(","):
        a, b = t.strip().split(":")
        windows.append((int(a), int(b)))

    parts_data = []
    for p in parts:
        d = np.load(p)
        parts_data.append((p, d["traces"][: args.n_traces].astype(np.float32), d["plaintexts"][: args.n_traces, :16].astype(np.uint8)))

    rows = []
    for w in windows:
        for m in ["hd_pt", "hd_inout"]:
            r = run_combo(parts_data, m, w, args.refine_radius)
            rows.append(r)
            print(
                f"[{m:8s} win={w}] consensus={r['consensus_sum']:2d}/80 "
                f"unanimous={r['unanimous_bytes']:2d}/16 margin={r['mean_margin']:.6f}"
            )
        rf = run_fusion(parts_data, w, args.refine_radius)
        rows.append(rf)
        print(
            f"[fusion   win={w}] consensus={rf['consensus_sum']:2d}/80 "
            f"unanimous={rf['unanimous_bytes']:2d}/16 margin={rf['mean_margin']:.6f}"
        )

    best = max(rows, key=lambda r: (r["consensus_sum"], r["unanimous_bytes"], r["mean_margin"]))
    out = {
        "parts": parts,
        "n_traces_per_part": int(args.n_traces),
        "rows": rows,
        "best": best,
        "selection_rule": "maximize consensus_sum, then unanimous_bytes, then mean_margin",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nBest:", {"model": best["model"], "window": best["base_window"], "consensus_sum": best["consensus_sum"], "unanimous_bytes": best["unanimous_bytes"], "voted_key_hex": best["voted_key_hex"]})
    print("saved:", args.out)


if __name__ == "__main__":
    main()

