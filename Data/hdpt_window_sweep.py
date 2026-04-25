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
    windows = {(0, tlen)}
    params = [
        (160, 120),
        (240, 120),
        (320, 140),
        (480, 160),
        (640, 160),
        (800, 200),
        (1000, 200),
        (1200, 240),
    ]
    for w, step in params:
        if w >= tlen:
            continue
        for s in range(0, tlen - w + 1, step):
            windows.add((s, s + w))
        windows.add((tlen - w, tlen))
    return sorted(windows)


def corr_all_keys(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray) -> np.ndarray:
    n = tc.shape[0]
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    y = SBOX[x]
    h = HW[np.bitwise_xor(y, pbyte[:, None])]  # HD_pt
    hc = h - h.mean(axis=0, keepdims=True)
    hstd = h.std(axis=0, ddof=1) + 1e-15
    corr = (hc.T @ tc) / ((n - 1) * hstd[:, None] * tstd[None, :])
    return np.abs(corr)


def eval_config_hdpt(
    traces: np.ndarray,
    plains: np.ndarray,
    key: np.ndarray,
    windows: list[tuple[int, int]],
) -> list[dict]:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15

    aggr = {
        (w0, w1): {
            "ranks": [],
            "tops": [],
        }
        for (w0, w1) in windows
    }

    for b in range(16):
        corr = corr_all_keys(tc, tstd, plains[:, b])  # (256, T)
        tk = int(key[b])
        for (w0, w1) in windows:
            scores = np.max(corr[:, w0:w1], axis=1)
            order = np.argsort(scores)[::-1]
            rank = int(np.where(order == tk)[0][0]) + 1
            aggr[(w0, w1)]["ranks"].append(rank)
            aggr[(w0, w1)]["tops"].append(int(order[0]))

    rows = []
    for (w0, w1), d in aggr.items():
        ranks = d["ranks"]
        tops = d["tops"]
        rows.append(
            {
                "win_start": int(w0),
                "win_end": int(w1),
                "win_len": int(w1 - w0),
                "rank1_count": int(sum(r == 1 for r in ranks)),
                "rank5_count": int(sum(r <= 5 for r in ranks)),
                "mean_rank": float(np.mean(ranks)),
                "recovered_key_hex": "".join(f"{x:02X}" for x in tops),
            }
        )
    return rows


def sources_from_dataset(npz_path: str) -> dict[str, np.ndarray]:
    d = np.load(npz_path)
    if "traces_a" in d.files and "traces_b" in d.files:
        ta = d["traces_a"].astype(np.float32)
        tb = d["traces_b"].astype(np.float32)
        return {
            "traces": d["traces"].astype(np.float32),
            "traces_a": ta,
            "traces_b": tb,
            "diff_ab": ta - tb,
        }
    return {"traces": d["traces"].astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Large HD_pt window sweep (high-side focused + low-side sanity).")
    ap.add_argument(
        "--highside",
        default="dataset_aes_sca_highside.npz,highside_traces_calib500_known.npz,highside_traces_a_calib500_known.npz,highside_traces_b_calib500_known.npz",
    )
    ap.add_argument("--lowside", default="dataset_lowside.npz,dataset_lowside_aligned.npz")
    ap.add_argument("--n-list", default="300,500,700,1000")
    ap.add_argument("--preproc", default="none,center_detrend")
    ap.add_argument("--out", default="hdpt_long_sweep_summary.json")
    args = ap.parse_args()

    highside_sets = [x.strip() for x in args.highside.split(",") if x.strip()]
    lowside_sets = [x.strip() for x in args.lowside.split(",") if x.strip()]
    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    preprocs = [x.strip() for x in args.preproc.split(",") if x.strip()]

    all_rows = []
    for group, dsets in [("highside", highside_sets), ("lowside", lowside_sets)]:
        for ds in dsets:
            d = np.load(ds)
            pt = d["plaintexts"][:, :16].astype(np.uint8)
            key = d["key"][:16].astype(np.uint8)
            src_map = sources_from_dataset(ds)
            print(f"\n== {group}: {ds} ==")
            for src_name, src in src_map.items():
                tlen = src.shape[1]
                windows = build_windows(tlen)
                print(f"  source={src_name} windows={len(windows)}")
                for prep in preprocs:
                    tr_all = preprocess(src, prep)
                    for n in n_list:
                        if n > tr_all.shape[0]:
                            continue
                        rows = eval_config_hdpt(tr_all[:n], pt[:n], key, windows)
                        best = max(rows, key=lambda r: (r["rank1_count"], r["rank5_count"], -r["mean_rank"]))
                        all_rows.append(
                            {
                                "group": group,
                                "dataset": ds,
                                "source": src_name,
                                "preproc": prep,
                                "n_traces": int(n),
                                "best_window": best,
                            }
                        )
                        print(
                            f"    [{prep:13s} n={n:4d}] "
                            f"best r1={best['rank1_count']:2d}/16 r5={best['rank5_count']:2d}/16 "
                            f"mean={best['mean_rank']:.2f} win=[{best['win_start']},{best['win_end']})"
                        )

    best_high = max(
        [r for r in all_rows if r["group"] == "highside"],
        key=lambda r: (
            r["best_window"]["rank1_count"],
            r["best_window"]["rank5_count"],
            -r["best_window"]["mean_rank"],
        ),
    )
    best_low = max(
        [r for r in all_rows if r["group"] == "lowside"],
        key=lambda r: (
            r["best_window"]["rank1_count"],
            r["best_window"]["rank5_count"],
            -r["best_window"]["mean_rank"],
        ),
    )

    out = {
        "model": "hd_pt = HW(SBOX(pt xor k) xor pt)",
        "all_configs": all_rows,
        "best_highside": best_high,
        "best_lowside": best_low,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nBest highside:", best_high)
    print("Best lowside :", best_low)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

