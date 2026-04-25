#!/usr/bin/env python3
"""Pipeline CPA 16 bytes: alignement optionnel + scan fenetre auto + attaque complete."""

import argparse
import json
import subprocess
from pathlib import Path

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
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def maybe_lowpass(traces: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return traces
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.apply_along_axis(lambda t: np.convolve(t, kernel, mode="same"), axis=1, arr=traces)


def cpa_scores_for_byte(traces_f: np.ndarray, pt_col: np.ndarray) -> np.ndarray:
    n = traces_f.shape[0]
    t_mean = traces_f.mean(axis=0)
    t_std = traces_f.std(axis=0)
    t_std[t_std < 1e-10] = 1e-10
    tc = traces_f - t_mean

    scores = np.zeros(256, dtype=np.float32)
    for k in range(256):
        h = HW[SBOX[np.bitwise_xor(pt_col, k)]]
        hc = h - h.mean()
        h_std = h.std()
        if h_std < 1e-10:
            continue
        cov = (hc @ tc) / n
        corr = np.abs(cov / (h_std * t_std))
        scores[k] = float(corr.max())
    return scores


def rank_of_true(scores: np.ndarray, true_k: int) -> int:
    order = np.argsort(scores)[::-1]
    return int(np.where(order == true_k)[0][0]) + 1


def auto_align(dataset: Path, out_aligned: Path, n_traces: int) -> Path:
    cmd = [
        "python3",
        "align_local_per_trace.py",
        "--in",
        str(dataset),
        "--out",
        str(out_aligned),
        "--center",
        "520",
        "--window",
        "260",
        "--max-shift",
        "80",
        "--iters",
        "2",
        "--ref",
        "median",
        "--mode",
        "ref",
    ]
    print("[ALIGN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="CPA 16 bytes auto")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-traces", type=int, default=20000)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--aligned-out", default="dataset_aligned_tmp.npz")
    ap.add_argument("--scan-byte", type=int, default=0)
    ap.add_argument("--start-min", type=int, default=200)
    ap.add_argument("--start-max", type=int, default=1200)
    ap.add_argument("--start-step", type=int, default=40)
    ap.add_argument("--lengths", default="80,120,160,220")
    ap.add_argument("--lp-window", type=int, default=1)
    ap.add_argument("--summary-json", default="cpa16_auto_summary.json")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if args.align:
        dataset_path = auto_align(dataset_path, Path(args.aligned_out), args.n_traces)

    d = np.load(dataset_path)
    traces = d["traces"][: args.n_traces].astype(np.float32)
    plains = d["plaintexts"][: args.n_traces].astype(np.uint8)
    true_key = d["key"].astype(np.uint8) if "key" in d.files else None

    traces = center_and_detrend(traces)
    traces = maybe_lowpass(traces, args.lp_window)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    scan_rows = []
    b = int(args.scan_byte)
    pt_col = plains[:, b]

    print(f"[SCAN] byte={b} windows ...")
    for start in range(args.start_min, args.start_max + 1, args.start_step):
        for L in lengths:
            end = start + L
            if end > traces.shape[1]:
                continue
            tw = traces[:, start:end]
            scores = cpa_scores_for_byte(tw, pt_col)
            best_k = int(scores.argmax())
            best_s = float(scores[best_k])
            row = {
                "start": int(start),
                "end": int(end),
                "len": int(L),
                "best_k": best_k,
                "best_corr": best_s,
            }
            if true_key is not None:
                row["true_rank"] = rank_of_true(scores, int(true_key[b]))
                row["true_corr"] = float(scores[int(true_key[b])])
            scan_rows.append(row)

    if true_key is not None:
        scan_rows.sort(key=lambda r: (r["true_rank"], -r.get("true_corr", 0.0)))
    else:
        scan_rows.sort(key=lambda r: -r["best_corr"])

    best_win = scan_rows[0]
    ws, we = int(best_win["start"]), int(best_win["end"])
    print(f"[SCAN] best window: [{ws}:{we}] len={we-ws}")

    print("[CPA16] attaque 16 bytes...")
    rec = []
    byte_rows = []
    tw = traces[:, ws:we]
    for bi in range(16):
        scores = cpa_scores_for_byte(tw, plains[:, bi])
        bk = int(scores.argmax())
        bc = float(scores[bk])
        rec.append(bk)
        row = {"byte": bi, "best_k": bk, "best_corr": bc}
        if true_key is not None:
            tk = int(true_key[bi])
            row["true_k"] = tk
            row["true_corr"] = float(scores[tk])
            row["true_rank"] = rank_of_true(scores, tk)
            row["ok"] = bool(bk == tk)
        byte_rows.append(row)

    rec_hex = " ".join(f"{x:02x}" for x in rec)
    print(f"[CPA16] key rec: {rec_hex}")

    if true_key is not None:
        n_ok = int(sum(1 for r in byte_rows if r.get("ok", False)))
        print(f"[CPA16] bytes corrects: {n_ok}/16")

    summary = {
        "dataset": str(dataset_path),
        "n_traces": int(args.n_traces),
        "scan_byte": int(args.scan_byte),
        "selected_window": {"start": ws, "end": we, "len": we - ws},
        "scan_top10": scan_rows[:10],
        "recovered_key": rec,
        "recovered_key_hex": rec_hex,
        "bytes": byte_rows,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] summary: {args.summary_json}")


if __name__ == "__main__":
    main()
