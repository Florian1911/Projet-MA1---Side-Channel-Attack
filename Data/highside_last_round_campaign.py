#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

INV_SBOX = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
], dtype=np.uint8)
SBOX = np.zeros(256, dtype=np.uint8)
for i, v in enumerate(INV_SBOX):
    SBOX[v] = i
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
KEYS = np.arange(256, dtype=np.uint8)
RCON = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint8)


def aes128_round10_key(master_key: np.ndarray) -> np.ndarray:
    w = np.zeros((44, 4), dtype=np.uint8)
    for i in range(4):
        w[i] = master_key[4 * i: 4 * i + 4]
    for i in range(4, 44):
        temp = w[i - 1].copy()
        if i % 4 == 0:
            temp = np.roll(temp, -1)
            temp = SBOX[temp]
            temp[0] ^= RCON[(i // 4) - 1]
        w[i] = np.bitwise_xor(w[i - 4], temp)
    return w[40:44].reshape(16)


def preprocess(traces: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return traces.astype(np.float64, copy=False)
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def byte_scores_last_round(tc: np.ndarray, tstd: np.ndarray, cbyte: np.ndarray) -> np.ndarray:
    h = HW[INV_SBOX[np.bitwise_xor(cbyte[:, None], KEYS[None, :])]]
    hc = h - h.mean(axis=0, keepdims=True)
    hstd = hc.std(axis=0, ddof=1) + 1e-15
    corr = (hc.T @ tc) / ((tc.shape[0] - 1) * hstd[:, None] * tstd[None, :])
    return np.max(np.abs(corr), axis=1)


def eval_combo(traces: np.ndarray, cts: np.ndarray, k10_true: np.ndarray) -> dict:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = tc.std(axis=0, ddof=1) + 1e-15
    recovered = []
    ranks = []
    for b in range(16):
        scores = byte_scores_last_round(tc, tstd, cts[:, b])
        order = np.argsort(scores)[::-1]
        recovered.append(int(order[0]))
        ranks.append(int(np.where(order == int(k10_true[b]))[0][0]) + 1)
    return {
        "rank1_count": int(sum(r == 1 for r in ranks)),
        "rank5_count": int(sum(r <= 5 for r in ranks)),
        "mean_rank": float(np.mean(ranks)),
        "blind_match": int(sum(recovered[i] == int(k10_true[i]) for i in range(16))),
        "recovered_k10_hex": "".join(f"{x:02X}" for x in recovered),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="High-side CPA campaign on AES last round")
    ap.add_argument("--npz", default="dataset_aes_sca_highside.npz")
    ap.add_argument("--out", default="highside_last_round_campaign_summary.json")
    ap.add_argument("--n-traces", type=int, default=1000)
    args = ap.parse_args()

    d = np.load(args.npz)
    required = {"traces", "traces_a", "traces_b", "ciphertexts", "key"}
    missing = required.difference(set(d.files))
    if missing:
        raise ValueError(f"Dataset missing required fields for last-round attack: {sorted(missing)}")

    n = min(args.n_traces, d["traces"].shape[0])
    cts = d["ciphertexts"][:n, :16].astype(np.uint8)
    master_key = d["key"][:16].astype(np.uint8)
    round10 = aes128_round10_key(master_key)

    src_map = {
        "traces": d["traces"][:n].astype(np.float32),
        "traces_a": d["traces_a"][:n].astype(np.float32),
        "traces_b": d["traces_b"][:n].astype(np.float32),
    }

    rows = []
    for src_name, src in src_map.items():
        for mode in ["none", "center_detrend"]:
            tr = preprocess(src, mode)
            res = eval_combo(tr, cts, round10)
            row = {"source": src_name, "preproc": mode, "n_traces": n, **res}
            rows.append(row)
            print(
                f"[{src_name:8s}|{mode:13s}] r1={res['rank1_count']:2d}/16 "
                f"r5={res['rank5_count']:2d}/16 mean_rank={res['mean_rank']:7.2f} "
                f"blind={res['blind_match']:2d}/16"
            )

    best = max(rows, key=lambda r: (r["rank1_count"], r["rank5_count"], -r["mean_rank"]))
    out = {
        "npz": args.npz,
        "model": "last_round_hw_invsbox = HW(InvSBox(C xor K10))",
        "master_key_hex": "".join(f"{x:02X}" for x in master_key),
        "true_round10_key_hex": "".join(f"{x:02X}" for x in round10),
        "results": rows,
        "best": best,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("best:", best)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
