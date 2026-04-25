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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


def preprocess(tr: np.ndarray) -> np.ndarray:
    t = tr.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def corr_vec(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, key: int, model: str) -> np.ndarray:
    x = np.bitwise_xor(pbyte, np.uint8(key))
    s = SBOX[x]
    if model == "hw_sbox":
        h = HW[s]
    elif model == "hd_ps":
        h = HW[np.bitwise_xor(s, pbyte)]
    elif model == "mix50":
        h = 0.5 * HW[s] + 0.5 * HW[np.bitwise_xor(s, pbyte)]
    else:
        raise ValueError(f"Unsupported model: {model}")
    hc = h - h.mean()
    hstd = h.std(ddof=1) + 1e-15
    return (hc @ tc) / ((tc.shape[0] - 1) * hstd * tstd)


def parse_true_key_hex(s: str) -> np.ndarray:
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "").upper()
    if len(t) != 32:
        raise ValueError("--true-key-hex must contain 32 hex chars")
    return np.array([int(t[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast full-blind CPA on full dataset (coarse+refine)")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-traces", type=int, default=20000)
    ap.add_argument("--decim", type=int, default=4)
    ap.add_argument("--refine-radius", type=int, default=24)
    ap.add_argument("--model", choices=["hw_sbox", "hd_ps", "mix50"], default="hw_sbox")
    ap.add_argument("--out", default="full_blind_direct_fast_summary.json")
    ap.add_argument("--true-key-hex", default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    tr = preprocess(d["traces"][:args.n_traces])
    pt = d["plaintexts"][:args.n_traces, :16].astype(np.uint8)

    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    coarse_idx = np.arange(0, tr.shape[1], args.decim, dtype=np.int32)
    tc_coarse = tc[:, coarse_idx]
    tstd_coarse = tstd[coarse_idx]

    recovered = []
    rows = []

    for b in range(16):
        # Coarse scan
        coarse_scores = np.zeros(256, dtype=np.float64)
        coarse_pos = np.zeros(256, dtype=np.int32)
        pbyte = pt[:, b]
        for k in range(256):
            c = corr_vec(tc_coarse, tstd_coarse, pbyte, k, args.model)
            ac = np.abs(c)
            i = int(np.argmax(ac))
            coarse_scores[k] = float(ac[i])
            coarse_pos[k] = int(coarse_idx[i])

        k_coarse = int(np.argmax(coarse_scores))
        c0 = int(coarse_pos[k_coarse])
        lo = max(0, c0 - args.refine_radius)
        hi = min(tc.shape[1] - 1, c0 + args.refine_radius)
        ref_idx = np.arange(lo, hi + 1, dtype=np.int32)
        tc_ref = tc[:, ref_idx]
        tstd_ref = tstd[ref_idx]

        # Refine in local window
        ref_scores = np.zeros(256, dtype=np.float64)
        ref_pos = np.zeros(256, dtype=np.int32)
        for k in range(256):
            c = corr_vec(tc_ref, tstd_ref, pbyte, k, args.model)
            ac = np.abs(c)
            i = int(np.argmax(ac))
            ref_scores[k] = float(ac[i])
            ref_pos[k] = int(ref_idx[i])

        order = np.argsort(ref_scores)[::-1]
        k1 = int(order[0])
        k2 = int(order[1])
        margin = float(ref_scores[k1] - ref_scores[k2])
        poi = int(ref_pos[k1])

        recovered.append(k1)
        rows.append(
            {
                "byte": b,
                "top_key": k1,
                "top_key_hex": f"{k1:02X}",
                "score": float(ref_scores[k1]),
                "margin_top1_top2": margin,
                "poi": poi,
                "coarse_center": c0,
                "refine_window": [int(lo), int(hi)],
                "top5": [{"key": int(k), "score": float(ref_scores[k])} for k in order[:5]],
            }
        )
        print(
            f"byte{b:02d}: top=0x{k1:02X} score={ref_scores[k1]:.6f} "
            f"margin={margin:.6f} coarse={c0} poi={poi}"
        )

    rec_hex = "".join(f"{x:02X}" for x in recovered)
    out = {
        "npz": args.npz,
        "n_traces": int(args.n_traces),
        "decim": int(args.decim),
        "refine_radius": int(args.refine_radius),
        "model": args.model,
        "recovered_key_hex": rec_hex,
        "recovered_key": [int(x) for x in recovered],
        "bytes": rows,
        "blind": True,
        "strict": True,
    }

    if args.true_key_hex:
        true_key = parse_true_key_hex(args.true_key_hex)
        match = sum(int(recovered[i] == int(true_key[i])) for i in range(16))
        out["true_key_hex"] = "".join(f"{int(x):02X}" for x in true_key)
        out["byte_match_count"] = int(match)
        out["byte_match_vector"] = [int(recovered[i] == int(true_key[i])) for i in range(16)]
        print(f"Validation (end-only): {match}/16 bytes match")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print("Recovered key (blind strict):", rec_hex)
    print("saved:", str(out_path))


if __name__ == "__main__":
    main()
