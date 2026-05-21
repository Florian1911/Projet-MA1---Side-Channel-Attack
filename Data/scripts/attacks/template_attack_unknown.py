#!/usr/bin/env python3
import argparse
import json
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
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def snr_curve(traces: np.ndarray, classes: np.ndarray, n_classes: int = 9) -> np.ndarray:
    means = np.zeros((n_classes, traces.shape[1]), dtype=np.float64)
    vars_ = np.zeros((n_classes, traces.shape[1]), dtype=np.float64)
    counts = np.zeros((n_classes,), dtype=np.int64)
    for c in range(n_classes):
        idx = (classes == c)
        nc = int(idx.sum())
        if nc == 0:
            continue
        counts[c] = nc
        tc = traces[idx]
        means[c] = tc.mean(axis=0)
        vars_[c] = tc.var(axis=0)
    valid = counts > 0
    return means[valid].var(axis=0) / (vars_[valid].mean(axis=0) + 1e-12)


def parse_key_hex(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    k = s.strip().replace(" ", "").replace(":", "").replace(",", "")
    if len(k) != 32:
        raise ValueError("--true-key-hex doit contenir 32 caractères hex")
    return np.array([int(k[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


def build_templates(xp: np.ndarray, classes: np.ndarray, poi_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Gaussian indépendantes par POI et classe HW.
    means = np.zeros((9, len(poi_idx)), dtype=np.float64)
    vars_ = np.ones((9, len(poi_idx)), dtype=np.float64)
    for c in range(9):
        idx = (classes == c)
        if idx.sum() < 4:
            continue
        xc = xp[idx][:, poi_idx]
        means[c] = xc.mean(axis=0)
        vars_[c] = xc.var(axis=0) + 1e-6
    return means, vars_


def score_key_likelihood(xa: np.ndarray, pbyte: np.ndarray, poi_idx: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    x = xa[:, poi_idx].astype(np.float64)
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        cls = HW[SBOX[np.bitwise_xor(pbyte, k)]]
        mu = means[cls]          # [n, p]
        va = vars_[cls]          # [n, p]
        # log-likelihood gaussienne diagonale (constantes communes retirées).
        z = (x - mu) ** 2 / va + np.log(va)
        scores[k] = -0.5 * float(np.sum(z))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Template attack unknown-key (HW Gaussian, per byte)")
    ap.add_argument("--profile-npz", required=True)
    ap.add_argument("--attack-npz", required=True)
    ap.add_argument("--out-prefix", default="template_unknown")
    ap.add_argument("--n-poi", type=int, default=30)
    ap.add_argument("--n-profile", type=int, default=0, help="0=all")
    ap.add_argument("--n-attack", type=int, default=0, help="0=all")
    ap.add_argument("--n-list", default="500,1000,2000,3000,5000,7000,10000,15000,20000")
    ap.add_argument("--true-key-hex", default=None)
    args = ap.parse_args()

    dp = np.load(args.profile_npz)
    da = np.load(args.attack_npz)
    xp = center_and_detrend(dp["traces"])
    xa = center_and_detrend(da["traces"])
    pp = dp["plaintexts"][:, :16].astype(np.uint8)
    pa = da["plaintexts"][:, :16].astype(np.uint8)
    kprof = dp["key"][:16].astype(np.uint8)
    ktrue = parse_key_hex(args.true_key_hex)

    if args.n_profile > 0:
        xp = xp[:args.n_profile]
        pp = pp[:args.n_profile]
    if args.n_attack > 0:
        xa = xa[:args.n_attack]
        pa = pa[:args.n_attack]

    n_list = sorted(set(int(x) for x in args.n_list.split(",") if x.strip()))
    n_list = [n for n in n_list if 100 <= n <= xa.shape[0]]
    if xa.shape[0] not in n_list:
        n_list.append(int(xa.shape[0]))

    recovered = []
    per_byte_ranks = {}
    for b in range(16):
        cls_prof = HW[SBOX[np.bitwise_xor(pp[:, b], int(kprof[b]))]]
        snr = snr_curve(xp, cls_prof, n_classes=9)
        poi_idx = np.argsort(snr)[::-1][: int(args.n_poi)].astype(np.int32)
        poi_idx.sort()
        means, vars_ = build_templates(xp, cls_prof, poi_idx)

        # key from all attack traces
        scores_all = score_key_likelihood(xa, pa[:, b], poi_idx, means, vars_)
        best = int(np.argmax(scores_all))
        recovered.append(best)

        if ktrue is not None:
            ranks = []
            for n in n_list:
                s = score_key_likelihood(xa[:n], pa[:n, b], poi_idx, means, vars_)
                order = np.argsort(s)[::-1]
                r = int(np.where(order == int(ktrue[b]))[0][0]) + 1
                ranks.append(r)
            per_byte_ranks[f"b{b:02d}"] = ranks

        print(f"byte{b:02d}: rec=0x{best:02X} n_poi={len(poi_idx)}")

    rec_hex = "".join(f"{x:02X}" for x in recovered)
    print("Recovered key:", rec_hex)
    out = {
        "profile_npz": args.profile_npz,
        "attack_npz": args.attack_npz,
        "n_poi": int(args.n_poi),
        "n_list": n_list,
        "recovered_key": recovered,
        "recovered_key_hex": rec_hex,
    }
    if ktrue is not None:
        rank1 = [sum(1 for b in range(16) if per_byte_ranks[f"b{b:02d}"][i] == 1) for i in range(len(n_list))]
        rank5 = [sum(1 for b in range(16) if per_byte_ranks[f"b{b:02d}"][i] <= 5) for i in range(len(n_list))]
        out["true_key_hex"] = "".join(f"{x:02X}" for x in ktrue)
        out["ranks_per_byte"] = per_byte_ranks
        out["rank1_count_vs_n"] = rank1
        out["rank5_count_vs_n"] = rank5
        print("rank1_count_vs_n=", rank1)
        print("rank5_count_vs_n=", rank5)

    out_path = Path(f"{args.out_prefix}_summary.json")
    out_path.write_text(json.dumps(out, indent=2))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
