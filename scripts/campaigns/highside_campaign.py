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


def preprocess(tr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return tr.astype(np.float64, copy=False)
    t = tr.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def cpa_scores(tr: np.ndarray, pbyte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = tr.shape[0]
    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    scores = np.zeros(256, dtype=np.float64)
    pois = np.zeros(256, dtype=np.int32)
    for k in range(256):
        h = HW[SBOX[np.bitwise_xor(pbyte, np.uint8(k))]]
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        c = (hc @ tc) / ((n - 1) * hstd * tstd)
        ac = np.abs(c)
        i = int(np.argmax(ac))
        scores[k] = float(ac[i])
        pois[k] = i
    return scores, pois


def eval_known(tr: np.ndarray, pt: np.ndarray, key: np.ndarray) -> dict:
    ranks = []
    rows = []
    recovered = []
    for b in range(16):
        scores, pois = cpa_scores(tr, pt[:, b])
        order = np.argsort(scores)[::-1]
        true_b = int(key[b])
        rank = int(np.where(order == true_b)[0][0]) + 1
        top = int(order[0])
        recovered.append(top)
        ranks.append(rank)
        rows.append(
            {
                "byte": b,
                "true_key": true_b,
                "top_key": top,
                "rank": rank,
                "poi": int(pois[top]),
                "true_score": float(scores[true_b]),
                "margin_top1_top2": float(scores[order[0]] - scores[order[1]]),
            }
        )
    return {
        "rank1_count": int(sum(r == 1 for r in ranks)),
        "rank5_count": int(sum(r <= 5 for r in ranks)),
        "mean_rank": float(np.mean(ranks)),
        "recovered_key_hex": "".join(f"{x:02X}" for x in recovered),
        "bytes": rows,
    }


def eval_blind_direct(tr: np.ndarray, pt: np.ndarray, true_key: np.ndarray) -> dict:
    recovered = []
    rows = []
    for b in range(16):
        scores, pois = cpa_scores(tr, pt[:, b])
        order = np.argsort(scores)[::-1]
        top = int(order[0])
        recovered.append(top)
        rows.append(
            {
                "byte": b,
                "top_key": top,
                "poi": int(pois[top]),
                "margin_top1_top2": float(scores[order[0]] - scores[order[1]]),
            }
        )
    match = int(sum(int(recovered[i]) == int(true_key[i]) for i in range(16)))
    return {
        "byte_match_count": match,
        "recovered_key_hex": "".join(f"{x:02X}" for x in recovered),
        "bytes": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="High-side CPA campaign")
    ap.add_argument("--npz", default="dataset_aes_sca_highside.npz")
    ap.add_argument("--out", default="highside_campaign_summary.json")
    args = ap.parse_args()

    d = np.load(args.npz)
    src_map = {
        "traces": d["traces"].astype(np.float32),
        "traces_a": d["traces_a"].astype(np.float32),
        "traces_b": d["traces_b"].astype(np.float32),
    }
    pt = d["plaintexts"][:, :16].astype(np.uint8)
    key = d["key"][:16].astype(np.uint8)

    n_list = [200, 300, 500, 700, 1000]
    modes = ["none", "center_detrend"]

    known_rows = []
    blind_rows = []
    for src_name, src_tr in src_map.items():
        for mode in modes:
            tr_all = preprocess(src_tr, mode)
            for n in n_list:
                tr = tr_all[:n]
                ptn = pt[:n]
                kn = eval_known(tr, ptn, key)
                bl = eval_blind_direct(tr, ptn, key)
                known_rows.append(
                    {
                        "source": src_name,
                        "preproc": mode,
                        "n_traces": n,
                        "rank1_count": kn["rank1_count"],
                        "rank5_count": kn["rank5_count"],
                        "mean_rank": kn["mean_rank"],
                        "recovered_key_hex": kn["recovered_key_hex"],
                    }
                )
                blind_rows.append(
                    {
                        "source": src_name,
                        "preproc": mode,
                        "n_traces": n,
                        "byte_match_count": bl["byte_match_count"],
                        "recovered_key_hex": bl["recovered_key_hex"],
                    }
                )
                print(
                    f"[{src_name:8s} | {mode:13s} | n={n:4d}] "
                    f"known r1={kn['rank1_count']:2d}/16 | blind match={bl['byte_match_count']:2d}/16"
                )

    best_known = max(known_rows, key=lambda r: (r["rank1_count"], r["rank5_count"], -r["mean_rank"]))
    best_blind = max(blind_rows, key=lambda r: r["byte_match_count"])

    out = {
        "npz": args.npz,
        "known_sweep": known_rows,
        "blind_sweep": blind_rows,
        "best_known": best_known,
        "best_blind": best_blind,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("best_known:", best_known)
    print("best_blind:", best_blind)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
