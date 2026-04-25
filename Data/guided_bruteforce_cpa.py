#!/usr/bin/env python3
import argparse
import heapq
import itertools
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


def parse_key_hex(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "").upper()
    if len(t) != 32:
        raise ValueError("--true-key-hex must contain 32 hex chars")
    return np.array([int(t[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


def preprocess(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def load_pois(path: str) -> list[int]:
    with open(path, "r") as f:
        j = json.load(f)
    if "bytes" in j:
        pois = [int(row["poi"]) for row in j["bytes"]]
    elif "poi_global_per_byte" in j:
        pois = [int(x) for x in j["poi_global_per_byte"][:16]]
    else:
        raise ValueError("POI JSON missing bytes/poi_global_per_byte")
    if len(pois) != 16:
        raise ValueError("Expected 16 POIs")
    return pois


def scores_at_poi_window(traces: np.ndarray, pbyte: np.ndarray, poi: int, half_window: int) -> np.ndarray:
    n, tlen = traces.shape
    if half_window <= 0:
        a = max(0, min(tlen - 1, poi))
        b = a + 1
    else:
        a = max(0, poi - half_window)
        b = min(tlen, poi + half_window + 1)
    tw = traces[:, a:b]
    tc = tw - tw.mean(axis=0, keepdims=True)
    tstd = tw.std(axis=0, ddof=1) + 1e-15
    out = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h = HW[SBOX[np.bitwise_xor(pbyte, np.uint8(k))]]
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((n - 1) * hstd * tstd)
        out[k] = float(np.max(np.abs(corr)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided brute-force over uncertain AES bytes using CPA scores")
    ap.add_argument("--npz", required=True, help="Attack dataset (traces + plaintexts)")
    ap.add_argument("--poi-json", required=True, help="POI JSON")
    ap.add_argument("--n-traces", type=int, default=15000)
    ap.add_argument("--cal-ratio", type=float, default=0.5, help="Fraction for calibration split")
    ap.add_argument("--poi-half-window", type=int, default=0)
    ap.add_argument("--n-var-bytes", type=int, default=6, help="How many uncertain bytes to brute-force")
    ap.add_argument("--topk-per-byte", type=int, default=8, help="Candidates kept per variable byte")
    ap.add_argument("--topn-keys", type=int, default=30, help="Keep top N full-key candidates")
    ap.add_argument("--max-combos", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--true-key-hex", default=None, help="Optional final validation only")
    ap.add_argument("--out-prefix", default="guided_bruteforce")
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d["traces"][:args.n_traces]
    plains = d["plaintexts"][:args.n_traces, :16].astype(np.uint8)
    pois = load_pois(args.poi_json)
    proc = preprocess(traces)

    n = proc.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    # Keep both splits non-empty for small datasets as well.
    n_cal = int(n * args.cal_ratio)
    n_cal = max(100, n_cal)
    n_cal = min(n - 100, n_cal)
    i_cal = perm[:n_cal]
    i_eval = perm[n_cal:]
    cal = proc[i_cal]
    eva = proc[i_eval]
    p_cal = plains[i_cal]
    p_eval = plains[i_eval]

    scores_cal = np.zeros((16, 256), dtype=np.float64)
    scores_eval = np.zeros((16, 256), dtype=np.float64)
    top1 = np.zeros(16, dtype=np.uint8)
    margins = np.zeros(16, dtype=np.float64)
    topk_lists: list[list[int]] = []
    for b in range(16):
        s_cal = scores_at_poi_window(cal, p_cal[:, b], pois[b], int(args.poi_half_window))
        s_eval = scores_at_poi_window(eva, p_eval[:, b], pois[b], int(args.poi_half_window))
        scores_cal[b] = s_cal
        scores_eval[b] = s_eval
        order = np.argsort(s_cal)[::-1]
        top1[b] = np.uint8(int(order[0]))
        margins[b] = float(s_cal[order[0]] - s_cal[order[1]])
        topk_lists.append([int(x) for x in order[:args.topk_per_byte]])

    uncertain = [int(x) for x in np.argsort(margins)[:args.n_var_bytes]]
    locked = [int(b) for b in range(16) if b not in uncertain]
    locked_key = top1.copy()
    locked_eval_score = float(sum(scores_eval[b, int(locked_key[b])] for b in locked))

    choices = [topk_lists[b] for b in uncertain]
    n_combos = 1
    for c in choices:
        n_combos *= len(c)
    if n_combos > args.max_combos:
        raise ValueError(
            f"Too many combinations ({n_combos}). Reduce --n-var-bytes or --topk-per-byte, or increase --max-combos."
        )

    heap: list[tuple[float, tuple[int, ...]]] = []
    for combo in itertools.product(*choices):
        score = locked_eval_score
        for j, b in enumerate(uncertain):
            score += float(scores_eval[b, combo[j]])
        item = (score, combo)
        if len(heap) < args.topn_keys:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    best = sorted(heap, key=lambda x: x[0], reverse=True)
    keys_rows = []
    for rank, (score, combo) in enumerate(best, start=1):
        k = locked_key.copy()
        for j, b in enumerate(uncertain):
            k[b] = np.uint8(combo[j])
        k_hex = "".join(f"{int(x):02X}" for x in k)
        keys_rows.append(
            {
                "rank": rank,
                "score_eval_sum": float(score),
                "key_hex": k_hex,
                "key": [int(x) for x in k],
            }
        )

    out = {
        "npz": args.npz,
        "poi_json": args.poi_json,
        "n_traces": int(n),
        "n_cal": int(n_cal),
        "n_eval": int(n - n_cal),
        "poi_half_window": int(args.poi_half_window),
        "n_var_bytes": int(args.n_var_bytes),
        "topk_per_byte": int(args.topk_per_byte),
        "max_combos": int(args.max_combos),
        "n_combos": int(n_combos),
        "uncertain_bytes": [int(x) for x in uncertain],
        "locked_bytes": [int(x) for x in locked],
        "base_key_from_cal_top1_hex": "".join(f"{int(x):02X}" for x in top1),
        "byte_margins_cal": [float(x) for x in margins],
        "byte_topk_cal": {f"b{b:02d}": topk_lists[b] for b in range(16)},
        "top_keys": keys_rows,
    }

    true_key = parse_key_hex(args.true_key_hex)
    if true_key is not None:
        true_hex = "".join(f"{int(x):02X}" for x in true_key)
        out["true_key_hex"] = true_hex
        hit_rank = None
        for row in keys_rows:
            if row["key_hex"] == true_hex:
                hit_rank = int(row["rank"])
                break
        out["true_key_rank_in_top_keys"] = hit_rank
        out["top1_match_count"] = int(sum(int(top1[i]) == int(true_key[i]) for i in range(16)))

    out_path = Path(f"{args.out_prefix}_summary.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("Base key (cal top1):", out["base_key_from_cal_top1_hex"])
    print("Uncertain bytes    :", uncertain)
    print("Combinations       :", n_combos)
    print("Best candidate     :", keys_rows[0]["key_hex"])
    if true_key is not None:
        print("True key rank in top keys:", out["true_key_rank_in_top_keys"])
        print("Top1 byte matches       :", out["top1_match_count"], "/16")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
