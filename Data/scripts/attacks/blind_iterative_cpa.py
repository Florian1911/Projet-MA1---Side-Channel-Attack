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


def parse_hex_key(s: str | None) -> np.ndarray | None:
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


def make_tc_tstd(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    return tc, tstd


def corr_all_keys(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray) -> np.ndarray:
    n = tc.shape[0]
    hyp = HW[SBOX[np.bitwise_xor(pbyte[:, None], np.arange(256, dtype=np.uint8)[None, :])]].T
    hyp_c = hyp - hyp.mean(axis=1, keepdims=True)
    hyp_std = hyp.std(axis=1, ddof=1, keepdims=True) + 1e-15
    corr = (hyp_c @ tc) / ((n - 1) * hyp_std * tstd[None, :])
    return np.abs(corr)


def corr_one_key(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, key: int) -> np.ndarray:
    h = HW[SBOX[np.bitwise_xor(pbyte, np.uint8(key))]]
    hc = h - h.mean()
    hstd = h.std(ddof=1) + 1e-15
    corr = (hc @ tc) / ((tc.shape[0] - 1) * hstd * tstd)
    return np.abs(corr)


def top1_top2(scores_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top1_key = np.argmax(scores_2d, axis=0)
    top1_val = scores_2d[top1_key, np.arange(scores_2d.shape[1])]
    part = np.partition(scores_2d, -2, axis=0)
    top2_val = part[-2, :]
    return top1_key, top1_val, top2_val


def choose_initial_poi_and_key(
    tc_a: np.ndarray,
    tstd_a: np.ndarray,
    pt_a: np.ndarray,
    tc_b: np.ndarray,
    tstd_b: np.ndarray,
    pt_b: np.ndarray,
    min_stable_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pois = np.zeros(16, dtype=np.int32)
    keys = np.zeros(16, dtype=np.uint8)
    conf = np.zeros(16, dtype=np.float64)
    tlen = tc_a.shape[1]
    for b in range(16):
        c_a = corr_all_keys(tc_a, tstd_a, pt_a[:, b])
        c_b = corr_all_keys(tc_b, tstd_b, pt_b[:, b])
        k1_a, s1_a, s2_a = top1_top2(c_a)
        k1_b, s1_b, s2_b = top1_top2(c_b)
        stable = (k1_a == k1_b)
        stable_ratio = float(stable.mean())

        margin_a = s1_a - s2_a
        margin_b = s1_b - s2_b
        metric = np.where(stable, margin_a + margin_b, -1e9) + 0.25 * (s1_a + s1_b)

        if stable_ratio < min_stable_ratio:
            comb = c_a + c_b
            k1_c, s1_c, s2_c = top1_top2(comb)
            metric = (s1_c - s2_c) + 0.2 * s1_c
            t = int(np.argmax(metric))
            k = int(k1_c[t])
            c = float(metric[t])
        else:
            t = int(np.argmax(metric))
            k = int(k1_a[t])
            c = float(metric[t])

        pois[b] = t
        keys[b] = np.uint8(k)
        conf[b] = c
    return pois, keys, conf


def refine_once(
    tc_a: np.ndarray,
    tstd_a: np.ndarray,
    pt_a: np.ndarray,
    tc_b: np.ndarray,
    tstd_b: np.ndarray,
    pt_b: np.ndarray,
    pois: np.ndarray,
    keys: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tlen = tc_a.shape[1]
    new_pois = pois.copy()
    new_keys = keys.copy()
    conf = np.zeros(16, dtype=np.float64)
    for b in range(16):
        c_a = corr_one_key(tc_a, tstd_a, pt_a[:, b], int(keys[b]))
        c_b = corr_one_key(tc_b, tstd_b, pt_b[:, b], int(keys[b]))
        left = max(0, int(pois[b]) - radius)
        right = min(tlen, int(pois[b]) + radius + 1)
        local_metric = np.minimum(c_a[left:right], c_b[left:right])
        t = int(left + np.argmax(local_metric))
        new_pois[b] = t

        col_a = tc_a[:, t]
        col_b = tc_b[:, t]
        scores = np.zeros(256, dtype=np.float64)
        for k in range(256):
            h_a = HW[SBOX[np.bitwise_xor(pt_a[:, b], np.uint8(k))]]
            h_b = HW[SBOX[np.bitwise_xor(pt_b[:, b], np.uint8(k))]]
            hc_a = h_a - h_a.mean()
            hc_b = h_b - h_b.mean()
            hstd_a = h_a.std(ddof=1) + 1e-15
            hstd_b = h_b.std(ddof=1) + 1e-15
            ca = abs((hc_a @ col_a) / ((len(h_a) - 1) * hstd_a * tstd_a[t]))
            cb = abs((hc_b @ col_b) / ((len(h_b) - 1) * hstd_b * tstd_b[t]))
            scores[k] = ca + cb

        order = np.argsort(scores)[::-1]
        new_keys[b] = np.uint8(int(order[0]))
        conf[b] = float(scores[order[0]] - scores[order[1]])
    return new_pois, new_keys, conf


def run_one_restart(
    traces: np.ndarray,
    plains: np.ndarray,
    rng: np.random.Generator,
    refine_iters: int,
    refine_radius: int,
) -> dict:
    n = traces.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    ia = perm[:half]
    ib = perm[half:]
    ta, tb = traces[ia], traces[ib]
    pa, pb = plains[ia], plains[ib]
    tc_a, tstd_a = make_tc_tstd(ta)
    tc_b, tstd_b = make_tc_tstd(tb)

    pois, keys, init_conf = choose_initial_poi_and_key(tc_a, tstd_a, pa, tc_b, tstd_b, pb)
    conf = init_conf.copy()

    for _ in range(refine_iters):
        pois2, keys2, conf2 = refine_once(tc_a, tstd_a, pa, tc_b, tstd_b, pb, pois, keys, refine_radius)
        if np.array_equal(keys2, keys) and np.array_equal(pois2, pois):
            conf = conf2
            break
        pois, keys, conf = pois2, keys2, conf2

    return {
        "key": keys.astype(np.uint8),
        "poi": pois.astype(np.int32),
        "conf": conf.astype(np.float64),
    }


def aggregate_restarts(results: list[dict]) -> tuple[np.ndarray, list[dict]]:
    final = np.zeros(16, dtype=np.uint8)
    rows = []
    for b in range(16):
        weights = np.zeros(256, dtype=np.float64)
        pois = []
        key_votes = []
        for r in results:
            k = int(r["key"][b])
            c = max(0.0, float(r["conf"][b]))
            weights[k] += c
            pois.append(int(r["poi"][b]))
            key_votes.append(k)
        kf = int(np.argmax(weights))
        final[b] = np.uint8(kf)
        rows.append(
            {
                "byte": b,
                "final_key": kf,
                "final_key_hex": f"{kf:02X}",
                "vote_weight": float(weights[kf]),
                "vote_count": int(sum(1 for x in key_votes if x == kf)),
                "median_poi": int(np.median(pois)),
                "all_votes": key_votes,
            }
        )
    return final, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust blind iterative CPA with split-consistency and multi-restart voting")
    ap.add_argument("--npz", required=True, help="Attack dataset with traces + plaintexts")
    ap.add_argument("--n-traces", type=int, default=20000)
    ap.add_argument("--win-start", type=int, default=0)
    ap.add_argument("--win-end", type=int, default=3968)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--refine-radius", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-prefix", default="blind_iterative_cpa")
    ap.add_argument("--true-key-hex", default=None, help="Optional, used only for final evaluation")
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d["traces"][:args.n_traces, args.win_start:args.win_end]
    plains = d["plaintexts"][:args.n_traces, :16].astype(np.uint8)
    proc = preprocess(traces)

    rng = np.random.default_rng(args.seed)
    restarts = []
    for i in range(args.restarts):
        r = run_one_restart(proc, plains, rng, args.refine_iters, args.refine_radius)
        key_hex = "".join(f"{int(x):02X}" for x in r["key"])
        med_conf = float(np.median(r["conf"]))
        print(f"restart {i + 1}/{args.restarts}: key={key_hex} median_conf={med_conf:.6f}")
        restarts.append(r)

    final_key, byte_rows = aggregate_restarts(restarts)
    final_hex = "".join(f"{int(x):02X}" for x in final_key)
    print("Final key (blind):", final_hex)

    out = {
        "npz": args.npz,
        "n_traces": int(args.n_traces),
        "window": [int(args.win_start), int(args.win_end)],
        "restarts": int(args.restarts),
        "refine_iters": int(args.refine_iters),
        "refine_radius": int(args.refine_radius),
        "seed": int(args.seed),
        "recovered_key": [int(x) for x in final_key],
        "recovered_key_hex": final_hex,
        "bytes": byte_rows,
        "restarts_raw": [
            {
                "idx": i,
                "key_hex": "".join(f"{int(x):02X}" for x in r["key"]),
                "median_conf": float(np.median(r["conf"])),
                "poi": [int(x) for x in r["poi"]],
                "conf": [float(x) for x in r["conf"]],
            }
            for i, r in enumerate(restarts)
        ],
        "blind": True,
    }

    true_key = parse_hex_key(args.true_key_hex)
    if true_key is not None:
        n_ok = int(np.sum(final_key == true_key))
        bad = [i for i in range(16) if int(final_key[i]) != int(true_key[i])]
        print("True key        :", "".join(f"{int(x):02X}" for x in true_key))
        print(f"Match           : {n_ok}/16")
        print("Bad bytes       :", bad)
        out["true_key_hex"] = "".join(f"{int(x):02X}" for x in true_key)
        out["match_count"] = n_ok
        out["bad_bytes"] = bad

    out_json = Path(f"{args.out_prefix}_summary.json")
    out_json.write_text(json.dumps(out, indent=2))
    print("saved:", out_json)


if __name__ == "__main__":
    main()
