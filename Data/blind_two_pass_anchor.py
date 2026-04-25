#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from blind_iterative_cpa import (
    aggregate_restarts,
    corr_all_keys,
    corr_one_key,
    make_tc_tstd,
    parse_hex_key,
    preprocess,
    run_one_restart,
)


def run_blind_pass(
    proc: np.ndarray,
    plains: np.ndarray,
    restarts: int,
    refine_iters: int,
    refine_radius: int,
    seed: int,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rr = []
    for i in range(restarts):
        r = run_one_restart(proc, plains, rng, refine_iters, refine_radius)
        med_conf = float(np.median(r["conf"]))
        key_hex = "".join(f"{int(x):02X}" for x in r["key"])
        print(f"restart {i + 1}/{restarts}: key={key_hex} median_conf={med_conf:.6f}")
        rr.append(r)
    key, rows = aggregate_restarts(rr)
    return key, rows, rr


def evaluate_ranks_at_poi(
    proc: np.ndarray,
    plains: np.ndarray,
    true_key: np.ndarray,
    poi_by_byte: np.ndarray,
) -> list[dict]:
    tc, tstd = make_tc_tstd(proc)
    out = []
    for b in range(16):
        poi = int(poi_by_byte[b])
        corr = corr_all_keys(tc, tstd, plains[:, b])
        scores = corr[:, poi]
        order = np.argsort(scores)[::-1]
        tk = int(true_key[b])
        rank = int(np.where(order == tk)[0][0]) + 1
        out.append(
            {
                "byte": b,
                "poi": poi,
                "top1": int(order[0]),
                "top2": int(order[1]),
                "true_key": tk,
                "rank": rank,
                "true_score": float(scores[tk]),
                "margin_top1_top2": float(scores[order[0]] - scores[order[1]]),
            }
        )
    return out


def recompute_anchor_pois(
    proc: np.ndarray,
    plains: np.ndarray,
    true_key: np.ndarray,
    anchor_bytes: list[int],
) -> dict[int, int]:
    tc, tstd = make_tc_tstd(proc)
    d = {}
    for b in anchor_bytes:
        c = corr_one_key(tc, tstd, plains[:, b], int(true_key[b]))
        d[int(b)] = int(np.argmax(c))
    return d


def rows_to_poi_array(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(16, dtype=np.int32)
    for r in rows:
        arr[int(r["byte"])] = int(r["median_poi"])
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-pass blind CPA: blind pass -> end-only rank eval -> anchor-guided pass")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-traces", type=int, default=10000)
    ap.add_argument("--win-start", type=int, default=0)
    ap.add_argument("--win-end", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--refine-radius", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--true-key-hex", required=True, help="Used only after pass1 to compute ranks and choose anchors.")
    ap.add_argument("--anchor-max-rank", type=int, default=1)
    ap.add_argument("--anchor-window-pad", type=int, default=220)
    ap.add_argument("--out-prefix", default="blind_two_pass_anchor")
    args = ap.parse_args()

    true_key = parse_hex_key(args.true_key_hex)
    if true_key is None:
        raise ValueError("true key required")

    d = np.load(args.npz)
    traces = d["traces"][: args.n_traces, args.win_start : args.win_end]
    plains = d["plaintexts"][: args.n_traces, :16].astype(np.uint8)
    proc = preprocess(traces)

    print("== pass1 (blind) ==")
    key1, rows1, rr1 = run_blind_pass(
        proc, plains, args.restarts, args.refine_iters, args.refine_radius, args.seed
    )
    key1_hex = "".join(f"{int(x):02X}" for x in key1)
    print("pass1 key:", key1_hex)

    poi1 = rows_to_poi_array(rows1)
    eval1 = evaluate_ranks_at_poi(proc, plains, true_key, poi1)
    rank1_count = int(sum(1 for r in eval1 if int(r["rank"]) == 1))
    print(f"pass1 rank1 bytes: {rank1_count}/16")

    anchor_bytes = [int(r["byte"]) for r in eval1 if int(r["rank"]) <= args.anchor_max_rank]
    print("anchor bytes:", anchor_bytes)

    anchor_pois = recompute_anchor_pois(proc, plains, true_key, anchor_bytes) if anchor_bytes else {}
    if anchor_pois:
        arr = np.array(list(anchor_pois.values()), dtype=np.int32)
        w0 = max(0, int(arr.min()) - args.anchor_window_pad)
        w1 = min(proc.shape[1], int(arr.max()) + args.anchor_window_pad + 1)
    else:
        w0, w1 = 0, proc.shape[1]
    print(f"anchor window (local): [{w0}, {w1}) / full [{0}, {proc.shape[1]})")

    print("== pass2 (anchor-guided blind) ==")
    proc2 = proc[:, w0:w1]
    key2, rows2, rr2 = run_blind_pass(
        proc2, plains, args.restarts, args.refine_iters, args.refine_radius, args.seed + 1000
    )
    key2_hex = "".join(f"{int(x):02X}" for x in key2)
    print("pass2 key:", key2_hex)

    poi2_local = rows_to_poi_array(rows2)
    poi2 = poi2_local + w0
    eval2 = evaluate_ranks_at_poi(proc, plains, true_key, poi2)
    rank2_count = int(sum(1 for r in eval2 if int(r["rank"]) == 1))
    print(f"pass2 rank1 bytes: {rank2_count}/16")

    out = {
        "npz": args.npz,
        "n_traces": int(args.n_traces),
        "window_global": [int(args.win_start), int(args.win_end)],
        "true_key_hex": "".join(f"{int(x):02X}" for x in true_key),
        "pass1": {
            "recovered_key_hex": key1_hex,
            "rank_eval_at_final_poi": eval1,
            "rank1_count": rank1_count,
            "bytes": rows1,
            "restarts_raw": [
                {
                    "idx": i,
                    "key_hex": "".join(f"{int(x):02X}" for x in r["key"]),
                    "median_conf": float(np.median(r["conf"])),
                    "poi": [int(x) for x in r["poi"]],
                    "conf": [float(x) for x in r["conf"]],
                }
                for i, r in enumerate(rr1)
            ],
        },
        "anchors": {
            "rule": f"rank <= {int(args.anchor_max_rank)} at end of pass1 only",
            "bytes": anchor_bytes,
            "recomputed_poi_truekey": {str(k): int(v) for k, v in anchor_pois.items()},
            "local_anchor_window": [int(w0), int(w1)],
            "anchor_window_pad": int(args.anchor_window_pad),
        },
        "pass2": {
            "recovered_key_hex": key2_hex,
            "rank_eval_at_final_poi": eval2,
            "rank1_count": rank2_count,
            "bytes": rows2,
            "restarts_raw": [
                {
                    "idx": i,
                    "key_hex": "".join(f"{int(x):02X}" for x in r["key"]),
                    "median_conf": float(np.median(r["conf"])),
                    "poi_local": [int(x) for x in r["poi"]],
                    "conf": [float(x) for x in r["conf"]],
                }
                for i, r in enumerate(rr2)
            ],
        },
    }

    out_json = Path(f"{args.out_prefix}_summary.json")
    out_json.write_text(json.dumps(out, indent=2))
    print("saved:", out_json)


if __name__ == "__main__":
    main()

