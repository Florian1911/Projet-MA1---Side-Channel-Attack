import argparse
import json

import numpy as np

from recover_byte_unknown import final_key_scores
from key_rank_from_mlp import center_and_detrend, infer_probs


def parse_points(raw: str, n_total: int):
    pts = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if 1 <= v <= n_total:
            pts.append(v)
    pts = sorted(set(pts))
    if n_total not in pts:
        pts.append(n_total)
    return pts


def best_for_subset(logp: np.ndarray, pbyte: np.ndarray, max_shift: int):
    best = None
    for s in range(-max_shift, max_shift + 1):
        scores = final_key_scores(logp, pbyte, s)
        order = np.argsort(scores)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin = float(scores[top1] - scores[top2])
        row = {
            "shift": int(s),
            "top1": top1,
            "top2": top2,
            "margin": margin,
        }
        if best is None or row["margin"] > best["margin"]:
            best = row
    return best


def main():
    ap = argparse.ArgumentParser(description="Unknown-key convergence by traces count for one byte")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=160)
    ap.add_argument("--points", default="100,200,500,1000,2000,5000,10000,20000")
    ap.add_argument("--out", default="unknown_convergence_byte.json")
    args = ap.parse_args()

    m = np.load(args.model)
    d = np.load(args.dataset)

    traces = d[args.traces_key].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)
    pbyte = plains[:, args.byte]

    x = center_and_detrend(traces)
    poi_idx = m["poi_idx"].astype(np.int32)
    mu = m["mu"].astype(np.float32)
    sigma = m["sigma"].astype(np.float32)
    x = x[:, poi_idx]
    x = (x - mu) / (sigma + 1e-6)

    probs = infer_probs(x, m)
    probs = np.clip(probs, 1e-12, 1.0)
    logp = np.log(probs)

    pts = parse_points(args.points, logp.shape[0])
    rows = []
    for n in pts:
        b = best_for_subset(logp[:n], pbyte[:n], args.max_shift)
        rows.append(
            {
                "n_traces": int(n),
                "top1": int(b["top1"]),
                "top2": int(b["top2"]),
                "margin": float(b["margin"]),
                "best_shift": int(b["shift"]),
            }
        )
        print(
            f"n={n:6d} top1=0x{b['top1']:02X} top2=0x{b['top2']:02X} margin={b['margin']:.4f} shift={b['shift']:+d}"
        )

    with open(args.out, "w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "byte": int(args.byte),
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
