import argparse
import json

import numpy as np

from key_rank_from_mlp import (
    AES_SBOX,
    HW,
    center_and_detrend,
    infer_probs,
)


def final_key_scores(logp: np.ndarray, pbyte: np.ndarray, shift: int) -> np.ndarray:
    n = logp.shape[0]
    if shift >= 0:
        n_eff = n - shift
        lp = logp[:n_eff]
        pb = pbyte[shift:]
    else:
        s = -shift
        n_eff = n - s
        lp = logp[s:]
        pb = pbyte[:n_eff]

    keys = np.arange(256, dtype=np.uint8)
    hw_map = HW[AES_SBOX[np.bitwise_xor(pb[:, None], keys[None, :])]]
    # Sum log-prob for each key
    return np.sum(lp[np.arange(n_eff)[:, None], hw_map], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_50k_hw_mlp.npz")
    ap.add_argument("--dataset", default="attack_20k_hw.npz")
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=160)
    ap.add_argument("--out", default="attack_shift_scan.json")
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

    rows = []
    best = None
    for s in range(-args.max_shift, args.max_shift + 1):
        scores = final_key_scores(logp, pbyte, s)
        order = np.argsort(scores)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin = float(scores[top1] - scores[top2])
        row = {
            "shift": int(s),
            "top1_key": top1,
            "top1_score": float(scores[top1]),
            "top2_key": top2,
            "top2_score": float(scores[top2]),
            "margin": margin,
        }
        rows.append(row)
        if best is None or margin > best["margin"]:
            best = row

    print(f"best shift={best['shift']} top1=0x{best['top1_key']:02X} margin={best['margin']:.6f}")
    print(f"top2 at best shift=0x{best['top2_key']:02X}")

    rows_sorted = sorted(rows, key=lambda r: r["margin"], reverse=True)
    print("top-5 shifts by margin:")
    for i, r in enumerate(rows_sorted[:5], start=1):
        print(
            f"  #{i}: shift={r['shift']:+d} top1=0x{r['top1_key']:02X} "
            f"top2=0x{r['top2_key']:02X} margin={r['margin']:.6f}"
        )

    with open(args.out, "w") as f:
        json.dump({"best": best, "top_by_margin": rows_sorted[:20]}, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
