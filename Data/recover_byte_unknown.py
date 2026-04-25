import argparse
import json

import numpy as np

from key_rank_from_mlp import AES_SBOX, HW, center_and_detrend, infer_probs


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
    return np.sum(lp[np.arange(n_eff)[:, None], hw_map], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Recover one AES key byte without knowing true key")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=160)
    ap.add_argument("--out", default="recover_byte_unknown.json")
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

    best = None
    rows = []
    for s in range(-args.max_shift, args.max_shift + 1):
        scores = final_key_scores(logp, pbyte, s)
        order = np.argsort(scores)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin = float(scores[top1] - scores[top2])
        row = {
            "shift": int(s),
            "top1_key": top1,
            "top2_key": top2,
            "margin": margin,
            "scores": scores,
        }
        rows.append(row)
        if best is None or row["margin"] > best["margin"]:
            best = row

    top_order = np.argsort(best["scores"])[::-1]
    top5 = [
        {"rank": i + 1, "key": int(k), "score": float(best["scores"][k])}
        for i, k in enumerate(top_order[:5])
    ]

    print(f"Recovered byte[{args.byte}] = 0x{best['top1_key']:02X}")
    print(f"best_shift={best['shift']} margin={best['margin']:.6f}")
    print("top-5 candidates:")
    for r in top5:
        print(f"  #{r['rank']}: 0x{r['key']:02X} score={r['score']:.4f}")

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "byte": int(args.byte),
        "best_shift": int(best["shift"]),
        "recovered_key_byte": int(best["top1_key"]),
        "margin": float(best["margin"]),
        "top5": top5,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
