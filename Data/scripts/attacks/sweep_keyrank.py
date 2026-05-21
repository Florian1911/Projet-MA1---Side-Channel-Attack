import argparse
import json
import numpy as np

from key_rank_from_mlp import AES_SBOX, HW, center_and_detrend, infer_probs


def rank_for_n(logp: np.ndarray, pbyte: np.ndarray, true_key: int, n_use: int, repeats: int, rng: np.random.Generator) -> float:
    lp = logp[:n_use]
    pb = pbyte[:n_use]
    keys = np.arange(256, dtype=np.uint8)
    hw_map = HW[AES_SBOX[np.bitwise_xor(pb[:, None], keys[None, :])]]  # [n,256]
    ll = lp[np.arange(n_use)[:, None], hw_map]  # [n,256]

    avg_rank = 0.0
    for _ in range(repeats):
        perm = rng.permutation(n_use)
        scores = np.sum(ll[perm], axis=0)
        tscore = scores[true_key]
        rank = 1 + int(np.count_nonzero(scores > tscore))
        avg_rank += rank
    return avg_rank / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_newkey_20k_mlp.npz")
    ap.add_argument("--dataset", default="attack_newkey_20k_hw.npz")
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--true-key", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--points", default="100,200,300,400,500,800,1000,1500,2000,3000,5000,8000,12000,16000,20000")
    ap.add_argument("--out", default="keyrank_sweep.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
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

    n_total = logp.shape[0]
    pts = []
    for tok in args.points.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if 1 <= v <= n_total:
            pts.append(v)
    pts = sorted(set(pts))
    if n_total not in pts:
        pts.append(n_total)

    rows = []
    for n in pts:
        r = rank_for_n(logp, pbyte, args.true_key, n, args.repeats, rng)
        rows.append({"n_traces": n, "avg_rank": float(r)})
        print(f"n={n:6d} -> avg_rank={r:.3f}")

    traces_to_rank1 = None
    for row in rows:
        if row["avg_rank"] <= 1.0:
            traces_to_rank1 = row["n_traces"]
            break

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "byte": args.byte,
        "true_key": args.true_key,
        "repeats": args.repeats,
        "rows": rows,
        "traces_to_rank1_from_grid": traces_to_rank1,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
