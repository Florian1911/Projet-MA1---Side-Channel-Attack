import argparse
import json
from pathlib import Path

import numpy as np

AES_SBOX = np.array([
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
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(z: np.ndarray) -> np.ndarray:
    zz = z - z.max(axis=1, keepdims=True)
    ee = np.exp(zz)
    return ee / (ee.sum(axis=1, keepdims=True) + 1e-12)


def infer_probs(x: np.ndarray, model: dict, batch_size: int = 4096) -> np.ndarray:
    w1, b1 = model["w1"], model["b1"]
    w2, b2 = model["w2"], model["b2"]
    w3, b3 = model["w3"], model["b3"]
    n = x.shape[0]
    out = np.zeros((n, b3.shape[0]), dtype=np.float32)
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        z1 = xb @ w1 + b1
        a1 = relu(z1)
        z2 = a1 @ w2 + b2
        a2 = relu(z2)
        z3 = a2 @ w3 + b3
        out[i:i + batch_size] = softmax(z3).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_50k_hw_mlp.npz")
    ap.add_argument("--dataset", default="dataset_50k_hw.npz")
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--true-key", type=lambda x: int(x, 0), default=None)
    ap.add_argument(
        "--label-mode",
        choices=["auto", "hw", "sbox"],
        default="auto",
        help="How model classes map to leakage: hw (9 classes) or sbox (256 classes).",
    )
    ap.add_argument("--n-traces", type=int, default=0, help="0 = use all traces")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="key_rank_result.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m = np.load(args.model)
    d = np.load(args.dataset)

    traces = d[args.traces_key].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)
    if traces.shape[0] != plains.shape[0]:
        raise ValueError("traces and plaintexts length mismatch")

    if args.true_key is not None:
        true_key = int(args.true_key)
    elif "key" in d.files and len(d["key"]) > args.byte:
        true_key = int(d["key"][args.byte])
    elif "key" in m.files and len(m["key"]) > args.byte:
        true_key = int(m["key"][args.byte])
    else:
        true_key = None

    x = center_and_detrend(traces)
    poi_idx = m["poi_idx"].astype(np.int32)
    mu = m["mu"].astype(np.float32)
    sigma = m["sigma"].astype(np.float32)
    x = x[:, poi_idx]
    x = (x - mu) / (sigma + 1e-6)

    probs = infer_probs(x, m)
    probs = np.clip(probs, 1e-12, 1.0)

    n_total = probs.shape[0]
    n_use = n_total if args.n_traces <= 0 else min(args.n_traces, n_total)
    probs = probs[:n_use]
    pbyte = plains[:n_use, args.byte]

    keys = np.arange(256, dtype=np.uint8)
    sbox_map = AES_SBOX[np.bitwise_xor(pbyte[:, None], keys[None, :])]  # [n,256]

    if args.label_mode == "auto":
        n_classes = int(probs.shape[1])
        if n_classes == 9:
            label_mode = "hw"
        elif n_classes == 256:
            label_mode = "sbox"
        else:
            raise ValueError(
                f"cannot infer label mode from n_classes={n_classes}; pass --label-mode hw|sbox"
            )
    else:
        label_mode = args.label_mode

    if label_mode == "hw":
        class_map = HW[sbox_map]
    else:
        class_map = sbox_map

    logp = np.log(probs[np.arange(n_use)[:, None], class_map])  # [n,256]

    avg_rank = np.zeros((n_use,), dtype=np.float64) if true_key is not None else None
    first_rank = None
    first_final_scores = None

    for r in range(args.repeats):
        perm = rng.permutation(n_use)
        lp = logp[perm]
        scores = np.zeros((256,), dtype=np.float64)
        ranks = np.zeros((n_use,), dtype=np.int32) if true_key is not None else None
        for i in range(n_use):
            scores += lp[i]
            if true_key is not None:
                tscore = scores[true_key]
                ranks[i] = 1 + int(np.count_nonzero(scores > tscore))
        if true_key is not None:
            avg_rank += ranks
        if r == 0:
            if ranks is not None:
                first_rank = ranks.copy()
            first_final_scores = scores.copy()

    final_rank = None
    if true_key is not None:
        avg_rank /= float(args.repeats)
        final_rank = int(round(avg_rank[-1]))

    traces_to_rank1 = None
    if true_key is not None:
        idx = np.where(avg_rank <= 1.0)[0]
        if idx.size > 0:
            traces_to_rank1 = int(idx[0] + 1)

    # top-5 keys from first repeat final scores
    order = np.argsort(first_final_scores)[::-1]
    top5 = [{"key": int(k), "score": float(first_final_scores[k])} for k in order[:5]]

    print(f"dataset={args.dataset}, model={args.model}")
    if true_key is None:
        print(f"n_use={n_use}, repeats={args.repeats}, true_key=UNKNOWN (blind mode)")
    else:
        print(f"n_use={n_use}, repeats={args.repeats}, true_key=0x{true_key:02X}")
    print(f"label_mode={label_mode}")
    if true_key is None:
        print("final avg rank=not available (blind mode)")
    else:
        print(f"final avg rank={final_rank}")
        if traces_to_rank1 is None:
            print("traces_to_rank1: not reached")
        else:
            print(f"traces_to_rank1: {traces_to_rank1}")
    print("top5 final keys (repeat #1):")
    for i, row in enumerate(top5, start=1):
        print(f"  #{i}: key=0x{row['key']:02X} score={row['score']:.4f}")

    cp = None
    if true_key is not None:
        checkpoints = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, n_use]
        checkpoints = sorted(set([c for c in checkpoints if 1 <= c <= n_use]))
        cp = {str(c): float(avg_rank[c - 1]) for c in checkpoints}

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "byte": int(args.byte),
        "true_key": (int(true_key) if true_key is not None else None),
        "n_use": int(n_use),
        "repeats": int(args.repeats),
        "label_mode": label_mode,
        "final_avg_rank": (int(final_rank) if final_rank is not None else None),
        "traces_to_rank1": traces_to_rank1,
        "checkpoints_avg_rank": cp,
        "top5_final_scores_repeat1": top5,
        "blind_mode": (true_key is None),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.out}")

    if true_key is not None and first_rank is not None:
        np.savez(
            Path(args.out).with_suffix(".npz"),
            avg_rank=avg_rank.astype(np.float32),
            first_rank=first_rank.astype(np.int32),
        )


if __name__ == "__main__":
    main()
