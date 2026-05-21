#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from highside_campaign import HW, SBOX, preprocess


KEYS = np.arange(256, dtype=np.uint8)


def hypotheses(model: str, pbyte: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    sx = SBOX[x]
    if model == "hw_sbox":
        return HW[sx].T
    if model == "hd_pt_sbox":
        return HW[np.bitwise_xor(pbyte[:, None], sx)].T
    if model == "hd_xor_sbox":
        return HW[np.bitwise_xor(x, sx)].T
    if model == "hw_xor":
        return HW[x].T
    if model == "hw_pt":
        return np.tile(HW[pbyte], (256, 1))
    raise ValueError(model)


def cpa_for_model(traces: np.ndarray, plaintexts: np.ndarray, key: np.ndarray, model: str) -> dict:
    tr = preprocess(traces.astype(np.float32), "center_detrend")
    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    rows = []
    ranks = []
    top5 = 0
    top20 = 0
    true_scores = []
    true_margins = []
    true_curve_max = np.zeros(tr.shape[1], dtype=np.float32)

    for b in range(16):
        hyp = hypotheses(model, plaintexts[:, b]).astype(np.float32)
        hc = hyp - hyp.mean(axis=1, keepdims=True)
        hstd = hyp.std(axis=1, ddof=1, keepdims=True) + 1e-15
        corr = np.abs((hc @ tc) / ((tr.shape[0] - 1) * hstd * tstd[None, :]))
        scores = corr.max(axis=1)
        pois = corr.argmax(axis=1)
        order = np.argsort(scores)[::-1]
        true_key = int(key[b])
        rank = int(np.where(order == true_key)[0][0]) + 1
        ranks.append(rank)
        top5 += int(rank <= 5)
        top20 += int(rank <= 20)
        true_scores.append(float(scores[true_key]))
        true_margin = float(scores[true_key] - scores[int(order[0])])
        true_margins.append(true_margin)
        true_curve_max = np.maximum(true_curve_max, corr[true_key])
        rows.append(
            {
                "byte": b,
                "true_key": true_key,
                "rank": rank,
                "true_score": float(scores[true_key]),
                "true_poi": int(pois[true_key]),
                "top_key": int(order[0]),
                "top_score": float(scores[int(order[0])]),
                "top_poi": int(pois[int(order[0])]),
                "true_minus_top_score": true_margin,
            }
        )

    return {
        "rank1": int(sum(r == 1 for r in ranks)),
        "top5": int(top5),
        "top20": int(top20),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_true_score": float(np.mean(true_scores)),
        "max_true_score": float(np.max(true_scores)),
        "mean_true_minus_top_score": float(np.mean(true_margins)),
        "ranks": ranks,
        "bytes": rows,
        "true_curve_global_max": float(true_curve_max.max()),
        "true_curve_global_poi": int(true_curve_max.argmax()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare first-round leakage models on an SCA dataset")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-traces", type=int, default=0)
    ap.add_argument("--sources", default="traces,traces_a,traces_b")
    ap.add_argument("--models", default="hw_sbox,hd_pt_sbox,hd_xor_sbox,hw_xor,hw_pt")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz)
    n = int(args.n_traces) if args.n_traces else int(d["traces"].shape[0])
    plaintexts = d["plaintexts"][:n, :16].astype(np.uint8)
    if "key" not in d.files:
        raise ValueError("Dataset must contain key for model validation")
    key = d["key"][:16].astype(np.uint8)
    sources = [s for s in args.sources.split(",") if s in d.files]
    models = [m for m in args.models.split(",") if m]

    out = {
        "npz": args.npz,
        "n_traces": n,
        "sources": {},
    }
    for src in sources:
        out["sources"][src] = {}
        print(f"\nSOURCE {src}")
        traces = d[src][:n]
        for model in models:
            res = cpa_for_model(traces, plaintexts, key, model)
            out["sources"][src][model] = res
            print(
                f"{model:12s} r1={res['rank1']:2d}/16 top5={res['top5']:2d}/16 "
                f"top20={res['top20']:2d}/16 mean_rank={res['mean_rank']:7.2f} "
                f"mean_true_corr={res['mean_true_score']:.4f} max_true_corr={res['max_true_score']:.4f}"
            )

    out_path = Path(args.out) if args.out else Path(args.npz).with_suffix("").with_name(Path(args.npz).stem + "_model_compare.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")

    if args.plot:
        for src in sources:
            labels = []
            means = []
            top5s = []
            for model in models:
                labels.append(model)
                means.append(out["sources"][src][model]["mean_rank"])
                top5s.append(out["sources"][src][model]["top5"])
            fig, ax1 = plt.subplots(figsize=(9, 4))
            ax1.bar(labels, means, color="tab:blue", alpha=0.75)
            ax1.set_ylabel("Mean rank")
            ax1.tick_params(axis="x", rotation=25)
            ax2 = ax1.twinx()
            ax2.plot(labels, top5s, color="tab:red", marker="o")
            ax2.set_ylabel("Top-5 bytes / 16")
            fig.tight_layout()
            fig.savefig(out_path.with_name(f"{out_path.stem}_{src}.png"), dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    main()
