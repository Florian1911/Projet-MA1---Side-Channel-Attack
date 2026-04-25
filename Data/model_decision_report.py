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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
KEYS = np.arange(256, dtype=np.uint8)


def preprocess(traces: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return traces.astype(np.float64, copy=False)
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    return t - np.outer((t @ x) / np.dot(x, x), x)


def tc_tstd(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    return tc, tstd


def hyp_matrix(pbyte: np.ndarray, model: str) -> np.ndarray:
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    y = SBOX[x]
    if model == "hw_sbox":
        h = HW[y]
    elif model == "hd_pt":
        h = HW[np.bitwise_xor(y, pbyte[:, None])]
    elif model == "hd_inout":
        h = HW[np.bitwise_xor(y, x)]
    else:
        raise ValueError(model)
    return h.astype(np.float64, copy=False)


def select_poi_on_calib(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, model: str, true_key: int) -> int:
    h = hyp_matrix(pbyte, model)[:, int(true_key)]
    hc = h - h.mean()
    hstd = h.std(ddof=1) + 1e-15
    corr = (hc @ tc) / ((tc.shape[0] - 1) * hstd * tstd)
    return int(np.argmax(np.abs(corr)))


def rank_on_eval(tc: np.ndarray, tstd: np.ndarray, pbyte: np.ndarray, model: str, poi: int, true_key: int) -> int:
    h = hyp_matrix(pbyte, model)
    hc = h - h.mean(axis=0, keepdims=True)
    hstd = h.std(axis=0, ddof=1) + 1e-15
    y = tc[:, poi]
    ystd = tstd[poi]
    scores = np.abs((hc.T @ y) / ((tc.shape[0] - 1) * hstd * ystd))
    order = np.argsort(scores)[::-1]
    return int(np.where(order == int(true_key))[0][0]) + 1


def eval_one_split(
    traces: np.ndarray,
    plains: np.ndarray,
    key: np.ndarray,
    model: str,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[int, int, float]:
    tc_c, tstd_c = tc_tstd(traces[calib_idx])
    tc_e, tstd_e = tc_tstd(traces[eval_idx])
    ranks = []
    for b in range(16):
        p_cal = plains[calib_idx, b]
        p_eval = plains[eval_idx, b]
        tk = int(key[b])
        poi = select_poi_on_calib(tc_c, tstd_c, p_cal, model, tk)
        r = rank_on_eval(tc_e, tstd_e, p_eval, model, poi, tk)
        ranks.append(r)
    return int(sum(r == 1 for r in ranks)), int(sum(r <= 5 for r in ranks)), float(np.mean(ranks))


def main() -> None:
    ap = argparse.ArgumentParser(description="Model decision report with calib/eval split (no leakage from eval).")
    ap.add_argument(
        "--datasets",
        default="dataset_aes_sca_highside.npz,highside_traces_calib500_known.npz,highside_traces_a_calib500_known.npz,highside_traces_b_calib500_known.npz",
    )
    ap.add_argument("--models", default="hw_sbox,hd_pt,hd_inout")
    ap.add_argument("--preproc", default="none,center_detrend")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--calib-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="model_decision_report.json")
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    preprocs = [x.strip() for x in args.preproc.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)
    all_rows = []

    for ds in datasets:
        d = np.load(ds)
        traces0 = d["traces"].astype(np.float32)
        plains = d["plaintexts"][:, :16].astype(np.uint8)
        key = d["key"][:16].astype(np.uint8)
        n = traces0.shape[0]
        n_cal = int(max(64, min(n - 64, round(args.calib_ratio * n))))
        print(f"\n== dataset: {ds} (n={n}, calib={n_cal}, eval={n-n_cal}) ==")

        for prep in preprocs:
            traces = preprocess(traces0, prep)
            for m in models:
                r1s, r5s, means = [], [], []
                for _ in range(args.splits):
                    perm = rng.permutation(n)
                    calib_idx = perm[:n_cal]
                    eval_idx = perm[n_cal:]
                    r1, r5, mr = eval_one_split(traces, plains, key, m, calib_idx, eval_idx)
                    r1s.append(r1)
                    r5s.append(r5)
                    means.append(mr)
                row = {
                    "dataset": ds,
                    "preproc": prep,
                    "model": m,
                    "splits": int(args.splits),
                    "rank1_mean": float(np.mean(r1s)),
                    "rank1_std": float(np.std(r1s)),
                    "rank5_mean": float(np.mean(r5s)),
                    "rank5_std": float(np.std(r5s)),
                    "mean_rank_mean": float(np.mean(means)),
                    "mean_rank_std": float(np.std(means)),
                }
                all_rows.append(row)
                print(
                    f"[{prep:13s} | {m:8s}] "
                    f"r1={row['rank1_mean']:.2f}±{row['rank1_std']:.2f} /16 "
                    f"r5={row['rank5_mean']:.2f}±{row['rank5_std']:.2f} /16 "
                    f"mean_rank={row['mean_rank_mean']:.2f}"
                )

    summary_by_model = []
    for m in models:
        rows = [r for r in all_rows if r["model"] == m]
        summary_by_model.append(
            {
                "model": m,
                "rank1_mean_global": float(np.mean([r["rank1_mean"] for r in rows])),
                "rank5_mean_global": float(np.mean([r["rank5_mean"] for r in rows])),
                "mean_rank_global": float(np.mean([r["mean_rank_mean"] for r in rows])),
            }
        )
    winner = max(
        summary_by_model,
        key=lambda r: (r["rank1_mean_global"], r["rank5_mean_global"], -r["mean_rank_global"]),
    )

    out = {
        "datasets": datasets,
        "models": models,
        "preprocs": preprocs,
        "splits": int(args.splits),
        "calib_ratio": float(args.calib_ratio),
        "rows": all_rows,
        "summary_by_model": summary_by_model,
        "winner": winner,
        "decision_rule": "maximize rank1_mean_global, then rank5_mean_global, then minimize mean_rank_global",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nWinner:", winner)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

