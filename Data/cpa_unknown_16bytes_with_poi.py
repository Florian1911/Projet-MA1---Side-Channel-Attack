#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = np.dot(x, x)
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t


def parse_key_hex(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    k = s.strip().replace(" ", "").replace(":", "").replace(",", "")
    if len(k) != 32:
        raise ValueError("--true-key-hex doit contenir 32 hex chars")
    return np.array([int(k[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


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

    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h = HW[SBOX[np.bitwise_xor(pbyte, k)]]
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((n - 1) * hstd * tstd)
        scores[k] = float(np.max(np.abs(corr)))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="CPA unknown key (16 bytes) with fixed POIs + convergence")
    ap.add_argument("--npz", required=True, help="Dataset d'attaque unknown-key (traces + plaintexts)")
    ap.add_argument("--poi-json", required=True, help="JSON de référence contenant les POI par byte")
    ap.add_argument("--out-prefix", default="unknown_key_attack")
    ap.add_argument("--n-list", default="500,1000,2000,3000,5000,7000,10000")
    ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--poi-half-window", type=int, default=0,
                    help="Fenêtre locale autour de chaque POI (0 = sample unique)")
    ap.add_argument("--true-key-hex", default=None,
                    help="Optionnel: clé vraie (hex) pour valider les ranks")
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d["traces"]
    plains = d["plaintexts"][:, :16].astype(np.uint8)

    if args.max_traces is not None:
        traces = traces[:args.max_traces]
        plains = plains[:args.max_traces]

    with open(args.poi_json, "r") as f:
        pj = json.load(f)
    if "bytes" in pj:
        pois = [int(x["poi"]) for x in pj["bytes"]]
    elif "poi_global_per_byte" in pj:
        pois = [int(x) for x in pj["poi_global_per_byte"]]
    else:
        raise ValueError("Impossible de trouver les POI dans --poi-json")
    if len(pois) != 16:
        raise ValueError("Le JSON POI doit contenir 16 POI")

    true_key = parse_key_hex(args.true_key_hex)
    proc = center_and_detrend(traces)
    n_total = proc.shape[0]

    n_list = sorted(set(int(x) for x in args.n_list.split(",") if x.strip()))
    n_list = [n for n in n_list if 20 <= n <= n_total]
    if n_total not in n_list:
        n_list.append(n_total)
    if not n_list:
        raise ValueError("n-list vide après filtrage")

    final_key = []
    final_scores = []
    per_byte_top1 = {b: [] for b in range(16)}
    per_byte_margin = {b: [] for b in range(16)}
    per_byte_true_rank = {b: [] for b in range(16)} if true_key is not None else None

    for b in range(16):
        y_all = proc[:, pois[b]]
        pbyte_all = plains[:, b]
        scores_full = scores_at_poi_window(proc, pbyte_all, pois[b], int(args.poi_half_window))
        order_full = np.argsort(scores_full)[::-1]
        final_key.append(int(order_full[0]))
        final_scores.append(float(scores_full[order_full[0]]))

        for n in n_list:
            scores_n = scores_at_poi_window(proc[:n], pbyte_all[:n], pois[b], int(args.poi_half_window))
            ord_n = np.argsort(scores_n)[::-1]
            k1 = int(ord_n[0])
            k2 = int(ord_n[1])
            per_byte_top1[b].append(k1)
            per_byte_margin[b].append(float(scores_n[k1] - scores_n[k2]))
            if true_key is not None:
                rank = int(np.where(ord_n == int(true_key[b]))[0][0]) + 1
                per_byte_true_rank[b].append(rank)

    stable_counts = []
    median_margins = []
    for i in range(len(n_list)):
        stable = sum(1 for b in range(16) if per_byte_top1[b][i] == final_key[b])
        med_margin = float(np.median([per_byte_margin[b][i] for b in range(16)]))
        stable_counts.append(stable)
        median_margins.append(med_margin)

    rank1_counts = None
    rank5_counts = None
    if true_key is not None:
        rank1_counts = [sum(1 for b in range(16) if per_byte_true_rank[b][i] == 1) for i in range(len(n_list))]
        rank5_counts = [sum(1 for b in range(16) if per_byte_true_rank[b][i] <= 5) for i in range(len(n_list))]

    rows = []
    for b in range(16):
        rows.append(
            {
                "byte": b,
                "poi": int(pois[b]),
                "recovered_key": int(final_key[b]),
                "recovered_key_hex": f"{final_key[b]:02X}",
                "score": float(final_scores[b]),
                "top1_over_n": [int(x) for x in per_byte_top1[b]],
                "margin_over_n": [float(x) for x in per_byte_margin[b]],
            }
        )

    out_json = {
        "npz": args.npz,
        "poi_json": args.poi_json,
        "n_list": [int(x) for x in n_list],
        "recovered_key": [int(x) for x in final_key],
        "recovered_key_hex": "".join(f"{x:02X}" for x in final_key),
        "poi_half_window": int(args.poi_half_window),
        "stable_count_vs_n": stable_counts,
        "median_margin_vs_n": median_margins,
        "bytes": rows,
    }
    if true_key is not None:
        out_json["true_key_hex"] = "".join(f"{x:02X}" for x in true_key.tolist())
        out_json["rank1_count_vs_n"] = rank1_counts
        out_json["rank5_count_vs_n"] = rank5_counts
        out_json["ranks_per_byte_vs_n"] = {f"b{b:02d}": [int(x) for x in per_byte_true_rank[b]] for b in range(16)}

    out_json_path = Path(f"{args.out_prefix}_summary.json")
    out_json_path.write_text(json.dumps(out_json, indent=2))

    n_plots = 3 if true_key is not None else 2
    fig, ax = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 1:
        ax = [ax]

    ax[0].plot(n_list, stable_counts, marker="o")
    ax[0].set_ylim(0, 16.5)
    ax[0].set_ylabel("Bytes stables")
    ax[0].set_title("Convergence unknown-key (stabilite top-1 par byte)")
    ax[0].grid(True, ls="--", lw=0.5)

    ax[1].plot(n_list, median_margins, marker="o", color="tab:orange")
    ax[1].set_ylabel("Median(top1-top2)")
    ax[1].grid(True, ls="--", lw=0.5)

    if true_key is not None:
        ax[2].plot(n_list, rank1_counts, marker="o", label="bytes rank=1")
        ax[2].plot(n_list, rank5_counts, marker="o", label="bytes rank<=5")
        ax[2].set_ylim(0, 16.5)
        ax[2].set_ylabel("Count over 16 bytes")
        ax[2].grid(True, ls="--", lw=0.5)
        ax[2].legend()

    ax[-1].set_xlabel("Number of traces")
    fig.tight_layout()
    out_png_path = Path(f"{args.out_prefix}_convergence.png")
    fig.savefig(out_png_path, dpi=150)

    print("Recovered key (hex):", "".join(f"{x:02X}" for x in final_key))
    print("Recovered key (C):", ",".join(f"0x{x:02X}" for x in final_key))
    for b in range(16):
        print(
            f"byte{b:02d}: key=0x{final_key[b]:02X} poi={pois[b]} "
            f"score={final_scores[b]:.6f}"
        )
    print("n_list=", n_list)
    print("stable_count_vs_n=", stable_counts)
    print("median_margin_vs_n=", [round(x, 6) for x in median_margins])
    if true_key is not None:
        print("rank1_count_vs_n=", rank1_counts)
        print("rank5_count_vs_n=", rank5_counts)
    print("saved:", out_png_path)
    print("saved:", out_json_path)


if __name__ == "__main__":
    main()
