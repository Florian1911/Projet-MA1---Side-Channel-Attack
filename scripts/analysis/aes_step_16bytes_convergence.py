#!/usr/bin/env python3
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

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


def preprocess(tr):
    t = tr.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def cpa_corr_curve(tr, pbyte, key):
    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    h = HW[SBOX[np.bitwise_xor(pbyte, key)]]
    hc = h - h.mean()
    hstd = h.std(ddof=1) + 1e-15
    return (hc @ tc) / ((tr.shape[0] - 1) * hstd * tstd)


def rank_at_poi(y, pbyte, true_key):
    yc = y - y.mean()
    ystd = y.std(ddof=1) + 1e-15
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h = HW[SBOX[np.bitwise_xor(pbyte, k)]]
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        scores[k] = abs((hc @ yc) / ((len(y) - 1) * hstd * ystd))
    order = np.argsort(scores)[::-1]
    rank = int(np.where(order == true_key)[0][0]) + 1
    top = int(order[0])
    return rank, top, float(scores[true_key])


def main():
    ap = argparse.ArgumentParser(description="Convergence plots for AES-step 16-byte target")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--win-start", type=int, default=50)
    ap.add_argument("--win-end", type=int, default=2600)
    ap.add_argument("--n-list", default="500,1000,1500,2000,3000,4000,5000,7000,10000,15000,20000")
    ap.add_argument("--out-prefix", default="aes_step_16b_20k")
    args = ap.parse_args()

    d = np.load(args.npz)
    tr = preprocess(d["traces"][:, args.win_start:args.win_end])
    pt = d["plaintexts"][:, :16].astype(np.uint8)
    key = d["key"][:16].astype(np.uint8)

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    n_list = [n for n in n_list if 10 <= n <= tr.shape[0]]
    if not n_list:
        raise ValueError("n-list vide après filtrage")

    pois = []
    for b in range(16):
        corr = cpa_corr_curve(tr, pt[:, b], int(key[b]))
        pois.append(int(np.argmax(np.abs(corr))))

    ranks = {b: [] for b in range(16)}
    tops = {b: [] for b in range(16)}
    true_scores = {b: [] for b in range(16)}
    for n in n_list:
        for b in range(16):
            r, top, ts = rank_at_poi(tr[:n, pois[b]], pt[:n, b], int(key[b]))
            ranks[b].append(r)
            tops[b].append(top)
            true_scores[b].append(ts)

    # Figure 1: small multiples (4x4) rank convergence by byte.
    fig1, axes = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    for b in range(16):
        ax = axes[b // 4, b % 4]
        ax.plot(n_list, ranks[b], marker="o", ms=3)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", lw=0.4)
        ax.set_title(f"b{b:02d} k=0x{int(key[b]):02x} poi={args.win_start+pois[b]}", fontsize=9)
    fig1.suptitle("AES-step 16-byte convergence (rank per byte)", fontsize=13)
    fig1.text(0.5, 0.04, "Number of traces", ha="center")
    fig1.text(0.04, 0.5, "Rank(true key)", va="center", rotation="vertical")
    fig1.tight_layout(rect=[0.05, 0.05, 1.0, 0.96])
    out_grid = f"{args.out_prefix}_convergence_grid.png"
    fig1.savefig(out_grid, dpi=150)

    # Figure 2: global summary.
    rank1_counts = [sum(1 for b in range(16) if ranks[b][i] == 1) for i in range(len(n_list))]
    rank5_counts = [sum(1 for b in range(16) if ranks[b][i] <= 5) for i in range(len(n_list))]
    med_ranks = [float(np.median([ranks[b][i] for b in range(16)])) for i in range(len(n_list))]

    fig2, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(n_list, rank1_counts, marker="o", label="bytes with rank=1")
    ax[0].plot(n_list, rank5_counts, marker="o", label="bytes with rank<=5")
    ax[0].set_ylabel("Count over 16 bytes")
    ax[0].set_ylim(0, 16.5)
    ax[0].grid(True, ls="--", lw=0.5)
    ax[0].legend()

    ax[1].plot(n_list, med_ranks, marker="o", color="tab:red")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Median rank (16 bytes)")
    ax[1].set_xlabel("Number of traces")
    ax[1].grid(True, which="both", ls="--", lw=0.5)
    fig2.suptitle("AES-step 16-byte convergence summary")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out_summary = f"{args.out_prefix}_convergence_summary.png"
    fig2.savefig(out_summary, dpi=150)

    summary = {
        "npz": args.npz,
        "window": [int(args.win_start), int(args.win_end)],
        "n_list": n_list,
        "poi_global_per_byte": [int(args.win_start + p) for p in pois],
        "rank1_counts": rank1_counts,
        "rank5_counts": rank5_counts,
        "median_rank": med_ranks,
        "ranks_per_byte": {f"b{b:02d}": [int(x) for x in ranks[b]] for b in range(16)},
    }
    out_json = f"{args.out_prefix}_convergence_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    for b in range(16):
        print(f"byte{b:02d} poi={args.win_start+pois[b]} rank@Nmax={ranks[b][-1]:3d} top@Nmax=0x{tops[b][-1]:02x}")
    print(f"n_list={n_list}")
    print(f"rank1_counts={rank1_counts}")
    print(f"rank5_counts={rank5_counts}")
    print(f"saved: {out_grid}")
    print(f"saved: {out_summary}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
