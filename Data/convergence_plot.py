#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)


def preprocess(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def signed_corr(y: np.ndarray, h: np.ndarray) -> float:
    yc = y - y.mean()
    hc = h - h.mean()
    ystd = y.std(ddof=1) + 1e-15
    hstd = h.std(ddof=1) + 1e-15
    return float((hc @ yc) / ((len(y) - 1) * hstd * ystd))


def rank_of_key(scores: np.ndarray, true_key: int) -> int:
    order = np.argsort(np.abs(scores))[::-1]
    return int(np.where(order == true_key)[0][0]) + 1


def main():
    ap = argparse.ArgumentParser(description='Convergence plot for XOR-CPA byte attack')
    ap.add_argument('--npz', required=True)
    ap.add_argument('--byte', type=int, default=0)
    ap.add_argument('--true-key', type=lambda x: int(x, 0), required=True)
    ap.add_argument('--alt-key', type=lambda x: int(x, 0), default=None,
                    help='Optional ambiguous key (e.g. true^0xFF)')
    ap.add_argument('--win-start', type=int, default=100)
    ap.add_argument('--win-end', type=int, default=260)
    ap.add_argument('--poi', type=int, default=-1,
                    help='Global POI sample. -1 = auto from true-key abs corr')
    ap.add_argument('--n-list', default='500,1000,2000,3000,5000,7000,10000')
    ap.add_argument('--out', default='convergence_byte.png')
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d['traces'][:, args.win_start:args.win_end]
    pbyte = d['plaintexts'][:, args.byte].astype(np.uint8)

    traces = preprocess(traces)

    if args.alt_key is None:
        alt_key = args.true_key ^ 0xFF
    else:
        alt_key = args.alt_key

    if args.poi >= 0:
        poi_local = args.poi - args.win_start
    else:
        h_true = HW[np.bitwise_xor(pbyte, args.true_key)]
        tc = traces - traces.mean(axis=0, keepdims=True)
        tstd = traces.std(axis=0, ddof=1) + 1e-15
        hc = h_true - h_true.mean()
        hstd = h_true.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((traces.shape[0] - 1) * hstd * tstd)
        poi_local = int(np.argmax(np.abs(corr)))

    y_all = traces[:, poi_local]

    n_list = [int(x) for x in args.n_list.split(',') if x.strip()]
    n_list = [n for n in n_list if n <= len(y_all)]

    corr_true = []
    corr_alt = []
    rank_true = []
    top_key = []

    for n in n_list:
        y = y_all[:n]
        scores = np.zeros(256, dtype=np.float64)
        for k in range(256):
            hk = HW[np.bitwise_xor(pbyte[:n], k)]
            scores[k] = signed_corr(y, hk)

        corr_true.append(scores[args.true_key])
        corr_alt.append(scores[alt_key])
        rank_true.append(rank_of_key(scores, args.true_key))
        top_key.append(int(np.argsort(np.abs(scores))[::-1][0]))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(n_list, corr_true, marker='o', label=f'true 0x{args.true_key:02x}')
    ax[0].plot(n_list, corr_alt, marker='o', label=f'alt 0x{alt_key:02x}')
    ax[0].axhline(0.0, color='k', lw=0.8)
    ax[0].set_ylabel('Signed corr at POI')
    ax[0].set_title(f'Convergence @ POI={args.win_start + poi_local} (window [{args.win_start}:{args.win_end}])')
    ax[0].legend()

    ax[1].plot(n_list, rank_true, marker='o', color='tab:red')
    ax[1].set_ylabel('Rank(true key)')
    ax[1].set_xlabel('Number of traces')
    ax[1].set_yscale('log')
    ax[1].grid(True, which='both', ls='--', lw=0.5)

    for x, k in zip(n_list, top_key):
        ax[1].annotate(f'0x{k:02x}', (x, rank_true[n_list.index(x)]), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(args.out, dpi=140)

    print(f'POI global: {args.win_start + poi_local}')
    for n, ct, ca, rk, tk in zip(n_list, corr_true, corr_alt, rank_true, top_key):
        print(f'n={n:5d}  corr_true={ct:+.5f}  corr_alt={ca:+.5f}  rank_true={rk:3d}  top=0x{tk:02x}')
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
