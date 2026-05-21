#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    slopes = (t @ x) / np.dot(x, x)
    t = t - np.outer(slopes, x)
    return t


def main():
    ap = argparse.ArgumentParser(description='Diff-of-means leakage check (2 classes)')
    ap.add_argument('--npz', required=True)
    ap.add_argument('--byte', type=int, default=0)
    ap.add_argument('--threshold', type=lambda x: int(x, 0), default=0x80,
                    help='classe A: pt<byte><threshold, classe B sinon')
    ap.add_argument('--win-start', type=int, default=0)
    ap.add_argument('--win-end', type=int, default=None)
    ap.add_argument('--out', default='diff_of_means.png')
    args = ap.parse_args()

    d = np.load(args.npz)
    tr = d['traces'].astype(np.float32)
    pt = d['plaintexts'][:, args.byte].astype(np.uint8)

    w0 = int(args.win_start)
    w1 = tr.shape[1] if args.win_end is None else int(args.win_end)
    tr = center_and_detrend(tr[:, w0:w1])

    a = tr[pt < args.threshold]
    b = tr[pt >= args.threshold]
    if len(a) < 50 or len(b) < 50:
        raise RuntimeError(f'classes trop petites: A={len(a)}, B={len(b)}')

    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    dom = mu_a - mu_b
    i = int(np.argmax(np.abs(dom)))

    print(f'dataset={args.npz} traces={tr.shape}')
    print(f'class A (<0x{args.threshold:02x}): {len(a)} traces')
    print(f'class B (>=0x{args.threshold:02x}): {len(b)} traces')
    print(f'max |diff of means|={abs(dom[i]):.6f} at sample {w0+i}')

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(mu_a, lw=0.8, label='mean A')
    ax[0].plot(mu_b, lw=0.8, label='mean B')
    ax[0].legend()
    ax[0].set_title('Class means')

    ax[1].plot(dom, lw=0.8, color='tab:red')
    ax[1].axvline(i, color='k', ls='--', lw=0.8)
    ax[1].set_title('Difference of means (A-B)')

    std = tr.std(axis=0)
    ax[2].plot(std, lw=0.8, color='tab:orange')
    ax[2].set_title('Global std')
    ax[2].set_xlabel(f'sample (local window [{w0}:{w1}])')

    plt.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'saved plot: {args.out}')


if __name__ == '__main__':
    main()
