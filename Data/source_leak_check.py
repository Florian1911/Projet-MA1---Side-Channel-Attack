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
    return t.astype(np.float32)


def corr_trace(traces: np.ndarray, y: np.ndarray) -> np.ndarray:
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    yc = y.astype(np.float64) - float(np.mean(y))
    ystd = float(np.std(y, ddof=1)) + 1e-15
    return (yc @ tc) / ((traces.shape[0] - 1) * ystd * tstd)


def main():
    ap = argparse.ArgumentParser(description="Leakage source sanity check (no key guess)")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--win-start", type=int, default=0)
    ap.add_argument("--win-end", type=int, default=None)
    ap.add_argument("--out", default="source_leak_check.png")
    args = ap.parse_args()

    d = np.load(args.npz)
    traces = d["traces"].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)

    w0 = int(args.win_start)
    w1 = traces.shape[1] if args.win_end is None else int(args.win_end)
    traces = traces[:, w0:w1]
    traces = center_and_detrend(traces)

    p = plains[:, args.byte].astype(np.uint8)
    hw = np.unpackbits(p[:, None], axis=1).sum(axis=1).astype(np.float64)
    bit0 = (p & 1).astype(np.float64)

    c_hw = corr_trace(traces, hw)
    c_b0 = corr_trace(traces, bit0)

    i_hw = int(np.argmax(np.abs(c_hw)))
    i_b0 = int(np.argmax(np.abs(c_b0)))

    print(f"dataset={args.npz} traces={traces.shape}")
    print(f"HW(pt[{args.byte}]) max |corr|={abs(c_hw[i_hw]):.6f} at sample {w0 + i_hw}")
    print(f"bit0(pt[{args.byte}]) max |corr|={abs(c_b0[i_b0]):.6f} at sample {w0 + i_b0}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(c_hw, lw=0.8)
    ax[0].set_title(f"Correlation with HW(pt[{args.byte}])")
    ax[0].set_ylabel("corr")
    ax[0].axvline(i_hw, color="r", ls="--", lw=0.8)

    ax[1].plot(c_b0, lw=0.8, color="tab:orange")
    ax[1].set_title(f"Correlation with bit0(pt[{args.byte}])")
    ax[1].set_ylabel("corr")
    ax[1].set_xlabel(f"sample (local window [{w0}:{w1}])")
    ax[1].axvline(i_b0, color="r", ls="--", lw=0.8)

    plt.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved plot: {args.out}")


if __name__ == "__main__":
    main()
