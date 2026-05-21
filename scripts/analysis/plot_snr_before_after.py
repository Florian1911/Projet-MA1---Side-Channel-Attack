#!/usr/bin/env python3
"""Trace une figure SNR avant/apres preprocessing pour des datasets AES.

Exemples:
  python plot_snr_before_after.py --raw unknown_B_5k_raw.npz --after unknown_B_5k_aligned.npz --out snr_before_after.png
  python plot_snr_before_after.py --raw dataset.npz --after-mode center_detrend --out snr_before_after.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def center_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    slopes = (t @ x) / float(np.dot(x, x))
    return (t - np.outer(slopes, x)).astype(np.float32)


def snr_curve(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    labels = labels.astype(np.int16, copy=False)
    classes = np.unique(labels)
    means = []
    variances = []
    for c in classes:
        x = t[labels == c]
        if len(x) < 2:
            continue
        means.append(x.mean(axis=0))
        variances.append(x.var(axis=0))
    if not means:
        raise ValueError("Pas assez de classes pour calculer le SNR")
    mean_var = np.var(np.vstack(means), axis=0)
    noise_var = np.mean(np.vstack(variances), axis=0)
    return mean_var / np.maximum(noise_var, 1e-12)


def load_npz(path: str, traces_key: str, n_traces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    d = np.load(path, allow_pickle=False)
    traces = d[traces_key].astype(np.float32)
    plaintexts = d["plaintexts"].astype(np.uint8)
    key = d["key"].astype(np.uint8) if "key" in d.files else None
    if n_traces > 0:
        traces = traces[:n_traces]
        plaintexts = plaintexts[:n_traces]
    return traces, plaintexts, key


def labels_from_plaintexts(plaintexts: np.ndarray, key: np.ndarray | None, byte: int, key_byte: int | None, model: str) -> np.ndarray:
    if key_byte is None:
        if key is None or len(key) <= byte:
            raise ValueError("Cle absente du dataset: passer --key-byte 0x..")
        key_byte = int(key[byte])
    sbox = AES_SBOX[np.bitwise_xor(plaintexts[:, byte], np.uint8(key_byte))]
    if model == "hw":
        return HW[sbox].astype(np.uint8)
    return sbox.astype(np.uint8)


def apply_mode(traces: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return traces
    if mode == "center":
        return (traces - traces.mean(axis=1, keepdims=True)).astype(np.float32)
    if mode == "center_detrend":
        return center_detrend(traces)
    raise ValueError(f"mode inconnu: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure SNR avant/apres preprocessing")
    ap.add_argument("--raw", required=True, help="Dataset avant preprocessing (.npz)")
    ap.add_argument("--after", default="", help="Dataset apres preprocessing/alignment (.npz). Si absent, utilise --after-mode sur --raw")
    ap.add_argument("--raw-traces-key", default="traces")
    ap.add_argument("--after-traces-key", default="traces")
    ap.add_argument("--before-mode", choices=["none", "center", "center_detrend"], default="none")
    ap.add_argument("--after-mode", choices=["none", "center", "center_detrend"], default="center_detrend")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--key-byte", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--label-model", choices=["hw", "sbox"], default="hw")
    ap.add_argument("--n-traces", type=int, default=0)
    ap.add_argument("--time-scale", choices=["sample", "us"], default="sample")
    ap.add_argument("--dt-ns", type=float, default=None, help="Pas temporel si axe en us")
    ap.add_argument("--out", default="snr_before_after.png")
    args = ap.parse_args()

    raw_traces, raw_pt, raw_key = load_npz(args.raw, args.raw_traces_key, args.n_traces)
    if args.after:
        after_traces, after_pt, after_key = load_npz(args.after, args.after_traces_key, args.n_traces)
    else:
        after_traces, after_pt, after_key = raw_traces.copy(), raw_pt, raw_key

    n = min(len(raw_traces), len(after_traces), len(raw_pt), len(after_pt))
    raw_traces = raw_traces[:n]
    after_traces = after_traces[:n]
    raw_pt = raw_pt[:n]

    labels = labels_from_plaintexts(raw_pt, raw_key if raw_key is not None else after_key, args.byte, args.key_byte, args.label_model)
    before = apply_mode(raw_traces, args.before_mode)
    after = apply_mode(after_traces, args.after_mode if not args.after else "none")
    before_snr = snr_curve(before, labels)
    after_snr = snr_curve(after, labels)

    if args.time_scale == "us":
        if args.dt_ns is None:
            raise ValueError("--dt-ns requis avec --time-scale us")
        x_before = np.arange(before_snr.size) * args.dt_ns / 1000.0
        x_after = np.arange(after_snr.size) * args.dt_ns / 1000.0
        xlabel = "Time (us)"
    else:
        x_before = np.arange(before_snr.size)
        x_after = np.arange(after_snr.size)
        xlabel = "Sample"

    bmax_i = int(np.argmax(before_snr))
    amax_i = int(np.argmax(after_snr))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    axes[0].plot(x_before, before_snr, color="#3b5b92", linewidth=1.0)
    axes[0].scatter([x_before[bmax_i]], [before_snr[bmax_i]], color="#1f2d4f", s=18, zorder=3)
    axes[0].set_title(f"Before preprocessing - max SNR {before_snr[bmax_i]:.4g}")
    axes[0].set_ylabel("SNR")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_after, after_snr, color="#b24a3b", linewidth=1.0)
    axes[1].scatter([x_after[amax_i]], [after_snr[amax_i]], color="#6e241c", s=18, zorder=3)
    axes[1].set_title(f"After preprocessing - max SNR {after_snr[amax_i]:.4g}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("SNR")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"AES byte {args.byte} SNR - {args.label_model.upper()} model - n={n}", y=0.98)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=180)
    print(f"[OK] figure: {out}")
    print(f"avant: max={before_snr[bmax_i]:.6g} sample={bmax_i}")
    print(f"apres: max={after_snr[amax_i]:.6g} sample={amax_i}")


if __name__ == "__main__":
    main()
