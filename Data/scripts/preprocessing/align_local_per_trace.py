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


def snr_curve(traces: np.ndarray, classes: np.ndarray, n_classes: int = 9) -> np.ndarray:
    means = np.zeros((n_classes, traces.shape[1]), dtype=np.float64)
    vars_ = np.zeros((n_classes, traces.shape[1]), dtype=np.float64)
    counts = np.zeros((n_classes,), dtype=np.int64)
    for c in range(n_classes):
        idx = (classes == c)
        nc = int(idx.sum())
        if nc == 0:
            continue
        counts[c] = nc
        tc = traces[idx]
        means[c] = tc.mean(axis=0)
        vars_[c] = tc.var(axis=0)
    valid = counts > 0
    return means[valid].var(axis=0) / (vars_[valid].mean(axis=0) + 1e-12)


def estimate_poi_from_snr(traces: np.ndarray, plains: np.ndarray, key_byte: int, byte_idx: int) -> int:
    proc = center_and_detrend(traces)
    hw = HW[AES_SBOX[np.bitwise_xor(plains[:, byte_idx], key_byte)]]
    snr = snr_curve(proc, hw, n_classes=9)
    return int(np.argmax(snr))


def estimate_lags_local_ref(
    proc: np.ndarray, center: int, window: int, max_shift: int, ref_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    n, tlen = proc.shape
    half = window // 2
    lo = max(0, center - half)
    hi = min(tlen, center + half)
    if hi - lo < 40:
        raise ValueError("window too small after clipping")

    seg0 = proc[:, lo:hi]
    if ref_kind == "mean":
        ref = seg0.mean(axis=0)
    else:
        ref = np.median(seg0, axis=0)
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref) + 1e-12

    pad = max_shift
    padded = np.pad(proc, ((0, 0), (pad, pad)), mode="edge")
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    scores = np.empty((n, lags.size), dtype=np.float32)

    base_lo = lo + pad
    base_hi = hi + pad

    for j, lag in enumerate(lags):
        seg = padded[:, base_lo + lag:base_hi + lag]
        seg = seg - seg.mean(axis=1, keepdims=True)
        num = (seg * ref[None, :]).sum(axis=1)
        den = (np.linalg.norm(seg, axis=1) + 1e-12) * ref_norm
        scores[:, j] = (num / den).astype(np.float32)

    best_idx = np.argmax(scores, axis=1)
    best_lags = lags[best_idx]
    best_scores = scores[np.arange(n), best_idx]
    return best_lags, best_scores


def estimate_lags_local_hw_template(
    proc: np.ndarray,
    labels_hw: np.ndarray,
    center: int,
    window: int,
    max_shift: int,
    ref_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    n, tlen = proc.shape
    half = window // 2
    lo = max(0, center - half)
    hi = min(tlen, center + half)
    if hi - lo < 40:
        raise ValueError("window too small after clipping")

    wlen = hi - lo
    templates = np.zeros((9, wlen), dtype=np.float32)
    for c in range(9):
        tc = proc[labels_hw == c, lo:hi]
        if tc.shape[0] == 0:
            continue
        if ref_kind == "mean":
            templates[c] = tc.mean(axis=0)
        else:
            templates[c] = np.median(tc, axis=0)
    ref = templates[labels_hw].astype(np.float32)
    ref = ref - ref.mean(axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref, axis=1) + 1e-12

    pad = max_shift
    padded = np.pad(proc, ((0, 0), (pad, pad)), mode="edge")
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    scores = np.empty((n, lags.size), dtype=np.float32)

    base_lo = lo + pad
    base_hi = hi + pad
    for j, lag in enumerate(lags):
        seg = padded[:, base_lo + lag:base_hi + lag]
        seg = seg - seg.mean(axis=1, keepdims=True)
        num = (seg * ref).sum(axis=1)
        den = (np.linalg.norm(seg, axis=1) + 1e-12) * ref_norm
        scores[:, j] = (num / den).astype(np.float32)

    best_idx = np.argmax(scores, axis=1)
    best_lags = lags[best_idx]
    best_scores = scores[np.arange(n), best_idx]
    return best_lags, best_scores


def apply_integer_shifts(traces: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    n, tlen = traces.shape
    out = np.empty_like(traces)
    for s in np.unique(shifts):
        idx = np.where(shifts == s)[0]
        si = int(s)
        if si == 0:
            out[idx] = traces[idx]
        elif si > 0:
            out[idx, si:] = traces[idx, :tlen - si]
            out[idx, :si] = traces[idx, 0:1]
        else:
            k = -si
            out[idx, :tlen - k] = traces[idx, k:]
            out[idx, tlen - k:] = traces[idx, -1:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Local per-trace alignment by correlation on anchor window")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--center", type=int, default=-1, help="-1 => auto from SNR on byte/key")
    ap.add_argument("--window", type=int, default=220)
    ap.add_argument("--max-shift", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--key", type=lambda x: int(x, 0), default=0x2B)
    ap.add_argument("--ref", choices=["median", "mean"], default="median")
    ap.add_argument("--mode", choices=["ref", "hw_template"], default="hw_template")
    args = ap.parse_args()

    d = np.load(args.inp)
    traces = d["traces"].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8) if "plaintexts" in d.files else None

    if traces.ndim != 2 or traces.shape[0] == 0:
        raise ValueError(
            f"Dataset vide ou invalide: traces shape={getattr(traces, 'shape', None)}. "
            "Vérifie l'acquisition (aucun trigger reçu ?)."
        )

    center = int(args.center)
    if center < 0:
        if plains is None:
            raise ValueError("--center=-1 requires plaintexts in dataset")
        center = estimate_poi_from_snr(traces, plains, args.key, args.byte)

    cur = traces.copy()
    cum_shift = np.zeros((cur.shape[0],), dtype=np.int32)
    per_iter = []

    labels_hw = None
    if plains is not None:
        labels_hw = HW[AES_SBOX[np.bitwise_xor(plains[:, args.byte], args.key)]]

    for it in range(args.iters):
        proc = center_and_detrend(cur)
        if args.mode == "hw_template":
            if labels_hw is None:
                raise ValueError("--mode=hw_template requires plaintexts in dataset")
            lags, scores = estimate_lags_local_hw_template(
                proc, labels_hw, center, args.window, args.max_shift, args.ref
            )
        else:
            lags, scores = estimate_lags_local_ref(
                proc, center, args.window, args.max_shift, args.ref
            )
        # If lag is +k, moving trace by -k aligns it to the reference.
        delta = (-lags).astype(np.int32)
        cur = apply_integer_shifts(cur, delta)
        cum_shift += delta
        per_iter.append(
            {
                "iter": it + 1,
                "lag_mean_abs": float(np.mean(np.abs(lags))),
                "lag_p95_abs": float(np.quantile(np.abs(lags), 0.95)),
                "score_mean": float(np.mean(scores)),
                "delta_mean_abs": float(np.mean(np.abs(delta))),
            }
        )

    save = {}
    for k in d.files:
        if k == "traces":
            save["traces"] = cur
        else:
            save[k] = d[k]
    save["local_align_center"] = np.int32(center)
    save["local_align_window"] = np.int32(args.window)
    save["local_align_max_shift"] = np.int32(args.max_shift)
    save["local_align_iters"] = np.int32(args.iters)
    save["local_align_shift_per_trace"] = cum_shift.astype(np.int16)
    np.savez(args.out, **save)

    out_json = Path(args.out).with_suffix(".json")
    with out_json.open("w") as f:
        json.dump(
            {
                "input": args.inp,
                "output": args.out,
                "center": int(center),
                "window": int(args.window),
                "max_shift": int(args.max_shift),
                "iters": int(args.iters),
                "ref": args.ref,
                "mode": args.mode,
                "shift_mean_abs": float(np.mean(np.abs(cum_shift))),
                "shift_p95_abs": float(np.quantile(np.abs(cum_shift), 0.95)),
                "iter_stats": per_iter,
            },
            f,
            indent=2,
        )
    print(f"saved: {args.out} | traces={cur.shape}")
    print(f"saved: {out_json}")
    print(
        "final shift stats: "
        f"mean_abs={float(np.mean(np.abs(cum_shift))):.3f}, "
        f"p95_abs={float(np.quantile(np.abs(cum_shift), 0.95)):.3f}"
    )


if __name__ == "__main__":
    main()
