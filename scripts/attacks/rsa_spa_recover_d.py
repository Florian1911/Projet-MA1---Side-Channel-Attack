#!/usr/bin/env python3
import argparse
import json
import numpy as np


def moving_avg(x, w):
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, k, mode="same")


def detrend_linear(y):
    x = np.linspace(-1.0, 1.0, y.size, dtype=np.float64)
    a = np.dot(y, x) / (np.dot(x, x) + 1e-15)
    return y - a * x


def pick_activity_window(env, q=97.0, pad=40):
    thr = np.percentile(env, q)
    idx = np.flatnonzero(env >= thr)
    if idx.size == 0:
        return 0, env.size
    s = max(0, int(idx[0]) - pad)
    e = min(env.size, int(idx[-1]) + pad + 1)
    if e - s < 200:
        mid = (s + e) // 2
        s = max(0, mid - 100)
        e = min(env.size, mid + 100)
    return s, e


def two_means_threshold(v, iters=20):
    a = float(np.min(v))
    b = float(np.max(v))
    if abs(b - a) < 1e-15:
        return a
    for _ in range(iters):
        da = np.abs(v - a)
        db = np.abs(v - b)
        ma = v[da <= db]
        mb = v[da > db]
        if ma.size:
            a = float(np.mean(ma))
        if mb.size:
            b = float(np.mean(mb))
    return 0.5 * (a + b)


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def main():
    ap = argparse.ArgumentParser(description="Blind SPA recovery for naive RSA square-and-multiply traces")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--trace-key", default="traces", help="array key in NPZ (default: traces)")
    ap.add_argument("--n-traces", type=int, default=0, help="0 => all")
    ap.add_argument("--bitlen", type=int, default=23, help="private exponent bit length (default 23 for demo key)")
    ap.add_argument("--start", type=int, default=-1, help="manual start sample")
    ap.add_argument("--end", type=int, default=-1, help="manual end sample")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--out", default="rsa_spa_blind_result.json")
    args = ap.parse_args()

    d = np.load(args.npz)
    if args.trace_key not in d.files:
        raise SystemExit(f"trace key '{args.trace_key}' not found. available={d.files}")

    tr = d[args.trace_key].astype(np.float64)
    if tr.ndim != 2:
        raise SystemExit("traces must be 2D")
    if args.n_traces and args.n_traces > 0:
        tr = tr[:args.n_traces]

    # Robust representative trace
    m = np.median(tr, axis=0)
    m = m - np.mean(m)
    m = detrend_linear(m)

    # Envelope from derivative energy
    der = np.diff(m, prepend=m[0])
    env = moving_avg(np.abs(der), args.smooth)

    if args.start >= 0 and args.end > args.start:
        s, e = int(args.start), int(args.end)
    else:
        s, e = pick_activity_window(env, q=97.0, pad=60)

    seg = m[s:e]
    env_seg = env[s:e]
    if seg.size < args.bitlen * 8:
        raise SystemExit(f"window too short for bitlen={args.bitlen}: len={seg.size}")

    edges = np.linspace(0, seg.size, args.bitlen + 1).astype(int)
    feats = []
    for i in range(args.bitlen):
        a, b = edges[i], edges[i + 1]
        ch = env_seg[a:b]
        if ch.size < 4:
            feats.append(0.0)
            continue
        cut = int(0.6 * ch.size)
        e1 = float(np.mean(ch[:cut]))
        e2 = float(np.mean(ch[cut:]))
        feats.append(e2 - e1)
    feats = np.asarray(feats, dtype=np.float64)

    thr = two_means_threshold(feats)
    bits = (feats > thr).astype(int)
    bits[0] = 1  # MSB of exponent is always 1 in canonical form

    d_est = bits_to_int(bits)
    bits_str = "".join(str(int(b)) for b in bits)

    out = {
        "npz": args.npz,
        "trace_key": args.trace_key,
        "n_traces_used": int(tr.shape[0]),
        "window": {"start": int(s), "end": int(e), "len": int(e - s)},
        "bitlen": int(args.bitlen),
        "threshold": float(thr),
        "features": [float(x) for x in feats],
        "bits_msb_to_lsb": bits_str,
        "d_est_int": int(d_est),
        "d_est_hex": f"0x{d_est:X}",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"saved: {args.out}")
    print("window:", s, e, "len=", e - s)
    print("bits :", bits_str)
    print("d_est:", d_est, f"(0x{d_est:X})")


if __name__ == "__main__":
    main()
