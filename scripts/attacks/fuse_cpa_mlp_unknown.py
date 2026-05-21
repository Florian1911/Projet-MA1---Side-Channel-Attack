#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

AES_SBOX = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
], dtype=np.uint8)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_key_hex(s: str) -> np.ndarray:
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "")
    if len(t) != 32:
        raise ValueError("--true-key-hex must contain 32 hex chars")
    return np.array([int(t[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(z: np.ndarray) -> np.ndarray:
    zz = z - z.max(axis=1, keepdims=True)
    ee = np.exp(zz)
    return ee / (ee.sum(axis=1, keepdims=True) + 1e-12)


def infer_probs(x: np.ndarray, model: dict, batch_size: int = 4096) -> np.ndarray:
    w1, b1 = model["w1"], model["b1"]
    w2, b2 = model["w2"], model["b2"]
    w3, b3 = model["w3"], model["b3"]
    n = x.shape[0]
    out = np.zeros((n, b3.shape[0]), dtype=np.float32)
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        z1 = xb @ w1 + b1
        a1 = relu(z1)
        z2 = a1 @ w2 + b2
        a2 = relu(z2)
        z3 = a2 @ w3 + b3
        out[i:i + batch_size] = softmax(z3).astype(np.float32)
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    m = float(v.mean())
    s = float(v.std()) + 1e-12
    return (v - m) / s


def cpa_scores_window(proc: np.ndarray, pbyte: np.ndarray, poi: int, half_window: int) -> np.ndarray:
    n, tlen = proc.shape
    a = max(0, poi - half_window)
    b = min(tlen, poi + half_window + 1)
    tw = proc[:, a:b]
    tc = tw - tw.mean(axis=0, keepdims=True)
    tstd = tw.std(axis=0, ddof=1) + 1e-15
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h = HW[AES_SBOX[np.bitwise_xor(pbyte, k)]].astype(np.float64)
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((n - 1) * hstd * tstd)
        scores[k] = float(np.max(np.abs(corr)))
    return scores


def mlp_key_scores(model_path: Path, attack_byte_path: Path, byte_idx: int) -> tuple[np.ndarray, str]:
    m = np.load(model_path)
    d = np.load(attack_byte_path)
    traces = d["traces"].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)
    pbyte = plains[:, byte_idx]

    x = center_and_detrend(traces)
    poi_idx = m["poi_idx"].astype(np.int32)
    mu = m["mu"].astype(np.float32)
    sigma = m["sigma"].astype(np.float32)
    x = x[:, poi_idx]
    x = (x - mu) / (sigma + 1e-6)
    probs = np.clip(infer_probs(x, m), 1e-12, 1.0)

    n_classes = int(probs.shape[1])
    if n_classes == 9:
        label_mode = "hw"
    elif n_classes == 256:
        label_mode = "sbox"
    else:
        raise ValueError(f"unsupported n_classes={n_classes} in {model_path}")

    keys = np.arange(256, dtype=np.uint8)
    sbox_map = AES_SBOX[np.bitwise_xor(pbyte[:, None], keys[None, :])]
    cls = HW[sbox_map] if label_mode == "hw" else sbox_map
    ll = np.log(probs[np.arange(probs.shape[0])[:, None], cls]).sum(axis=0)
    return ll.astype(np.float64), label_mode


def load_pois(path: str) -> list[int]:
    j = json.load(open(path, "r"))
    if "bytes" in j:
        return [int(row["poi"]) for row in j["bytes"]]
    if "poi_global_per_byte" in j:
        return [int(x) for x in j["poi_global_per_byte"][:16]]
    raise ValueError("POI json missing bytes/poi_global_per_byte")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse CPA + MLP key scores for unknown-key attack")
    ap.add_argument("--unknown-npz", required=True)
    ap.add_argument("--poi-json", required=True)
    ap.add_argument("--mlp-campaign-dir", required=True, help="e.g. campaign_multisession_mlp16_full")
    ap.add_argument("--poi-half-window", type=int, default=40)
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0", help="alpha for fused=z(mlp)*a + z(cpa)*(1-a)")
    ap.add_argument("--true-key-hex", default="")
    ap.add_argument("--out-prefix", default="fuse_cpa_mlp_unknown")
    args = ap.parse_args()

    d = np.load(args.unknown_npz)
    traces = d["traces"].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)
    proc = center_and_detrend(traces)
    pois = load_pois(args.poi_json)
    if len(pois) != 16:
        raise ValueError("need 16 POIs")

    true = parse_key_hex(args.true_key_hex) if args.true_key_hex.strip() else None
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out = {"alphas": alphas, "bytes": [], "metrics": {}}

    cpa_all = []
    mlp_all = []
    label_modes = []
    campaign = Path(args.mlp_campaign_dir)
    for b in range(16):
        cpa = cpa_scores_window(proc, plains[:, b], pois[b], int(args.poi_half_window))
        model_path = campaign / f"byte_{b:02d}" / f"model_b{b:02d}.npz"
        attack_path = campaign / f"byte_{b:02d}" / f"attack_b{b:02d}.npz"
        mlp, mode = mlp_key_scores(model_path, attack_path, b)
        cpa_all.append(cpa)
        mlp_all.append(mlp)
        label_modes.append(mode)

    for a in alphas:
        rec = []
        rank1 = 0
        rank5 = 0
        ranks = []
        for b in range(16):
            sc = (1.0 - a) * zscore(cpa_all[b]) + a * zscore(mlp_all[b])
            ordk = np.argsort(sc)[::-1]
            k = int(ordk[0])
            rec.append(k)
            if true is not None:
                r = int(np.where(ordk == int(true[b]))[0][0]) + 1
                ranks.append(r)
                if r == 1:
                    rank1 += 1
                if r <= 5:
                    rank5 += 1
        out["metrics"][f"{a:.3f}"] = {
            "recovered_key_hex": "".join(f"{x:02X}" for x in rec),
            "rank1_count": rank1 if true is not None else None,
            "rank5_count": rank5 if true is not None else None,
            "mean_rank": float(np.mean(ranks)) if true is not None else None,
            "median_rank": float(np.median(ranks)) if true is not None else None,
            "ranks": ranks if true is not None else None,
        }

    for b in range(16):
        out["bytes"].append(
            {
                "byte": b,
                "poi": int(pois[b]),
                "mlp_label_mode": label_modes[b],
                "cpa_top1": int(np.argmax(cpa_all[b])),
                "mlp_top1": int(np.argmax(mlp_all[b])),
            }
        )

    out_path = Path(f"{args.out_prefix}_summary.json")
    out_path.write_text(json.dumps(out, indent=2))
    print("saved:", out_path)
    for a in alphas:
        m = out["metrics"][f"{a:.3f}"]
        print(
            f"alpha={a:.3f} key={m['recovered_key_hex']}"
            + (f" | rank1={m['rank1_count']}/16 rank5={m['rank5_count']}/16 mean_rank={m['mean_rank']:.1f}" if true is not None else "")
        )


if __name__ == "__main__":
    main()
