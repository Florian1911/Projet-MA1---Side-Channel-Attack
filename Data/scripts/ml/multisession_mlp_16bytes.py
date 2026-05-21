#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
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


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_bytes(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(16))
    out = []
    for t in raw.split(","):
        t = t.strip()
        if not t:
            continue
        v = int(t)
        if not (0 <= v < 16):
            raise ValueError(f"invalid byte index: {v}")
        out.append(v)
    out = sorted(set(out))
    if not out:
        raise ValueError("empty byte list")
    return out


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_poi_centers(path: str | None) -> dict[int, int]:
    if not path:
        return {}
    with open(path, "r") as f:
        j = json.load(f)
    centers: dict[int, int] = {}
    if "bytes" in j:
        for row in j["bytes"]:
            centers[int(row["byte"])] = int(row["poi"])
    elif "poi_global_per_byte" in j:
        arr = j["poi_global_per_byte"]
        for b in range(min(16, len(arr))):
            centers[b] = int(arr[b])
    return centers


def build_profile_byte_npz(
    profile_paths: list[str],
    byte_idx: int,
    out_path: Path,
    center: int | None,
    window: int,
    label_mode: str,
) -> tuple[int, int]:
    traces_all = []
    labels_all = []
    for p in profile_paths:
        d = np.load(p)
        if "traces" not in d.files or "plaintexts" not in d.files or "key" not in d.files:
            raise ValueError(f"{p}: missing traces/plaintexts/key")
        tr = d["traces"].astype(np.float32)
        pt = d["plaintexts"][:, byte_idx].astype(np.uint8)
        key_b = int(d["key"][byte_idx])
        if center is not None:
            half = window // 2
            a = max(0, center - half)
            b = min(tr.shape[1], center + half)
            tr = tr[:, a:b]
        inter = AES_SBOX[np.bitwise_xor(pt, np.uint8(key_b))]
        if label_mode == "hw":
            lab = HW[inter].astype(np.int64)
        elif label_mode == "sbox":
            lab = inter.astype(np.int64)
        else:
            raise ValueError(f"unsupported label_mode: {label_mode}")
        traces_all.append(tr)
        labels_all.append(lab)
    x = np.concatenate(traces_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    np.savez(out_path, traces=x, labels=y)
    return x.shape[0], x.shape[1]


def build_attack_byte_npz(
    attack_path: str,
    byte_idx: int,
    out_path: Path,
    center: int | None,
    window: int,
) -> int:
    d = np.load(attack_path)
    tr = d["traces"].astype(np.float32)
    pt = d["plaintexts"].astype(np.uint8)
    key = d["key"] if "key" in d.files else np.array([], dtype=np.uint8)
    if center is not None:
        half = window // 2
        a = max(0, center - half)
        b = min(tr.shape[1], center + half)
        tr = tr[:, a:b]
    np.savez(out_path, traces=tr, plaintexts=pt, key=key)
    return tr.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-session profiling MLP for 16-byte unknown-key attack")
    ap.add_argument("--profile-datasets", required=True, help="Comma-separated known datasets (.npz)")
    ap.add_argument("--attack-dataset", required=True, help="Unknown dataset (.npz)")
    ap.add_argument("--outdir", default="campaign_multisession_mlp16")
    ap.add_argument("--bytes", default="all", help="all or comma-separated indices, e.g. 0,1,2")
    ap.add_argument("--poi-json", default="", help="Optional POI JSON (centers per byte)")
    ap.add_argument("--window", type=int, default=1200, help="Crop length around center if poi-json provided")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--poi", type=int, default=300)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--label-mode", choices=["hw", "sbox"], default="hw")
    ap.add_argument("--true-key-hex", default="", help="Optional for evaluation")
    args = ap.parse_args()

    profile_paths = parse_list(args.profile_datasets)
    byte_list = parse_bytes(args.bytes)
    centers = load_poi_centers(args.poi_json if args.poi_json else None)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    true_key = None
    if args.true_key_hex.strip():
        t = args.true_key_hex.strip().replace(" ", "").replace(":", "").replace(",", "")
        if len(t) != 32:
            raise ValueError("--true-key-hex must be 32 hex chars")
        true_key = [int(t[i:i + 2], 16) for i in range(0, 32, 2)]

    summary = {"bytes": []}
    for b in byte_list:
        bdir = outdir / f"byte_{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        center = centers.get(b, None)

        profile_b = bdir / f"profile_b{b:02d}.npz"
        n_prof, tlen = build_profile_byte_npz(
            profile_paths, b, profile_b, center, args.window, args.label_mode
        )
        attack_b = bdir / f"attack_b{b:02d}.npz"
        n_att = build_attack_byte_npz(args.attack_dataset, b, attack_b, center, args.window)
        print(f"[BYTE {b:02d}] profile={n_prof} attack={n_att} tlen={tlen} center={center}", flush=True)

        model_b = bdir / f"model_b{b:02d}.npz"
        run_cmd(
            [
                py, "train_mlp_numpy.py",
                "--npz", str(profile_b),
                "--out", str(model_b),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--poi", str(args.poi),
                "--h1", str(args.h1),
                "--h2", str(args.h2),
                "--dropout", str(args.dropout),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--l2", str(args.l2),
                "--seed", str(args.seed),
            ]
        )

        rank_json = bdir / f"keyrank_b{b:02d}.json"
        cmd = [
            py, "key_rank_from_mlp.py",
            "--model", str(model_b),
            "--dataset", str(attack_b),
            "--byte", str(b),
            "--label-mode", args.label_mode,
            "--n-traces", "0",
            "--repeats", str(args.repeats),
            "--seed", str(args.seed),
            "--out", str(rank_json),
        ]
        if true_key is not None:
            cmd.extend(["--true-key", hex(true_key[b])])
        run_cmd(cmd)

        with open(rank_json, "r") as f:
            jr = json.load(f)
        summary["bytes"].append(
            {
                "byte": b,
                "center": center,
                "profile_n": n_prof,
                "attack_n": n_att,
                "label_mode": args.label_mode,
                "final_avg_rank": jr.get("final_avg_rank"),
                "traces_to_rank1": jr.get("traces_to_rank1"),
                "top5": jr.get("top5_final_scores_repeat1", []),
            }
        )

    out_sum = outdir / "summary.json"
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    print("[DONE] saved:", out_sum)


if __name__ == "__main__":
    main()
