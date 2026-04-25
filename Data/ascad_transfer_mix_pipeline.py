#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
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


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_key_hex(s: str) -> str:
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "").upper()
    if len(t) != 32:
        raise ValueError("--true-key-hex must contain 32 hex chars")
    int(t, 16)
    return t


def load_pois(path: str) -> list[int]:
    with open(path, "r") as f:
        j = json.load(f)
    if "bytes" in j:
        pois = [int(row["poi"]) for row in j["bytes"]]
    elif "poi_global_per_byte" in j:
        pois = [int(x) for x in j["poi_global_per_byte"][:16]]
    else:
        raise ValueError("POI JSON missing bytes/poi_global_per_byte")
    if len(pois) != 16:
        raise ValueError("Expected 16 POIs")
    return pois


def build_labels(plains: np.ndarray, key16: np.ndarray, b: int) -> np.ndarray:
    return AES_SBOX[np.bitwise_xor(plains[:, b], key16[b])].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="ASCAD transfer mix pipeline (cross-session + local calib)")
    ap.add_argument("--profile-cross-npz", required=True)
    ap.add_argument("--profile-local-npz", required=True)
    ap.add_argument("--attack-npz", required=True)
    ap.add_argument("--poi-json", required=True)
    ap.add_argument("--true-key-hex", required=True)
    ap.add_argument("--workdir", default="campaign_ascad16_transfer_mix")
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-shift", type=int, default=120)
    ap.add_argument("--local-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5050)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pois = load_pois(args.poi_json)
    true_hex = parse_key_hex(args.true_key_hex)
    py = sys.executable

    dc = np.load(args.profile_cross_npz)
    dl = np.load(args.profile_local_npz)
    xc = dc["traces"].astype(np.float32)
    xl = dl["traces"].astype(np.float32)
    pc = dc["plaintexts"].astype(np.uint8)
    pl = dl["plaintexts"].astype(np.uint8)
    kc = dc["key"][:16].astype(np.uint8)
    kl = dl["key"][:16].astype(np.uint8)

    recovered = []
    ranks = []
    details = []

    for b in range(16):
        bdir = workdir / f"byte_{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        mix_npz = bdir / "train_mix.npz"
        model_path = bdir / "model.keras"
        stats_path = bdir / "model.npz"
        rec_path = bdir / "recover.json"

        yc = build_labels(pc, kc, b)
        yl = build_labels(pl, kl, b)

        xmix = np.concatenate([xc] + [xl] * args.local_repeat, axis=0)
        ymix = np.concatenate([yc] + [yl] * args.local_repeat, axis=0)

        # Labels are byte-specific; plaintexts are kept only for compatibility/debug.
        np.savez(
            mix_npz,
            traces=xmix,
            labels=ymix,
            plaintexts=np.zeros((xmix.shape[0], 16), dtype=np.uint8),
        )

        run(
            [
                py,
                "train_ascad_cnn.py",
                "--train-npz",
                str(mix_npz),
                "--byte",
                str(b),
                "--key",
                "0x00",
                "--window",
                str(args.window),
                "--center",
                str(int(pois[b])),
                "--preproc",
                "center_detrend",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--seed",
                str(args.seed + b),
                "--out",
                str(model_path),
            ]
        )

        run(
            [
                py,
                "ascad_attack_unknown.py",
                "--model",
                str(model_path),
                "--stats-npz",
                str(stats_path),
                "--attack-npz",
                args.attack_npz,
                "--byte",
                str(b),
                "--center",
                str(int(pois[b])),
                "--window",
                str(args.window),
                "--preproc",
                "center_detrend",
                "--max-shift",
                str(args.max_shift),
                "--true-key-hex",
                true_hex,
                "--out",
                str(rec_path),
            ]
        )

        rec = json.loads(rec_path.read_text())
        recovered.append(int(rec["recovered_key_byte"]))
        ranks.append(int(rec["true_rank"]))
        details.append(
            {
                "byte": b,
                "center": int(pois[b]),
                "recovered_key_byte": int(rec["recovered_key_byte"]),
                "true_rank": int(rec["true_rank"]),
                "best_shift": int(rec["best_shift"]),
                "margin": float(rec["margin"]),
                "top5": rec["top5"],
            }
        )

    rec_hex = "".join(f"{x:02X}" for x in recovered)
    rank1 = sum(1 for r in ranks if r == 1)
    rank5 = sum(1 for r in ranks if r <= 5)

    out = {
        "profile_cross_npz": args.profile_cross_npz,
        "profile_local_npz": args.profile_local_npz,
        "attack_npz": args.attack_npz,
        "poi_json": args.poi_json,
        "true_key_hex": true_hex,
        "window": int(args.window),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "patience": int(args.patience),
        "max_shift": int(args.max_shift),
        "local_repeat": int(args.local_repeat),
        "recovered_key": recovered,
        "recovered_key_hex": rec_hex,
        "rank1_count": int(rank1),
        "rank5_count": int(rank5),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ranks_per_byte": ranks,
        "details": details,
    }
    out_path = workdir / "summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("[DONE] recovered:", rec_hex)
    print(f"[DONE] rank1={rank1}/16 rank5={rank5}/16 mean_rank={np.mean(ranks):.2f}")
    print("[DONE] saved:", out_path)


if __name__ == "__main__":
    main()
