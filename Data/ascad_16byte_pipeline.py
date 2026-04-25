#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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


def parse_key_hex(s: str) -> str:
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "").upper()
    if len(t) != 32:
        raise ValueError("--true-key-hex must contain 32 hex chars")
    int(t, 16)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="ASCAD 16-byte train+attack pipeline")
    ap.add_argument("--profile-npz", required=True)
    ap.add_argument("--attack-npz", required=True)
    ap.add_argument("--poi-json", required=True)
    ap.add_argument("--true-key-hex", required=True)
    ap.add_argument("--workdir", default="campaign_ascad16")
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-shift", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pois = load_pois(args.poi_json)
    true_hex = parse_key_hex(args.true_key_hex)

    dprof = np.load(args.profile_npz)
    key = dprof["key"][:16].astype(np.uint8)
    py = sys.executable

    recovered = []
    ranks = []
    details = []
    for b in range(16):
        bdir = workdir / f"byte_{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        model_path = bdir / "model.keras"
        stats_path = bdir / "model.npz"
        rec_path = bdir / "recover.json"

        run(
            [
                py,
                "train_ascad_cnn.py",
                "--train-npz",
                args.profile_npz,
                "--byte",
                str(b),
                "--key",
                hex(int(key[b])),
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
        "profile_npz": args.profile_npz,
        "attack_npz": args.attack_npz,
        "poi_json": args.poi_json,
        "true_key_hex": true_hex,
        "window": int(args.window),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "patience": int(args.patience),
        "max_shift": int(args.max_shift),
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
