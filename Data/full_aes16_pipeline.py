import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_key_16(s: str):
    t = s.strip().lower().replace("0x", "")
    for sep in [",", ":", "-", " "]:
        t = t.replace(sep, "")
    if len(t) != 32:
        raise ValueError("profile key must contain exactly 16 bytes (32 hex chars)")
    out = []
    for i in range(0, 32, 2):
        out.append(int(t[i:i + 2], 16))
    return out


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_and_merge_profile_datasets(profile_paths, out_path: Path, profile_key):
    traces_list = []
    plains_list = []
    over_list = []
    has_over = True

    for p in profile_paths:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(f"missing profiling dataset: {pp}")
        d = np.load(str(pp))
        if "traces" not in d.files or "plaintexts" not in d.files:
            raise ValueError(f"{pp} must contain traces and plaintexts")
        traces_list.append(d["traces"])
        plains_list.append(d["plaintexts"])
        if "overflows" in d.files:
            over_list.append(d["overflows"])
        else:
            has_over = False

    traces = np.concatenate(traces_list, axis=0)
    plains = np.concatenate(plains_list, axis=0)
    save = {
        "traces": traces,
        "plaintexts": plains,
        "key": np.array(profile_key, dtype=np.uint8),
    }
    if has_over and len(over_list) == len(traces_list):
        save["overflows"] = np.concatenate(over_list, axis=0)

    np.savez(str(out_path), **save)
    print(f"[MAIN] merged profiling dataset saved: {out_path} | traces={traces.shape}", flush=True)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Train 16-byte profiling models then recover 16-byte unknown AES key"
    )
    ap.add_argument("--profile-dataset", default="", help="Aligned profiling dataset (.npz), full traces")
    ap.add_argument(
        "--profile-datasets",
        default="",
        help="Comma-separated list of aligned profiling datasets; if provided, they are merged automatically",
    )
    ap.add_argument("--attack-dataset", required=True, help="Aligned attack dataset (.npz), full traces")
    ap.add_argument(
        "--profile-key",
        required=True,
        help="Known profiling AES-128 key, ex: 9144A72E5CD3186FB2097AE133C85DF0",
    )
    ap.add_argument("--workdir", default="campaign_aes16")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--poi", type=int, default=200)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--max-shift", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip-train", action="store_true", help="Reuse existing trained models in workdir")
    args = ap.parse_args()

    profile_key = parse_key_16(args.profile_key)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    profile_paths = []
    if args.profile_datasets.strip():
        profile_paths = [x.strip() for x in args.profile_datasets.split(",") if x.strip()]
    elif args.profile_dataset.strip():
        profile_paths = [args.profile_dataset.strip()]
    else:
        raise ValueError("provide --profile-dataset or --profile-datasets")

    if len(profile_paths) > 1:
        merged_profile = workdir / "profile_merged_multisession.npz"
        profile_dataset = load_and_merge_profile_datasets(profile_paths, merged_profile, profile_key)
    else:
        profile_dataset = profile_paths[0]
        if not Path(profile_dataset).exists():
            raise FileNotFoundError(f"missing profiling dataset: {profile_dataset}")

    if not Path(args.attack_dataset).exists():
        raise FileNotFoundError(f"missing attack dataset: {args.attack_dataset}")

    py = sys.executable

    recovered = []
    details = []

    print("[MAIN] phase 1/2: profiling train (16 bytes)", flush=True)
    for b in range(16):
        bdir = workdir / f"byte_{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)

        profile_hw = bdir / f"profile_hw_b{b:02d}.npz"
        profile_hw_json = profile_hw.with_suffix(".json")
        model_npz = bdir / f"model_b{b:02d}.npz"

        # Auto center per byte from profiling SNR
        run_cmd(
            [
                py,
                "prepare_hw_dataset.py",
                "--in",
                profile_dataset,
                "--out",
                str(profile_hw),
                "--byte",
                str(b),
                "--key",
                hex(profile_key[b]),
                "--center",
                "-1",
                "--window",
                str(args.window),
            ]
        )

        if not args.skip_train:
            run_cmd(
                [
                    py,
                    "train_mlp_numpy.py",
                    "--npz",
                    str(profile_hw),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--poi",
                    str(args.poi),
                    "--h1",
                    str(args.h1),
                    "--h2",
                    str(args.h2),
                    "--dropout",
                    str(args.dropout),
                    "--lr",
                    str(args.lr),
                    "--l2",
                    str(args.l2),
                    "--patience",
                    str(args.patience),
                    "--seed",
                    str(args.seed),
                    "--out",
                    str(model_npz),
                ]
            )

        if not model_npz.exists():
            raise FileNotFoundError(f"missing model: {model_npz}")

        if not profile_hw_json.exists():
            raise FileNotFoundError(f"missing metadata: {profile_hw_json}")

    print("[MAIN] phase 2/2: unknown-key recovery (16 bytes)", flush=True)
    for b in range(16):
        bdir = workdir / f"byte_{b:02d}"
        profile_hw_json = bdir / f"profile_hw_b{b:02d}.json"
        model_npz = bdir / f"model_b{b:02d}.npz"

        with profile_hw_json.open("r") as f:
            meta = json.load(f)
        center = int(meta["window_center"])

        # Crop attack traces with same window as profiling for this byte.
        # key=0x00 here is only a placeholder for labels; labels are not used in unknown-key recovery.
        attack_hw = bdir / f"attack_hw_b{b:02d}.npz"
        run_cmd(
            [
                py,
                "prepare_hw_dataset.py",
                "--in",
                args.attack_dataset,
                "--out",
                str(attack_hw),
                "--byte",
                str(b),
                "--key",
                "0x00",
                "--center",
                str(center),
                "--window",
                str(args.window),
            ]
        )

        rec_json = bdir / f"recover_b{b:02d}.json"
        run_cmd(
            [
                py,
                "recover_byte_unknown.py",
                "--model",
                str(model_npz),
                "--dataset",
                str(attack_hw),
                "--byte",
                str(b),
                "--max-shift",
                str(args.max_shift),
                "--out",
                str(rec_json),
            ]
        )

        with rec_json.open("r") as f:
            rec = json.load(f)
        key_b = int(rec["recovered_key_byte"])
        recovered.append(key_b)
        details.append(
            {
                "byte": b,
                "recovered": key_b,
                "best_shift": int(rec["best_shift"]),
                "margin": float(rec["margin"]),
                "top5": rec["top5"],
                "window_center": center,
            }
        )

    key_hex = "".join(f"{x:02X}" for x in recovered)
    key_c = ",".join(f"0x{x:02X}" for x in recovered)

    summary = {
        "profile_dataset": profile_dataset,
        "profile_datasets_input": profile_paths,
        "attack_dataset": args.attack_dataset,
        "profile_key": [int(x) for x in profile_key],
        "recovered_key": recovered,
        "recovered_key_hex": key_hex,
        "details": details,
    }

    out_json = workdir / "recovered_key_16bytes.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] recovered AES-128 key:")
    print(f"HEX: {key_hex}")
    print(f"C  : {{{key_c}}}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
