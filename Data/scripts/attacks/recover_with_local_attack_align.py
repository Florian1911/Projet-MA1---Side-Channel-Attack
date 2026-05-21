import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from align_local_per_trace import (
    apply_integer_shifts,
    center_and_detrend,
    estimate_lags_local_ref,
)


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def align_attack_for_center(
    traces: np.ndarray,
    center: int,
    align_window: int,
    align_max_shift: int,
    iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    cur = traces.copy()
    cum_shift = np.zeros((traces.shape[0],), dtype=np.int32)
    for _ in range(iters):
        proc = center_and_detrend(cur)
        lags, _ = estimate_lags_local_ref(
            proc=proc,
            center=center,
            window=align_window,
            max_shift=align_max_shift,
            ref_kind="median",
        )
        delta = (-lags).astype(np.int32)
        cur = apply_integer_shifts(cur, delta)
        cum_shift += delta
    return cur, cum_shift


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recover 16-byte unknown key using per-byte local attack alignment + existing models"
    )
    ap.add_argument("--workdir", required=True, help="Existing campaign folder with byte_XX/model_bXX.npz")
    ap.add_argument("--attack-npz", required=True, help="Attack dataset with traces/plaintexts")
    ap.add_argument("--outdir", default="campaign_recover_local_attack_align")
    ap.add_argument("--align-window", type=int, default=220)
    ap.add_argument("--align-max-shift", type=int, default=12)
    ap.add_argument("--align-iters", type=int, default=2)
    ap.add_argument("--recover-max-shift", type=int, default=160)
    ap.add_argument("--bytes", default="", help="Optional comma-separated bytes, e.g. 0,1,2")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.attack_npz)
    traces_full = d["traces"].astype(np.float32)
    plains = d["plaintexts"].astype(np.uint8)

    if args.bytes.strip():
        byte_list = [int(x.strip()) for x in args.bytes.split(",") if x.strip()]
    else:
        byte_list = list(range(16))

    py = sys.executable
    recovered = []
    details = []

    for b in byte_list:
        bdir_in = workdir / f"byte_{b:02d}"
        bdir_out = outdir / f"byte_{b:02d}"
        bdir_out.mkdir(parents=True, exist_ok=True)

        model_npz = bdir_in / f"model_b{b:02d}.npz"
        meta_json = bdir_in / f"profile_hw_b{b:02d}.json"
        if not model_npz.exists():
            raise FileNotFoundError(f"missing model: {model_npz}")
        if not meta_json.exists():
            raise FileNotFoundError(f"missing profile metadata: {meta_json}")

        with meta_json.open("r") as f:
            meta = json.load(f)
        center = int(meta["window_center"])
        w0 = int(meta["window_start"])
        w1 = int(meta["window_end"])

        aligned_full, shift_vec = align_attack_for_center(
            traces=traces_full,
            center=center,
            align_window=args.align_window,
            align_max_shift=args.align_max_shift,
            iters=args.align_iters,
        )
        traces_crop = aligned_full[:, w0:w1]

        attack_npz = bdir_out / f"attack_hw_b{b:02d}_local.npz"
        np.savez(
            attack_npz,
            traces=traces_crop,
            plaintexts=plains,
            local_align_center=np.int32(center),
            local_align_window=np.int32(args.align_window),
            local_align_max_shift=np.int32(args.align_max_shift),
            local_align_iters=np.int32(args.align_iters),
            local_align_shift_per_trace=shift_vec.astype(np.int16),
        )
        print(
            f"[BYTE {b:02d}] saved attack: {attack_npz} | "
            f"crop={traces_crop.shape} | shift_mean_abs={float(np.mean(np.abs(shift_vec))):.3f}",
            flush=True,
        )

        rec_json = bdir_out / f"recover_b{b:02d}.json"
        run_cmd(
            [
                py,
                "recover_byte_unknown.py",
                "--model",
                str(model_npz),
                "--dataset",
                str(attack_npz),
                "--byte",
                str(b),
                "--max-shift",
                str(args.recover_max_shift),
                "--out",
                str(rec_json),
            ]
        )

        with rec_json.open("r") as f:
            rec = json.load(f)
        recovered.append(int(rec["recovered_key_byte"]))
        details.append(
            {
                "byte": int(b),
                "recovered": int(rec["recovered_key_byte"]),
                "best_shift": int(rec["best_shift"]),
                "margin": float(rec["margin"]),
                "top5": rec["top5"],
                "window_center": int(center),
                "align_shift_mean_abs": float(np.mean(np.abs(shift_vec))),
                "align_shift_p95_abs": float(np.quantile(np.abs(shift_vec), 0.95)),
            }
        )

    if byte_list == list(range(16)):
        key_hex = "".join(f"{x:02X}" for x in recovered)
    else:
        key_hex = "".join(f"{x:02X}" for x in recovered)

    out = {
        "workdir_models": str(workdir),
        "attack_dataset": str(args.attack_npz),
        "outdir": str(outdir),
        "bytes": byte_list,
        "align_window": int(args.align_window),
        "align_max_shift": int(args.align_max_shift),
        "align_iters": int(args.align_iters),
        "recover_max_shift": int(args.recover_max_shift),
        "recovered_key": recovered,
        "recovered_key_hex_partial": key_hex,
        "details": details,
    }
    out_json = outdir / "recovered_key_local_attack_align.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] saved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
