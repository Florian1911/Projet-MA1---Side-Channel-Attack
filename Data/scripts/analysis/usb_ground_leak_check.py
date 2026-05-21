import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_pico_runtime() -> None:
    candidates = [
        Path("/opt/picoscope/lib/libpicoipp.so"),
        Path("/usr/local/lib/libpicoipp.so"),
        Path("/usr/lib64/libpicoipp.so"),
        Path("/usr/lib/libpicoipp.so"),
    ]
    lib = next((p for p in candidates if p.is_file()), None)
    if lib is None:
        searched = "\n".join(f" - {p}" for p in candidates)
        raise RuntimeError(f"libpicoipp.so introuvable.\nChemins verifies:\n{searched}")

    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir

    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


_ensure_pico_runtime()

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok


def mv_to_range_id(mv: int) -> int:
    lut = {
        10: ps.PS5000A_RANGE["PS5000A_10MV"],
        20: ps.PS5000A_RANGE["PS5000A_20MV"],
        50: ps.PS5000A_RANGE["PS5000A_50MV"],
        100: ps.PS5000A_RANGE["PS5000A_100MV"],
        200: ps.PS5000A_RANGE["PS5000A_200MV"],
        500: ps.PS5000A_RANGE["PS5000A_500MV"],
        1000: ps.PS5000A_RANGE["PS5000A_1V"],
        2000: ps.PS5000A_RANGE["PS5000A_2V"],
        5000: ps.PS5000A_RANGE["PS5000A_5V"],
        10000: ps.PS5000A_RANGE["PS5000A_10V"],
        20000: ps.PS5000A_RANGE["PS5000A_20V"],
    }
    if mv not in lut:
        raise ValueError(f"Range non supporte: {mv} mV")
    return lut[mv]


def main() -> None:
    ap = argparse.ArgumentParser(description="USB ground leakage check on CH B (continuous blocks, no UART).")
    ap.add_argument("--out-prefix", default="usb_leak_check")
    ap.add_argument("--n-blocks", type=int, default=400)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument("--range-mv", type=int, default=200)
    ap.add_argument("--probe-att", type=int, default=1)
    ap.add_argument("--sleep-ms", type=float, default=20.0)
    args = ap.parse_args()

    pwr_not_conn = 286
    non_usb3 = 282

    ch = ctypes.c_int16()
    res = getattr(ps, "PS5000A_RESOLUTION", None) or ps.PS5000A_DEVICE_RESOLUTION
    status = ps.ps5000aOpenUnit(ctypes.byref(ch), None, res["PS5000A_DR_12BIT"])
    if status in (pwr_not_conn, non_usb3):
        status = ps.ps5000aChangePowerSource(ch, status)
    assert_pico_ok(status)

    ch_meas = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
    range_id = mv_to_range_id(args.range_mv)
    assert_pico_ok(ps.ps5000aSetChannel(ch, ch_meas, 1, coupling, range_id, 0))

    # No trigger: immediate block capture.
    assert_pico_ok(ps.ps5000aSetSimpleTrigger(
        ch,
        0,
        ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"],
        0,
        ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
        0,
        0,
    ))

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(ch, ctypes.byref(max_adc)))
    dt_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        ch, args.timebase, args.num_samples, ctypes.byref(dt_ns), ctypes.byref(returned), 0
    ))

    buf = (ctypes.c_int16 * args.num_samples)()
    assert_pico_ok(ps.ps5000aSetDataBuffers(
        ch, ch_meas, ctypes.byref(buf), None, args.num_samples, 0, 0
    ))

    traces = np.zeros((args.n_blocks, args.num_samples), dtype=np.float32)
    means = np.zeros((args.n_blocks,), dtype=np.float32)
    stds = np.zeros((args.n_blocks,), dtype=np.float32)
    t_wall = np.zeros((args.n_blocks,), dtype=np.float64)

    print("Capture en cours. Debranche/rebranche l'USB pendant ce run.", flush=True)
    t0 = time.time()
    try:
        for i in range(args.n_blocks):
            assert_pico_ok(ps.ps5000aRunBlock(ch, 0, args.num_samples, args.timebase, None, 0, None, None))
            ready = ctypes.c_int16(0)
            deadline = time.time() + 2.0
            while not ready.value:
                if time.time() >= deadline:
                    raise RuntimeError("Timeout block capture")
                assert_pico_ok(ps.ps5000aIsReady(ch, ctypes.byref(ready)))

            num = ctypes.c_int32(args.num_samples)
            overflow = ctypes.c_int16()
            assert_pico_ok(ps.ps5000aGetValues(ch, 0, ctypes.byref(num), 1, 0, 0, ctypes.byref(overflow)))

            adc = np.array(buf[:num.value], dtype=np.int32)
            mv = np.asarray(adc2mV(adc.tolist(), range_id, max_adc), dtype=np.float32) * float(args.probe_att)
            if num.value < args.num_samples:
                pad = np.zeros((args.num_samples,), dtype=np.float32)
                pad[:num.value] = mv
                mv = pad

            traces[i] = mv
            means[i] = float(mv.mean())
            stds[i] = float(mv.std())
            t_wall[i] = time.time() - t0
            if (i + 1) <= 3 or (i + 1) % 50 == 0:
                print(f"{i+1}/{args.n_blocks} mean={means[i]:.3f}mV std={stds[i]:.3f}mV ovf={int(overflow.value)}", flush=True)

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        ps.ps5000aStop(ch)
        ps.ps5000aCloseUnit(ch)

    out_npz = f"{args.out_prefix}.npz"
    out_json = f"{args.out_prefix}.json"
    out_png = f"{args.out_prefix}.png"
    np.savez(out_npz, traces=traces, means=means, stds=stds, t_wall_s=t_wall)
    with open(out_json, "w") as f:
        json.dump(
            {
                "n_blocks": int(args.n_blocks),
                "num_samples": int(args.num_samples),
                "timebase": int(args.timebase),
                "dt_ns": float(dt_ns.value),
                "range_mv": int(args.range_mv),
                "probe_att": int(args.probe_att),
                "sleep_ms": float(args.sleep_ms),
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t_wall, means, lw=1.1)
    ax[0].set_ylabel("mean CHB (mV)")
    ax[0].grid(alpha=0.25)
    ax[1].plot(t_wall, stds, lw=1.1, color="tab:orange")
    ax[1].set_ylabel("std CHB (mV)")
    ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=0.25)
    fig.suptitle("USB leakage check (CHB low-side)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)

    print(f"Saved: {out_npz}, {out_json}, {out_png}")


if __name__ == "__main__":
    main()
