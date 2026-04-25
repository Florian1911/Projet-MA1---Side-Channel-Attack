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
from picosdk.errors import PicoSDKCtypesError


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
        raise ValueError(f"Range non supportee: {mv} mV")
    return lut[mv]


def main() -> None:
    ap = argparse.ArgumentParser(description="High-side differential stability check using CH B and CH C.")
    ap.add_argument("--out-prefix", default="highside_diff_check")
    ap.add_argument("--n-blocks", type=int, default=400)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument("--range-b-mv", type=int, default=5000)
    ap.add_argument("--range-c-mv", type=int, default=5000)
    ap.add_argument("--probe-att-b", type=int, default=10)
    ap.add_argument("--probe-att-c", type=int, default=10)
    ap.add_argument("--sleep-ms", type=float, default=20.0)
    args = ap.parse_args()

    pwr_not_conn = 286
    non_usb3 = 282

    chandle = ctypes.c_int16()
    res = getattr(ps, "PS5000A_RESOLUTION", None) or ps.PS5000A_DEVICE_RESOLUTION
    status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, res["PS5000A_DR_12BIT"])
    if status in (pwr_not_conn, non_usb3):
        status = ps.ps5000aChangePowerSource(chandle, status)
    assert_pico_ok(status)

    ch_b = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    ch_c = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
    coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
    range_b = mv_to_range_id(args.range_b_mv)
    range_c = mv_to_range_id(args.range_c_mv)

    assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_b, 1, coupling, range_b, 0))
    try:
        assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_c, 1, coupling, range_c, 0))
    except PicoSDKCtypesError as e:
        ps.ps5000aCloseUnit(chandle)
        raise RuntimeError(
            "Impossible d'activer CH C en alimentation USB (PICO_CHANNEL_DISABLED_DUE_TO_USB_POWERED).\n"
            "Il faut une alimentation externe du PicoScope pour utiliser B/C en meme temps."
        ) from e

    # No trigger: immediate captures.
    assert_pico_ok(ps.ps5000aSetSimpleTrigger(
        chandle,
        0,
        ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"],
        0,
        ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
        0,
        0,
    ))

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))
    dt_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        chandle, args.timebase, args.num_samples, ctypes.byref(dt_ns), ctypes.byref(returned), 0
    ))

    buf_b = (ctypes.c_int16 * args.num_samples)()
    buf_c = (ctypes.c_int16 * args.num_samples)()
    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_b, ctypes.byref(buf_b), None, args.num_samples, 0, 0))
    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_c, ctypes.byref(buf_c), None, args.num_samples, 0, 0))

    tr_b = np.zeros((args.n_blocks, args.num_samples), dtype=np.float32)
    tr_c = np.zeros((args.n_blocks, args.num_samples), dtype=np.float32)
    tr_d = np.zeros((args.n_blocks, args.num_samples), dtype=np.float32)
    mean_b = np.zeros((args.n_blocks,), dtype=np.float32)
    mean_c = np.zeros((args.n_blocks,), dtype=np.float32)
    mean_d = np.zeros((args.n_blocks,), dtype=np.float32)
    std_d = np.zeros((args.n_blocks,), dtype=np.float32)
    t_wall = np.zeros((args.n_blocks,), dtype=np.float64)

    print("Capture diff B-C en cours (sans trigger).", flush=True)
    t0 = time.time()
    try:
        for i in range(args.n_blocks):
            assert_pico_ok(ps.ps5000aRunBlock(chandle, 0, args.num_samples, args.timebase, None, 0, None, None))
            ready = ctypes.c_int16(0)
            deadline = time.time() + 2.0
            while not ready.value:
                if time.time() >= deadline:
                    raise RuntimeError("Timeout capture bloc")
                assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))

            num = ctypes.c_int32(args.num_samples)
            overflow = ctypes.c_int16()
            assert_pico_ok(ps.ps5000aGetValues(chandle, 0, ctypes.byref(num), 1, 0, 0, ctypes.byref(overflow)))

            adc_b = np.array(buf_b[:num.value], dtype=np.int32)
            adc_c = np.array(buf_c[:num.value], dtype=np.int32)
            mv_b = np.asarray(adc2mV(adc_b.tolist(), range_b, max_adc), dtype=np.float32) * float(args.probe_att_b)
            mv_c = np.asarray(adc2mV(adc_c.tolist(), range_c, max_adc), dtype=np.float32) * float(args.probe_att_c)
            mv_d = mv_b - mv_c  # high-side differential

            if num.value < args.num_samples:
                pb = np.zeros((args.num_samples,), dtype=np.float32)
                pc = np.zeros((args.num_samples,), dtype=np.float32)
                pd = np.zeros((args.num_samples,), dtype=np.float32)
                pb[:num.value] = mv_b
                pc[:num.value] = mv_c
                pd[:num.value] = mv_d
                mv_b, mv_c, mv_d = pb, pc, pd

            tr_b[i] = mv_b
            tr_c[i] = mv_c
            tr_d[i] = mv_d
            mean_b[i] = float(mv_b.mean())
            mean_c[i] = float(mv_c.mean())
            mean_d[i] = float(mv_d.mean())
            std_d[i] = float(mv_d.std())
            t_wall[i] = time.time() - t0

            if (i + 1) <= 3 or (i + 1) % 50 == 0:
                print(
                    f"{i+1}/{args.n_blocks} meanB={mean_b[i]:.3f}mV meanC={mean_c[i]:.3f}mV "
                    f"meanDiff={mean_d[i]:.3f}mV stdDiff={std_d[i]:.3f}mV ovf={int(overflow.value)}",
                    flush=True,
                )

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)

    out_npz = f"{args.out_prefix}.npz"
    out_json = f"{args.out_prefix}.json"
    out_png = f"{args.out_prefix}.png"
    np.savez(
        out_npz,
        traces_b=tr_b,
        traces_c=tr_c,
        traces_diff=tr_d,
        mean_b=mean_b,
        mean_c=mean_c,
        mean_diff=mean_d,
        std_diff=std_d,
        t_wall_s=t_wall,
    )
    with open(out_json, "w") as f:
        json.dump(
            {
                "n_blocks": int(args.n_blocks),
                "num_samples": int(args.num_samples),
                "timebase": int(args.timebase),
                "dt_ns": float(dt_ns.value),
                "range_b_mv": int(args.range_b_mv),
                "range_c_mv": int(args.range_c_mv),
                "probe_att_b": int(args.probe_att_b),
                "probe_att_c": int(args.probe_att_c),
                "sleep_ms": float(args.sleep_ms),
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t_wall, mean_b, lw=1.0, label="mean CHB")
    ax[0].plot(t_wall, mean_c, lw=1.0, label="mean CHC")
    ax[0].set_ylabel("mV")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.25)

    ax[1].plot(t_wall, mean_d, lw=1.0, color="tab:green", label="mean (B-C)")
    ax[1].set_ylabel("mV")
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.25)

    ax[2].plot(t_wall, std_d, lw=1.0, color="tab:orange", label="std (B-C)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("mV")
    ax[2].legend(loc="upper right")
    ax[2].grid(alpha=0.25)

    fig.suptitle("High-side differential stability (CHB - CHC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)

    print(f"Saved: {out_npz}, {out_json}, {out_png}")


if __name__ == "__main__":
    main()
