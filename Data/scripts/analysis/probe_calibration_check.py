#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_pico_runtime() -> None:
    candidates = [
        Path("/opt/picoscope/lib/libpicoipp.so"),
        Path("/usr/local/lib/libpicoipp.so"),
        Path("/usr/lib64/libpicoipp.so"),
        Path("/usr/lib/libpicoipp.so"),
    ]
    lib = next((p for p in candidates if p.is_file()), None)
    if lib is None:
        raise RuntimeError("libpicoipp.so introuvable")
    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


ensure_pico_runtime()

from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps5000a import ps5000a as ps


POWER_WARNINGS = {286, 282}
RANGES = [
    ("50mV", "PS5000A_50MV", 50),
    ("100mV", "PS5000A_100MV", 100),
    ("200mV", "PS5000A_200MV", 200),
    ("500mV", "PS5000A_500MV", 500),
    ("1V", "PS5000A_1V", 1000),
    ("2V", "PS5000A_2V", 2000),
    ("5V", "PS5000A_5V", 5000),
]


def open_scope():
    chandle = ctypes.c_int16()
    resolution_dict = getattr(ps, "PS5000A_RESOLUTION", None) or ps.PS5000A_DEVICE_RESOLUTION
    status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution_dict["PS5000A_DR_12BIT"])
    if status in POWER_WARNINGS:
        status = ps.ps5000aChangePowerSource(chandle, status)
    assert_pico_ok(status)
    return chandle


def capture(chandle, channel_name: str, range_name: str, samples: int, timebase: int) -> tuple[np.ndarray, float, int, int]:
    channel = ps.PS5000A_CHANNEL[channel_name]
    range_id = ps.PS5000A_RANGE[range_name]
    assert_pico_ok(
        ps.ps5000aSetChannel(
            chandle,
            channel,
            1,
            ps.PS5000A_COUPLING["PS5000A_DC"],
            range_id,
            0.0,
        )
    )
    assert_pico_ok(
        ps.ps5000aSetSimpleTrigger(
            chandle,
            0,
            channel,
            0,
            ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
            0,
            0,
        )
    )

    interval_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(
        ps.ps5000aGetTimebase2(
            chandle,
            timebase,
            samples,
            ctypes.byref(interval_ns),
            ctypes.byref(returned),
            0,
        )
    )

    buf = (ctypes.c_int16 * samples)()
    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, channel, ctypes.byref(buf), None, samples, 0, 0))
    assert_pico_ok(ps.ps5000aRunBlock(chandle, 0, samples, timebase, None, 0, None, None))

    ready = ctypes.c_int16(0)
    while not ready.value:
        assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))
        time.sleep(0.001)

    n = ctypes.c_int32(samples)
    overflow = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aGetValues(chandle, 0, ctypes.byref(n), 1, 0, 0, ctypes.byref(overflow)))

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))
    adc = np.array(buf[: n.value], dtype=np.int16)
    mv = np.asarray(adc2mV(adc.tolist(), range_id, max_adc), dtype=np.float64)
    return mv, float(interval_ns.value), int(overflow.value), int(max_adc.value)


def choose_range(y: np.ndarray) -> str:
    # Keep both absolute level and swing under about 70% of full-scale.
    # The calibration output is often unipolar (around 0 V to 2 V), so p2p
    # alone may select +/-2 V and clip the high plateau.
    p2p_mv = float(np.ptp(y))
    max_abs_mv = float(np.max(np.abs(y)))
    target = max(50.0, p2p_mv * 0.60, max_abs_mv * 1.35)
    for _, range_name, mv in RANGES:
        if target < 2 * mv * 0.70:
            return range_name
    return "PS5000A_5V"


def crossing_indices(y: np.ndarray, level: float, rising: bool) -> np.ndarray:
    if rising:
        mask = (y[:-1] < level) & (y[1:] >= level)
    else:
        mask = (y[:-1] > level) & (y[1:] <= level)
    return np.flatnonzero(mask)


def interp_crossing(y: np.ndarray, idx: int, level: float) -> float:
    y0 = y[idx]
    y1 = y[idx + 1]
    if y1 == y0:
        return float(idx)
    return float(idx + (level - y0) / (y1 - y0))


def analyze_square(y: np.ndarray, dt_ns: float) -> dict:
    lo = float(np.percentile(y, 5))
    hi = float(np.percentile(y, 95))
    amp = hi - lo
    mid = 0.5 * (lo + hi)
    rising = crossing_indices(y, mid, True)
    falling = crossing_indices(y, mid, False)

    periods = np.diff([interp_crossing(y, int(i), mid) for i in rising])
    period_samples = float(np.median(periods)) if len(periods) else float("nan")
    freq_hz = 1e9 / (period_samples * dt_ns) if np.isfinite(period_samples) and period_samples > 0 else float("nan")

    high = y > mid
    duty = float(np.mean(high))
    low_noise = float(np.std(y[y < lo + 0.2 * amp], ddof=1)) if np.any(y < lo + 0.2 * amp) else float("nan")
    high_noise = float(np.std(y[y > hi - 0.2 * amp], ddof=1)) if np.any(y > hi - 0.2 * amp) else float("nan")

    rise_times = []
    fall_times = []
    l10 = lo + 0.1 * amp
    l90 = lo + 0.9 * amp
    for i in rising[1:-1]:
        left = max(0, int(i) - 200)
        right = min(len(y) - 2, int(i) + 200)
        local = y[left : right + 2]
        r10 = crossing_indices(local, l10, True)
        r90 = crossing_indices(local, l90, True)
        if len(r10) and len(r90):
            t10 = interp_crossing(local, int(r10[0]), l10) + left
            t90 = interp_crossing(local, int(r90[0]), l90) + left
            if t90 > t10:
                rise_times.append((t90 - t10) * dt_ns)
    for i in falling[1:-1]:
        left = max(0, int(i) - 200)
        right = min(len(y) - 2, int(i) + 200)
        local = y[left : right + 2]
        f90 = crossing_indices(local, l90, False)
        f10 = crossing_indices(local, l10, False)
        if len(f90) and len(f10):
            t90 = interp_crossing(local, int(f90[0]), l90) + left
            t10 = interp_crossing(local, int(f10[0]), l10) + left
            if t10 > t90:
                fall_times.append((t10 - t90) * dt_ns)

    overshoot = float((np.max(y) - hi) / amp * 100.0) if amp else float("nan")
    undershoot = float((lo - np.min(y)) / amp * 100.0) if amp else float("nan")

    return {
        "min_mv": float(np.min(y)),
        "max_mv": float(np.max(y)),
        "p2p_mv": float(np.ptp(y)),
        "low_level_p5_mv": lo,
        "high_level_p95_mv": hi,
        "amplitude_p95_p5_mv": amp,
        "mean_mv": float(np.mean(y)),
        "std_mv": float(np.std(y, ddof=1)),
        "frequency_hz": freq_hz,
        "rising_edges": int(len(rising)),
        "falling_edges": int(len(falling)),
        "duty_cycle": duty,
        "plateau_noise_low_mv_rms": low_noise,
        "plateau_noise_high_mv_rms": high_noise,
        "rise_time_10_90_ns_median": float(np.median(rise_times)) if rise_times else float("nan"),
        "fall_time_90_10_ns_median": float(np.median(fall_times)) if fall_times else float("nan"),
        "overshoot_percent_of_amplitude": overshoot,
        "undershoot_percent_of_amplitude": undershoot,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe compensation/calibration check on PicoScope Channel A")
    ap.add_argument("--channel", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument("--probe", choices=["x1", "x10"], default="x1", help="Used only to report probe-tip voltage")
    ap.add_argument("--out-prefix", default="probe_A_calibration")
    args = ap.parse_args()
    channel_name = f"PS5000A_CHANNEL_{args.channel}"

    chandle = open_scope()
    try:
        coarse, dt_ns, overflow, max_adc = capture(chandle, channel_name, "PS5000A_5V", args.samples, args.timebase)
        chosen = choose_range(coarse)
        y, dt_ns, overflow, max_adc = capture(chandle, channel_name, chosen, args.samples, args.timebase)
    finally:
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)

    factor = 10.0 if args.probe == "x10" else 1.0
    report = {
        "channel": args.channel,
        "probe_setting": args.probe,
        "range": chosen,
        "samples": int(len(y)),
        "dt_ns": dt_ns,
        "duration_ms": float(len(y) * dt_ns * 1e-6),
        "overflow": overflow,
        "max_adc": max_adc,
        "bnc_input": analyze_square(y, dt_ns),
    }
    report["probe_tip_estimated"] = {
        k: (float(v) * factor if isinstance(v, float) and k.endswith("_mv") else v)
        for k, v in report["bnc_input"].items()
    }

    out_json = Path(f"{args.out_prefix}_summary.json")
    out_csv = Path(f"{args.out_prefix}_trace.csv")
    out_png = Path(f"{args.out_prefix}_trace.png")

    out_json.write_text(json.dumps(report, indent=2))
    t_us = np.arange(len(y)) * dt_ns * 1e-3
    np.savetxt(out_csv, np.column_stack([t_us, y]), delimiter=",", header="time_us,chA_mV_at_BNC", comments="")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    ax[0].plot(t_us, y, lw=0.8)
    ax[0].set_title(f"Channel {args.channel} calibration waveform")
    ax[0].set_xlabel("Time (us)")
    ax[0].set_ylabel("BNC input (mV)")
    ax[0].grid(True, alpha=0.25)
    zoom = min(len(y), max(2000, len(y) // 20))
    ax[1].plot(t_us[:zoom], y[:zoom], lw=1.0)
    ax[1].set_title("Zoom")
    ax[1].set_xlabel("Time (us)")
    ax[1].set_ylabel("BNC input (mV)")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_png}")
    print(json.dumps(report["bnc_input"], indent=2))


if __name__ == "__main__":
    main()
