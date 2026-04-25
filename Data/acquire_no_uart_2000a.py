#!/usr/bin/env python3
"""
Acquisition SCA sans UART – rapid block mode (PicoScope 2000A).
Interface CLI alignée avec acquire_no_uart.py.
"""

import os
import time
import ctypes
import argparse
from pathlib import Path

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
        raise RuntimeError("libpicoipp.so introuvable.")
    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def main():
    _ensure_pico_runtime()
    from picosdk.ps2000a import ps2000a as ps
    from picosdk.functions import assert_pico_ok, mV2adc

    ap = argparse.ArgumentParser(description="Acquisition no-UART rapid-block (Pico 2000A)")
    ap.add_argument("--n-traces", type=int, default=5000)
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--pre-trigger", type=int, default=200)
    ap.add_argument("--output", default="dataset_no_uart_2000a.npz")
    ap.add_argument("--plaintexts", default="plaintexts_no_uart.npy")
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument("--probe-att-trig", type=int, default=10, help="Atténuation sonde trigger (1 ou 10)")
    ap.add_argument("--trig-mv-probe", type=int, default=1500, help="Seuil trigger en mV côté sonde")
    ap.add_argument("--min-duration-factor", type=float, default=0.5)
    args = ap.parse_args()

    n_traces = args.n_traces
    num_samples = args.num_samples
    pre_trigger = args.pre_trigger
    post_trigger = num_samples - pre_trigger

    # Plaintexts
    pts_all = np.load(args.plaintexts)
    if len(pts_all) < n_traces:
        raise ValueError(f"{args.plaintexts} contient {len(pts_all)} plaintexts, besoin de {n_traces}")
    pts = pts_all[:n_traces]
    print(f"[OK] {n_traces} plaintexts chargés depuis {args.plaintexts}")

    # Open scope
    print("[PICO2000A] ouverture...", flush=True)
    chandle = ctypes.c_int16()
    status = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
    assert_pico_ok(status)

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps2000aMaximumValue(chandle, ctypes.byref(max_adc)))

    # Channels: A=trigger, B=measure (comme setup précédent)
    ch_trig = ps.PS2000A_CHANNEL["PS2000A_CHANNEL_A"]
    ch_meas = ps.PS2000A_CHANNEL["PS2000A_CHANNEL_B"]
    coupling = ps.PS2000A_COUPLING["PS2000A_DC"]
    range_trig = ps.PS2000A_RANGE["PS2000A_500MV"]
    range_meas = ps.PS2000A_RANGE["PS2000A_200MV"]
    volt_range_mv = 200.0

    assert_pico_ok(ps.ps2000aSetChannel(chandle, ch_trig, 1, coupling, range_trig, 0.0))
    assert_pico_ok(ps.ps2000aSetChannel(chandle, ch_meas, 1, coupling, range_meas, 0.0))

    # Trigger
    if args.probe_att_trig not in (1, 10):
        raise ValueError("--probe-att-trig doit valoir 1 ou 10")
    thr_bnc_mv = max(1, int(args.trig_mv_probe) // int(args.probe_att_trig))
    thr_adc = mV2adc(thr_bnc_mv, range_trig, max_adc)
    trig_dir = ps.PS2000A_THRESHOLD_DIRECTION["PS2000A_RISING"]
    auto_trigger_ms = 0
    assert_pico_ok(ps.ps2000aSetSimpleTrigger(chandle, 1, ch_trig, thr_adc, trig_dir, 0, auto_trigger_ms))
    print(f"[PICO2000A] trigger: {args.trig_mv_probe} mV sonde / x{args.probe_att_trig} -> {thr_bnc_mv} mV BNC")

    # Timebase
    dt_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps2000aGetTimebase2(
        chandle, args.timebase, num_samples,
        ctypes.byref(dt_ns), 0, ctypes.byref(returned), 0
    ))
    print(f"[PICO2000A] dt_ns={dt_ns.value:.1f}  ({1e3/dt_ns.value:.1f} MS/s)")

    # Memory segmentation
    print(f"[PICO2000A] segmentation mémoire ({n_traces} segments)...", flush=True)
    max_samp_seg = ctypes.c_int32(0)
    assert_pico_ok(ps.ps2000aMemorySegments(chandle, n_traces, ctypes.byref(max_samp_seg)))
    if max_samp_seg.value < num_samples:
        ps.ps2000aCloseUnit(chandle)
        raise RuntimeError(
            f"Mémoire insuffisante: max {max_samp_seg.value} samples/segment, besoin {num_samples}. "
            f"Réduis --num-samples ou --n-traces."
        )
    assert_pico_ok(ps.ps2000aSetNoOfCaptures(chandle, n_traces))
    print(f"[PICO2000A] max samples/seg = {max_samp_seg.value}  (besoin {num_samples} ✓)")

    # Arm
    raw_adc = np.zeros((n_traces, num_samples), dtype=np.int16)
    assert_pico_ok(ps.ps2000aRunBlock(chandle, pre_trigger, post_trigger, args.timebase, 0, None, 0, None, None))

    print()
    print("=" * 60)
    print(f"  SCOPE ARMÉ – {n_traces} captures en attente")
    print()
    print("  ACTION :  Appuie sur RESET de la carte STM32 maintenant")
    print()
    print("  La LED clignote 3× puis attend 3 s avant de démarrer.")
    expected_s = n_traces * 11 / 1000
    print(f"  Durée estimée : ~{expected_s:.0f} s")
    print("=" * 60)
    print()

    # Wait ready
    total_timeout = n_traces * 0.02 + 15
    deadline = time.time() + total_timeout
    ready = ctypes.c_int16(0)
    t_start = time.time()

    while not ready.value:
        if time.time() >= deadline:
            ps.ps2000aStop(chandle)
            ps.ps2000aCloseUnit(chandle)
            raise RuntimeError(
                f"Timeout : {n_traces} captures non reçues en {total_timeout:.0f} s.\n"
                "Vérifie la carte et le trigger PB8 -> ChA."
            )
        assert_pico_ok(ps.ps2000aIsReady(chandle, ctypes.byref(ready)))
        time.sleep(0.2)

    elapsed = time.time() - t_start
    print(f"[OK] {n_traces} captures reçues en {elapsed:.1f} s")
    min_expected = max(1.0, expected_s * args.min_duration_factor)
    if elapsed < min_expected:
        ps.ps2000aStop(chandle)
        ps.ps2000aCloseUnit(chandle)
        raise RuntimeError(
            f"Capture trop rapide ({elapsed:.1f}s < {min_expected:.1f}s). "
            "Probable faux trigger; augmente --trig-mv-probe."
        )

    # Read segments
    print("[PICO2000A] lecture des données...", flush=True)
    overflows = np.zeros(n_traces, dtype=np.int16)
    ratio_mode_none = ps.PS2000A_RATIO_MODE["PS2000A_RATIO_MODE_NONE"]

    for seg in range(n_traces):
        ptr = raw_adc[seg].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        assert_pico_ok(ps.ps2000aSetDataBuffers(
            chandle, ch_meas, ptr, None, num_samples, seg, ratio_mode_none
        ))
        num = ctypes.c_int32(num_samples)
        ov = ctypes.c_int16(0)
        assert_pico_ok(ps.ps2000aGetValues(
            chandle, 0, ctypes.byref(num), 1, ratio_mode_none, seg, ctypes.byref(ov)
        ))
        overflows[seg] = ov.value
        if (seg + 1) % 500 == 0 or seg == 0:
            print(f"  {seg + 1}/{n_traces}", flush=True)

    ps.ps2000aStop(chandle)
    ps.ps2000aCloseUnit(chandle)
    print("[PICO2000A] fermé")

    # ADC -> mV
    print("[POST] conversion ADC -> mV…", flush=True)
    traces = raw_adc.astype(np.float32) * (volt_range_mv / float(max_adc.value))

    key = np.array([
        0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
        0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C
    ], dtype=np.uint8)

    np.savez(args.output, traces=traces, plaintexts=pts, key=key, overflows=overflows)
    print(f"[OK] Dataset -> {args.output}")
    print(f"     {traces.shape[0]} traces x {traces.shape[1]} samples")
    print(f"     overflows : {(overflows != 0).sum()}/{n_traces}")


if __name__ == "__main__":
    main()
