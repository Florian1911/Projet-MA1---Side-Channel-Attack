#!/usr/bin/env python3
"""Acquisition SCA no-UART en rapid-block pour PicoScope 5000A."""

import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np


def _parse_key_hex(raw: str) -> np.ndarray:
    t = raw.strip().replace(" ", "").replace(":", "").replace(",", "")
    if len(t) != 32:
        raise ValueError("--key-hex doit contenir 32 caractères hex")
    try:
        key = np.array([int(t[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)
    except ValueError as exc:
        raise ValueError("--key-hex invalide (non-hex)") from exc
    return key


def _ensure_pico_runtime() -> None:
    candidates = [
        Path("/opt/picoscope/lib/libpicoipp.so"),
        Path("/usr/local/lib/libpicoipp.so"),
        Path("/usr/lib64/libpicoipp.so"),
        Path("/usr/lib/libpicoipp.so"),
    ]
    lib = next((p for p in candidates if p.is_file()), None)
    if lib is None:
        raise RuntimeError("libpicoipp.so introuvable (PicoSDK non installe ou chemin absent).")

    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquisition SCA no-UART (Pico 5000A)")
    ap.add_argument("--n-traces", type=int, default=5000)
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--pre-trigger", type=int, default=200)
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument(
        "--meas-range",
        default="PS5000A_200MV",
        choices=[
            "PS5000A_10MV", "PS5000A_20MV", "PS5000A_50MV", "PS5000A_100MV", "PS5000A_200MV",
            "PS5000A_500MV", "PS5000A_1V", "PS5000A_2V", "PS5000A_5V", "PS5000A_10V", "PS5000A_20V",
        ],
        help="Plage du canal mesure (B)",
    )
    ap.add_argument(
        "--meas-mode",
        choices=["single_b", "diff_ab"],
        default="single_b",
        help="single_b: mesure uniquement B. diff_ab: capture A et B, puis traces = A-B.",
    )
    ap.add_argument(
        "--meas-range-a",
        default="PS5000A_5V",
        choices=[
            "PS5000A_10MV", "PS5000A_20MV", "PS5000A_50MV", "PS5000A_100MV", "PS5000A_200MV",
            "PS5000A_500MV", "PS5000A_1V", "PS5000A_2V", "PS5000A_5V", "PS5000A_10V", "PS5000A_20V",
        ],
        help="Plage du canal A en mode diff_ab (A-B)",
    )
    ap.add_argument("--probe-att-trig", type=int, default=10, help="Attenuation sonde trigger: 1 ou 10")
    ap.add_argument("--trig-mv-probe", type=int, default=1500, help="Seuil trigger en mV cote sonde")
    ap.add_argument(
        "--trigger-source",
        choices=["cha", "ext"],
        default="cha",
        help="Source trigger: cha (canal A) ou ext (entree externe/AUX)",
    )
    ap.add_argument(
        "--ext-threshold-adc",
        type=int,
        default=1200,
        help="Seuil trigger en comptes ADC quand --trigger-source ext",
    )
    ap.add_argument(
        "--debug-trigger",
        action="store_true",
        help="Affiche la config trigger (source/seuil) au lancement",
    )
    ap.add_argument("--plaintexts", default="plaintexts_no_uart.npy")
    ap.add_argument("--pt-offset", type=int, default=0, help="Offset de depart dans plaintexts")
    ap.add_argument(
        "--key-hex",
        default="",
        help="Optionnel: clé AES (32 hex chars) à stocker dans le NPZ pour validation",
    )
    ap.add_argument("--output", default="dataset_pico5000a_no_uart.npz")
    ap.add_argument("--min-duration-factor", type=float, default=0.5)
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Timeout total acquisition en secondes (sinon auto: n*0.02+15)",
    )
    ap.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periode d'affichage de progression pendant l'attente (s)",
    )
    args = ap.parse_args()

    if args.pre_trigger >= args.num_samples:
        raise ValueError("--pre-trigger doit etre strictement inferieur a --num-samples")
    if args.probe_att_trig not in (1, 10):
        raise ValueError("--probe-att-trig doit valoir 1 ou 10")
    if args.meas_mode == "diff_ab" and args.trigger_source != "ext":
        raise ValueError("--meas-mode diff_ab requiert --trigger-source ext")

    pts_all = np.load(args.plaintexts)
    if args.pt_offset < 0:
        raise ValueError("--pt-offset doit etre >= 0")
    end = args.pt_offset + args.n_traces
    if len(pts_all) < end:
        raise ValueError(
            f"{args.plaintexts} contient {len(pts_all)} plaintexts, besoin de {end} (offset inclus)"
        )
    pts = pts_all[args.pt_offset:end]

    _ensure_pico_runtime()

    from picosdk.ps5000a import ps5000a as ps
    from picosdk.functions import assert_pico_ok, mV2adc

    PICO_POWER_SUPPLY_NOT_CONNECTED = 286
    PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

    chandle = ctypes.c_int16()
    resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

    status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
        assert_pico_ok(ps.ps5000aChangePowerSource(chandle, status))
    else:
        assert_pico_ok(status)

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))

    ch_a = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    ch_b = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    if args.trigger_source == "ext":
        ch_trig = ps.PS5000A_CHANNEL["PS5000A_EXTERNAL"]
    else:
        ch_trig = ch_a
    coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
    range_trig = ps.PS5000A_RANGE["PS5000A_500MV"]
    range_meas_a = ps.PS5000A_RANGE[args.meas_range_a]
    range_meas_b = ps.PS5000A_RANGE[args.meas_range]
    range_mv_map = {
        "PS5000A_10MV": 10.0,
        "PS5000A_20MV": 20.0,
        "PS5000A_50MV": 50.0,
        "PS5000A_100MV": 100.0,
        "PS5000A_200MV": 200.0,
        "PS5000A_500MV": 500.0,
        "PS5000A_1V": 1000.0,
        "PS5000A_2V": 2000.0,
        "PS5000A_5V": 5000.0,
        "PS5000A_10V": 10000.0,
        "PS5000A_20V": 20000.0,
    }
    meas_range_a_mv = range_mv_map[args.meas_range_a]
    meas_range_b_mv = range_mv_map[args.meas_range]

    if args.trigger_source == "cha":
        assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_trig, 1, coupling, range_trig, 0.0))
    else:
        # Trigger EXTERNAL: canal A active seulement si mesure diff_ab.
        en_a = 1 if args.meas_mode == "diff_ab" else 0
        assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_a, en_a, coupling, range_meas_a, 0.0))
    assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_b, 1, coupling, range_meas_b, 0.0))

    trig_mv_bnc = max(1, int(args.trig_mv_probe) // int(args.probe_att_trig))
    if args.trigger_source == "ext":
        # EXT n'utilise pas la meme conversion d'echelle qu'un canal analogique standard.
        # On expose directement un seuil en ADC pour calibration pratique.
        trig_adc = int(args.ext_threshold_adc)
    else:
        trig_adc = mV2adc(trig_mv_bnc, range_trig, max_adc)
    trig_dir = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
    if args.debug_trigger:
        print(
            f"[DBG] trigger_source={args.trigger_source} trig_adc={int(trig_adc)} "
            f"(mv_probe={args.trig_mv_probe}, att={args.probe_att_trig}) "
            f"meas_mode={args.meas_mode} meas_range_a={args.meas_range_a} meas_range_b={args.meas_range}",
            flush=True,
        )
    assert_pico_ok(ps.ps5000aSetSimpleTrigger(chandle, 1, ch_trig, trig_adc, trig_dir, 0, 0))

    dt_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        chandle,
        args.timebase,
        args.num_samples,
        ctypes.byref(dt_ns),
        ctypes.byref(returned),
        0,
    ))

    max_samp_seg = ctypes.c_int32(0)
    assert_pico_ok(ps.ps5000aMemorySegments(chandle, args.n_traces, ctypes.byref(max_samp_seg)))
    if max_samp_seg.value < args.num_samples:
        ps.ps5000aCloseUnit(chandle)
        raise RuntimeError(
            f"Memoire insuffisante: {max_samp_seg.value} samples/segment max, besoin {args.num_samples}."
        )

    assert_pico_ok(ps.ps5000aSetNoOfCaptures(chandle, args.n_traces))

    post_trigger = args.num_samples - args.pre_trigger
    raw_adc_b = np.zeros((args.n_traces, args.num_samples), dtype=np.int16)
    raw_adc_a = np.zeros((args.n_traces, args.num_samples), dtype=np.int16) if args.meas_mode == "diff_ab" else None

    assert_pico_ok(ps.ps5000aRunBlock(
        chandle,
        args.pre_trigger,
        post_trigger,
        args.timebase,
        None,
        0,
        None,
        None,
    ))

    print("=" * 62)
    print(f"SCOPE ARME - {args.n_traces} captures en attente")
    print("Appuie sur RESET STM32 maintenant")
    print("=" * 62)

    expected_s = args.n_traces * 0.011
    timeout_s = float(args.timeout_s) if args.timeout_s is not None else (args.n_traces * 0.02 + 15.0)
    min_expected = max(1.0, expected_s * float(args.min_duration_factor))

    ready = ctypes.c_int16(0)
    deadline = time.time() + timeout_s
    t0 = time.time()
    next_status = time.time() + max(0.5, float(args.status_every_s))
    while not ready.value:
        if time.time() >= deadline:
            ps.ps5000aStop(chandle)
            ps.ps5000aCloseUnit(chandle)
            raise RuntimeError(f"Timeout acquisition ({timeout_s:.1f}s)")
        now = time.time()
        if now >= next_status:
            elapsed_wait = now - t0
            print(f"[WAIT] en cours... {elapsed_wait:.1f}s/{timeout_s:.1f}s", flush=True)
            next_status = now + max(0.5, float(args.status_every_s))
        assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))
        time.sleep(0.2)

    elapsed = time.time() - t0
    if elapsed < min_expected:
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)
        raise RuntimeError(
            f"Capture trop rapide ({elapsed:.1f}s < {min_expected:.1f}s): faux trigger probable."
        )

    overflows = np.zeros(args.n_traces, dtype=np.int16)
    for seg in range(args.n_traces):
        ptr_b = raw_adc_b[seg].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_b, ptr_b, None, args.num_samples, seg, 0))
        if raw_adc_a is not None:
            ptr_a = raw_adc_a[seg].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_a, ptr_a, None, args.num_samples, seg, 0))

        count = ctypes.c_int32(args.num_samples)
        ov = ctypes.c_int16(0)
        assert_pico_ok(ps.ps5000aGetValues(chandle, 0, ctypes.byref(count), 1, 0, seg, ctypes.byref(ov)))
        overflows[seg] = ov.value

    ps.ps5000aStop(chandle)
    ps.ps5000aCloseUnit(chandle)

    traces_b = raw_adc_b.astype(np.float32) * (meas_range_b_mv / float(max_adc.value))
    if raw_adc_a is not None:
        traces_a = raw_adc_a.astype(np.float32) * (meas_range_a_mv / float(max_adc.value))
        traces_diff = traces_a - traces_b
        traces = traces_diff
    else:
        traces_a = None
        traces_diff = None
        traces = traces_b

    save = {
        "traces": traces,
        "plaintexts": pts,
        "overflows": overflows,
        "meta_timebase": np.int32(args.timebase),
        "meta_dt_ns": np.float32(dt_ns.value),
        "meta_pt_offset": np.int32(args.pt_offset),
        "meta_meas_mode": np.array(args.meas_mode),
        "meta_meas_range_b": np.array(args.meas_range),
    }
    if traces_a is not None:
        save["traces_a"] = traces_a
        save["traces_b"] = traces_b
        save["traces_diff"] = traces_diff
        save["meta_meas_range_a"] = np.array(args.meas_range_a)
    if args.key_hex.strip():
        save["key"] = _parse_key_hex(args.key_hex)

    np.savez(args.output, **save)

    print(f"[OK] Dataset: {args.output}")
    print(f"[OK] Shape  : {traces.shape[0]} x {traces.shape[1]}")
    print(f"[OK] dt_ns  : {dt_ns.value:.3f}")
    print(f"[OK] overflow segments: {(overflows != 0).sum()}")


if __name__ == "__main__":
    main()
