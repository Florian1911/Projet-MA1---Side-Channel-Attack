#!/usr/bin/env python3
"""Smoke test PicoScope 5000A: ouvre le scope, capture 1 bloc, sauvegarde un plot."""

import ctypes
import os
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
        raise RuntimeError("libpicoipp.so introuvable (PicoSDK non installe ou chemin absent).")

    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir

    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    _ensure_pico_runtime()

    from picosdk.ps5000a import ps5000a as ps
    from picosdk.functions import assert_pico_ok, adc2mV

    PICO_POWER_SUPPLY_NOT_CONNECTED = 286
    PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

    chandle = ctypes.c_int16()
    resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

    status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
        assert_pico_ok(ps.ps5000aChangePowerSource(chandle, status))
    else:
        assert_pico_ok(status)

    ch_a = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    rng = ps.PS5000A_RANGE["PS5000A_500MV"]
    coupling = ps.PS5000A_COUPLING["PS5000A_DC"]

    assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_a, 1, coupling, rng, 0.0))

    pre_trigger = 0
    post_trigger = 100000
    max_samples = pre_trigger + post_trigger
    timebase = 100

    dt_ns = ctypes.c_float()
    returned_max = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        chandle,
        timebase,
        max_samples,
        ctypes.byref(dt_ns),
        ctypes.byref(returned_max),
        0,
    ))

    assert_pico_ok(ps.ps5000aRunBlock(chandle, pre_trigger, post_trigger, timebase, None, 0, None, None))

    ready = ctypes.c_int16(0)
    while not ready.value:
        assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))

    buffer_a = (ctypes.c_int16 * max_samples)()
    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_a, ctypes.byref(buffer_a), None, max_samples, 0, 0))

    cmax = ctypes.c_int32(max_samples)
    assert_pico_ok(ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmax), 0, 0, 0, None))

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))

    adc = np.array(buffer_a[: cmax.value], dtype=np.int16)
    mv = adc2mV(adc, rng, max_adc)

    plt.figure(figsize=(11, 4))
    plt.plot(mv, linewidth=0.8)
    plt.title("PicoScope 5000A - smoke test (canal A)")
    plt.xlabel("Sample")
    plt.ylabel("mV")
    plt.tight_layout()
    plt.savefig("smoke_test_pico5000a.png", dpi=120)

    ps.ps5000aStop(chandle)
    ps.ps5000aCloseUnit(chandle)

    print(f"[OK] Trace capturee: {cmax.value} samples")
    print(f"[OK] dt_ns: {dt_ns.value:.3f}")
    print("[OK] Figure: smoke_test_pico5000a.png")


if __name__ == "__main__":
    main()
