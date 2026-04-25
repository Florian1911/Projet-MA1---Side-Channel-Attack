import os
import time
import ctypes
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

# -----------------------
# CONFIG UTILISATEUR
# -----------------------
OUT_DIR = Path("dataset_aes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRACES = 20000

SERIAL_PORT = None  # None = auto-detect (/dev/ttyACM* puis /dev/ttyUSB*)
BAUD = 115200
SERIAL_TIMEOUT = 2.0

# Pico channels
PROBE_ATTENUATION_A = 10  # PB8 en x10
PROBE_ATTENUATION_B = 1   # shunt en x1 (IMPORTANT)
INVERT_CH_B = True        # selon ton sens

CH_A_RANGE = ps.PS5000A_RANGE["PS5000A_500MV"]   # PB8 3.3V via x10 -> 330mV au BNC
CH_B_RANGE = ps.PS5000A_RANGE["PS5000A_200MV"]   # shunt ~50-80mV DC

# Acquisition
NUM_SAMPLES = 50000
TIMEBASE = 5              # chez toi: ~32 ns => ~1.6 ms pour 50k
PRETRIG_FRAC = 0.2

TRIGGER_A_THRESHOLD_MV_PROBE = 2000  # seuil sur la sonde (mV "vu" côté probe)
R_SHUNT_OHM = 1.0

# -----------------------
# PICO HELPERS
# -----------------------
PICO_POWER_SUPPLY_NOT_CONNECTED = 286
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282


def open_unit() -> ctypes.c_int16:
    chandle = ctypes.c_int16()
    status = ps.ps5000aOpenUnit(
        ctypes.byref(chandle),
        None,
        ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"],
    )
    if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
        status = ps.ps5000aChangePowerSource(chandle, status)
    assert_pico_ok(status)
    return chandle


def configure_pico(chandle: ctypes.c_int16):
    ch_a = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    ch_b = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]

    assert_pico_ok(ps.ps5000aSetChannel(
        chandle, ch_a, 1,
        ps.PS5000A_COUPLING["PS5000A_DC"],
        CH_A_RANGE, 0
    ))
    assert_pico_ok(ps.ps5000aSetChannel(
        chandle, ch_b, 1,
        ps.PS5000A_COUPLING["PS5000A_DC"],
        CH_B_RANGE, 0
    ))

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))

    # Trigger on CH A rising
    threshold_bnc_mv = max(1, int(TRIGGER_A_THRESHOLD_MV_PROBE / PROBE_ATTENUATION_A))
    threshold_adc = mV2adc(threshold_bnc_mv, CH_A_RANGE, max_adc)

    assert_pico_ok(ps.ps5000aSetSimpleTrigger(
        chandle, 1, ch_a, threshold_adc,
        ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
        0, 2000
    ))

    # Timebase
    dt_ns = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        chandle, TIMEBASE, NUM_SAMPLES,
        ctypes.byref(dt_ns), ctypes.byref(returned), 0
    ))

    return ch_a, ch_b, max_adc, float(dt_ns.value)


def arm_capture(chandle, ch_a, ch_b):
    pre = int(NUM_SAMPLES * PRETRIG_FRAC)
    post = NUM_SAMPLES - pre

    buffer_a = (ctypes.c_int16 * NUM_SAMPLES)()
    buffer_b = (ctypes.c_int16 * NUM_SAMPLES)()

    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_a, ctypes.byref(buffer_a), None, NUM_SAMPLES, 0, 0))
    assert_pico_ok(ps.ps5000aSetDataBuffers(chandle, ch_b, ctypes.byref(buffer_b), None, NUM_SAMPLES, 0, 0))

    assert_pico_ok(ps.ps5000aRunBlock(chandle, pre, post, TIMEBASE, None, 0, None, None))
    return buffer_a, buffer_b


def wait_and_read(chandle, buffer_a, buffer_b, ch_a, ch_b, max_adc):
    ready = ctypes.c_int16(0)
    while not ready.value:
        assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))

    cmax = ctypes.c_int32(NUM_SAMPLES)
    overflow = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmax), 1, 0, 0, ctypes.byref(overflow)))

    adc_a = np.array(buffer_a[:cmax.value], dtype=np.int32)
    adc_b = np.array(buffer_b[:cmax.value], dtype=np.int32)

    mv_a = np.asarray(adc2mV(adc_a.tolist(), CH_A_RANGE, max_adc), dtype=np.float32) * PROBE_ATTENUATION_A
    mv_b = np.asarray(adc2mV(adc_b.tolist(), CH_B_RANGE, max_adc), dtype=np.float32) * PROBE_ATTENUATION_B
    if INVERT_CH_B:
        mv_b = -mv_b

    # DC/AC
    mv_b_dc = mv_b
    mv_b_ac = mv_b - mv_b.mean()

    return mv_a, mv_b_dc, mv_b_ac, int(overflow.value)


# -----------------------
# UART HELPERS
# -----------------------
def read_exact(ser: serial.Serial, n: int) -> bytes:
    data = ser.read(n)
    if len(data) != n:
        raise RuntimeError(f"UART timeout: expected {n} bytes, got {len(data)}")
    return data


def choose_serial_port(configured_port: str | None) -> str:
    if configured_port:
        return configured_port

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError(
            "Aucun port serie detecte.\n"
            "Branche la carte STM32 puis verifie avec: ls -l /dev/ttyACM* /dev/ttyUSB*"
        )

    devices = [p.device for p in ports]

    for dev in devices:
        if "/dev/ttyACM" in dev:
            return dev
    for dev in devices:
        if "/dev/ttyUSB" in dev:
            return dev

    return devices[0]


# -----------------------
# MAIN ACQ LOOP
# -----------------------
def main():
    port = choose_serial_port(SERIAL_PORT)
    print(f"UART port: {port}")

    # UART
    ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Pico
    chandle = open_unit()
    ch_a, ch_b, max_adc, dt_ns = configure_pico(chandle)

    total_s = NUM_SAMPLES * (dt_ns * 1e-9)
    print(f"Pico dt={dt_ns:.1f} ns, window={total_s*1e3:.3f} ms, samples={NUM_SAMPLES}")

    # Save global metadata
    meta = {
        "n_traces": N_TRACES,
        "dt_ns": dt_ns,
        "num_samples": NUM_SAMPLES,
        "timebase": TIMEBASE,
        "pretrig_frac": PRETRIG_FRAC,
        "r_shunt_ohm": R_SHUNT_OHM,
        "probe_att_a": PROBE_ATTENUATION_A,
        "probe_att_b": PROBE_ATTENUATION_B,
        "invert_ch_b": INVERT_CH_B,
    }
    np.save(OUT_DIR / "meta.npy", meta, allow_pickle=True)

    try:
        for i in range(N_TRACES):
            # 1) random plaintext
            pt = os.urandom(16)

            # 2) Arm Pico BEFORE sending PT (important)
            buffer_a, buffer_b = arm_capture(chandle, ch_a, ch_b)

            # 3) Send PT to STM32
            ser.write(pt)

            # 4) Read CT from STM32 (16 bytes)
            ct = read_exact(ser, 16)

            # 5) Fetch Pico samples
            mv_a, mv_b_dc, mv_b_ac, overflow = wait_and_read(chandle, buffer_a, buffer_b, ch_a, ch_b, max_adc)

            # 6) Save (per trace)
            np.save(OUT_DIR / f"pt_{i:06d}.npy", np.frombuffer(pt, dtype=np.uint8))
            np.save(OUT_DIR / f"ct_{i:06d}.npy", np.frombuffer(ct, dtype=np.uint8))
            np.save(OUT_DIR / f"chA_{i:06d}.npy", mv_a)
            np.save(OUT_DIR / f"chBdc_{i:06d}.npy", mv_b_dc)
            np.save(OUT_DIR / f"chBac_{i:06d}.npy", mv_b_ac)

            if i % 200 == 0:
                print(f"[{i}/{N_TRACES}] overflow={overflow}  CHBdc mean={mv_b_dc.mean():.2f} mV  CHBac std={mv_b_ac.std():.2f} mV")

    finally:
        try:
            ps.ps5000aStop(chandle)
            ps.ps5000aCloseUnit(chandle)
        except Exception:
            pass
        ser.close()


if __name__ == "__main__":
    main()
