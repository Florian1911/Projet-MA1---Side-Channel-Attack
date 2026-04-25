import os
import json
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
        searched = "\n".join(f" - {p}" for p in candidates)
        raise RuntimeError(
            "libpicoipp.so introuvable.\n"
            f"Chemins verifiés:\n{searched}"
        )

    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir

    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


_ensure_pico_runtime()

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

# =========================
# CONFIG
# =========================
N_TRACES = 1000
NUM_SAMPLES = 2000
OUTPUT_FILE = "dataset_aes_sca_no_uart.npz"

PICO_POWER_SUPPLY_NOT_CONNECTED = 286
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

PICO_CH_TRIG = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]  # PB8 trigger
PICO_CH_MEAS = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]  # shunt
PICO_COUPLING = ps.PS5000A_COUPLING["PS5000A_DC"]

PROBE_ATTENUATION_TRIG = 10
PROBE_ATTENUATION_MEAS = 1
PICO_RANGE_TRIG = ps.PS5000A_RANGE["PS5000A_500MV"]
PICO_RANGE_MEAS = ps.PS5000A_RANGE["PS5000A_200MV"]

TIMEBASE = 8
PRE_TRIGGER_SAMPLES = 200
POST_TRIGGER_SAMPLES = NUM_SAMPLES - PRE_TRIGGER_SAMPLES
TRIGGER_THRESHOLD_MV_PROBE = 1000
TRIGGER_DIR = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
AUTO_TRIGGER_MS = 2000

PRNG_SEED = 0x12345678
AUTO_ALIGN = True
ALIGN_BYTE = 0
ALIGN_KEY = 0x2B
ALIGN_MAX_SHIFT = 64
KEY = np.array([
    0xA4, 0x1F, 0xC7, 0x6B, 0xD2, 0x09, 0x5E, 0x93,
    0x7A, 0xE1, 0x4C, 0xB8, 0x2D, 0xF0, 0x66, 0x19
], dtype=np.uint8)

AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)


class XorShift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_u8(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state & 0xFF

    def next_block16(self) -> np.ndarray:
        return np.array([self.next_u8() for _ in range(16)], dtype=np.uint8)


class PicoCapture:
    def __init__(self):
        print("[PICO] ouverture...", flush=True)
        self.chandle = ctypes.c_int16()
        resolution_dict = getattr(ps, "PS5000A_RESOLUTION", None)
        if resolution_dict is None:
            resolution_dict = ps.PS5000A_DEVICE_RESOLUTION
        status = ps.ps5000aOpenUnit(
            ctypes.byref(self.chandle),
            None,
            resolution_dict["PS5000A_DR_12BIT"],
        )
        if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
            status = ps.ps5000aChangePowerSource(self.chandle, status)
        assert_pico_ok(status)

        assert_pico_ok(ps.ps5000aSetChannel(
            self.chandle, PICO_CH_TRIG, 1, PICO_COUPLING, PICO_RANGE_TRIG, 0
        ))
        assert_pico_ok(ps.ps5000aSetChannel(
            self.chandle, PICO_CH_MEAS, 1, PICO_COUPLING, PICO_RANGE_MEAS, 0
        ))

        self.max_adc = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.max_adc)))

        threshold_bnc_mv = max(1, int(TRIGGER_THRESHOLD_MV_PROBE / PROBE_ATTENUATION_TRIG))
        threshold_adc = mV2adc(threshold_bnc_mv, PICO_RANGE_TRIG, self.max_adc)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.chandle, 1, PICO_CH_TRIG, threshold_adc, TRIGGER_DIR, 0, AUTO_TRIGGER_MS
        ))

        self.dt_ns = ctypes.c_float()
        returned = ctypes.c_int32()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.chandle, TIMEBASE, NUM_SAMPLES,
            ctypes.byref(self.dt_ns), ctypes.byref(returned), 0
        ))

        self.buffer_meas = (ctypes.c_int16 * NUM_SAMPLES)()
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            self.chandle, PICO_CH_MEAS, ctypes.byref(self.buffer_meas), None, NUM_SAMPLES, 0, 0
        ))

        print(f"[PICO] prêt dt_ns={self.dt_ns.value:.3f}", flush=True)

    def capture_one(self) -> tuple[np.ndarray, int]:
        assert_pico_ok(ps.ps5000aRunBlock(
            self.chandle, PRE_TRIGGER_SAMPLES, POST_TRIGGER_SAMPLES, TIMEBASE, None, 0, None, None
        ))

        ready = ctypes.c_int16(0)
        t0 = time.time()
        while not ready.value:
            if time.time() - t0 > 2.0:
                raise RuntimeError("Timeout Pico: trigger non détecté")
            assert_pico_ok(ps.ps5000aIsReady(self.chandle, ctypes.byref(ready)))

        num = ctypes.c_int32(NUM_SAMPLES)
        overflow = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aGetValues(
            self.chandle, 0, ctypes.byref(num), 1, 0, 0, ctypes.byref(overflow)
        ))

        adc = np.array(self.buffer_meas[:num.value], dtype=np.int32)
        trace = np.asarray(adc2mV(adc.tolist(), PICO_RANGE_MEAS, self.max_adc), dtype=np.float32)
        trace *= PROBE_ATTENUATION_MEAS

        if num.value != NUM_SAMPLES:
            padded = np.zeros(NUM_SAMPLES, dtype=np.float32)
            padded[:num.value] = trace
            trace = padded

        return trace, int(overflow.value)

    def close(self):
        try:
            ps.ps5000aStop(self.chandle)
        finally:
            ps.ps5000aCloseUnit(self.chandle)


def compute_label(pt: np.ndarray, key: np.ndarray, byte_idx: int = 0) -> int:
    return int(AES_SBOX[int(pt[byte_idx] ^ key[byte_idx])])


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def estimate_shift_with_cpa(
    traces: np.ndarray, plaintexts: np.ndarray, key_guess: int, byte_idx: int, max_shift: int
) -> tuple[int, float]:
    proc = center_and_detrend(traces)
    best_shift = 0
    best_corr = -1.0
    n = proc.shape[0]
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            tr = proc[: n - s]
            pt = plaintexts[s:]
        else:
            tr = proc[-s:]
            pt = plaintexts[: n + s]
        if tr.shape[0] < 50:
            continue

        tc = tr - tr.mean(axis=0, keepdims=True)
        tstd = tr.std(axis=0, ddof=1) + 1e-15
        hyp = AES_SBOX[np.bitwise_xor(pt[:, byte_idx], key_guess)].astype(np.uint8)
        hw = np.unpackbits(hyp[:, None], axis=1).sum(axis=1).astype(np.float64)
        hc = hw - hw.mean()
        hstd = hw.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((tr.shape[0] - 1) * hstd * tstd)
        score = float(np.max(np.abs(corr)))
        if score > best_corr:
            best_corr = score
            best_shift = s
    return best_shift, best_corr


def apply_shift(
    traces: np.ndarray, plaintexts: np.ndarray, labels: np.ndarray, overflows: np.ndarray, shift: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = traces.shape[0]
    if shift == 0:
        return traces, plaintexts, labels, overflows
    if shift > 0:
        # plaintext i correspond a trace i-shift
        return traces[: n - shift], plaintexts[shift:], labels[shift:], overflows[: n - shift]
    # shift < 0
    s = -shift
    return traces[s:], plaintexts[: n - s], labels[: n - s], overflows[s:]


def main():
    global N_TRACES, OUTPUT_FILE, NUM_SAMPLES, PRE_TRIGGER_SAMPLES, POST_TRIGGER_SAMPLES
    global TIMEBASE, TRIGGER_THRESHOLD_MV_PROBE

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traces", type=int, default=N_TRACES)
    ap.add_argument("--output", type=str, default=OUTPUT_FILE)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--timebase", type=int, default=TIMEBASE)
    ap.add_argument("--pre-trigger", type=int, default=PRE_TRIGGER_SAMPLES)
    ap.add_argument("--trig-mv", type=int, default=TRIGGER_THRESHOLD_MV_PROBE)
    ap.add_argument("--auto-align", type=int, default=1, help="1=on, 0=off")
    args = ap.parse_args()

    NUM_SAMPLES = int(args.num_samples)
    TIMEBASE = int(args.timebase)
    PRE_TRIGGER_SAMPLES = int(args.pre_trigger)
    POST_TRIGGER_SAMPLES = int(NUM_SAMPLES - PRE_TRIGGER_SAMPLES)
    if POST_TRIGGER_SAMPLES <= 0:
        raise ValueError("num-samples must be > pre-trigger")
    TRIGGER_THRESHOLD_MV_PROBE = int(args.trig_mv)

    N_TRACES = int(args.n_traces)
    OUTPUT_FILE = str(args.output)
    n_traces = int(args.n_traces)
    output_file = args.output
    auto_align = bool(args.auto_align)

    print("[MAIN] acquisition no-UART", flush=True)
    pico = PicoCapture()
    prng = XorShift32(PRNG_SEED)

    traces = np.zeros((n_traces, NUM_SAMPLES), dtype=np.float32)
    plaintexts = np.zeros((n_traces, 16), dtype=np.uint8)
    labels = np.zeros((n_traces,), dtype=np.uint8)
    overflows = np.zeros((n_traces,), dtype=np.int16)

    try:
        for i in range(n_traces):
            pt = prng.next_block16()
            trace, overflow = pico.capture_one()

            traces[i] = trace
            plaintexts[i] = pt
            labels[i] = compute_label(pt, KEY, byte_idx=0)
            overflows[i] = overflow

            if (i + 1) <= 3 or (i + 1) % 100 == 0:
                print(f"{i+1}/{n_traces} | overflow={overflow}", flush=True)
    finally:
        pico.close()

    align_shift = 0
    align_score = 0.0
    if AUTO_ALIGN and auto_align:
        align_shift, align_score = estimate_shift_with_cpa(
            traces, plaintexts, ALIGN_KEY, ALIGN_BYTE, ALIGN_MAX_SHIFT
        )
        if align_shift != 0:
            print(
                f"[ALIGN] shift estimé={align_shift} (score={align_score:.6f}) -> correction appliquée",
                flush=True,
            )
            traces, plaintexts, labels, overflows = apply_shift(
                traces, plaintexts, labels, overflows, align_shift
            )
        else:
            print(f"[ALIGN] shift estimé=0 (score={align_score:.6f})", flush=True)

    np.savez(
        output_file,
        traces=traces,
        plaintexts=plaintexts,
        labels=labels,
        key=KEY,
        prng_seed=np.uint32(PRNG_SEED),
        overflows=overflows,
        align_shift=np.int32(align_shift),
        align_score=np.float32(align_score),
    )

    meta = {
        "mode": "no_uart_prng_deterministic",
        "n_traces": int(n_traces),
        "num_samples": int(NUM_SAMPLES),
        "timebase": int(TIMEBASE),
        "pre_trigger_samples": int(PRE_TRIGGER_SAMPLES),
        "trigger_threshold_mv_probe": int(TRIGGER_THRESHOLD_MV_PROBE),
        "probe_att_trig": int(PROBE_ATTENUATION_TRIG),
        "probe_att_meas": int(PROBE_ATTENUATION_MEAS),
        "dt_ns": float(pico.dt_ns.value),
        "label_definition": "SBox(plaintext[0] xor key[0])",
        "prng": "xorshift32",
        "prng_seed": int(PRNG_SEED),
        "auto_align": bool(AUTO_ALIGN and auto_align),
        "align_key_guess": int(ALIGN_KEY),
        "align_byte": int(ALIGN_BYTE),
        "align_max_shift": int(ALIGN_MAX_SHIFT),
        "align_shift_applied": int(align_shift),
        "align_score": float(align_score),
        "n_traces_saved": int(traces.shape[0]),
    }

    with open(output_file.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset sauvegardé dans {output_file}", flush=True)


if __name__ == "__main__":
    main()
