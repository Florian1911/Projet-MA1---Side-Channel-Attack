import os
import time
import json
import ctypes
import argparse
from pathlib import Path

import numpy as np
import serial

# This script assumes a PicoScope 5000a series oscilloscope is used.
# Make sure the picosdk is installed and the required libraries are found.
def _ensure_pico_runtime() -> None:
    """Ensure the PicoScope library can be found by ctypes."""
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
            "libpicoipp.so not found.\n"
            f"Searched paths:\n{searched}"
        )
    # Pre-load the library to satisfy dependencies.
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)

_ensure_pico_runtime()

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

# =========================
#  Default Configuration
# =========================
# -- Serial Port Config --
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
UART_TIMEOUT_S = 2.0

# -- Acquisition Config --
N_TRACES = 5000
NUM_SAMPLES = 2000
OUTPUT_FILE = "cpa_dataset.npz"

# -- PicoScope Config --
PICO_CH_TRIG = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
PICO_CH_MEAS = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
PICO_COUPLING = ps.PS5000A_COUPLING["PS5000A_DC"]
PICO_RANGE_TRIG_MV = 500
PICO_RANGE_MEAS_MV = 200
TIMEBASE = 8
PRE_TRIGGER_SAMPLES = 200
TRIGGER_THRESHOLD_MV = 50

# The key used in the target firmware (for verification)
KNOWN_KEY = np.array([
    0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
    0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C
], dtype=np.uint8)


def mv_to_range_id(mv: int) -> int:
    """Converts a voltage in mV to a PicoScope range identifier."""
    lut = {
        10: ps.PS5000A_RANGE["PS5000A_10MV"], 20: ps.PS5000A_RANGE["PS5000A_20MV"],
        50: ps.PS5000A_RANGE["PS5000A_50MV"], 100: ps.PS5000A_RANGE["PS5000A_100MV"],
        200: ps.PS5000A_RANGE["PS5000A_200MV"], 500: ps.PS5000A_RANGE["PS5000A_500MV"],
        1000: ps.PS5000A_RANGE["PS5000A_1V"], 2000: ps.PS5000A_RANGE["PS5000A_2V"],
        5000: ps.PS5000A_RANGE["PS5000A_5V"], 10000: ps.PS5000A_RANGE["PS5000A_10V"],
        20000: ps.PS5000A_RANGE["PS5000A_20V"],
    }
    if mv not in lut:
        raise ValueError(f"Unsupported scope range {mv} mV. Allowed: {sorted(lut.keys())}")
    return lut[mv]

class PicoCapture:
    """A class to handle the PicoScope 5000a series."""
    def __init__(self, num_samples, timebase, ch_meas_range_id, ch_trig_range_id, trigger_mv, pre_trigger_samples):
        print("[PICO] Opening PicoScope...")
        self.chandle = ctypes.c_int16()
        self.num_samples = num_samples
        self.pre_trigger_samples = pre_trigger_samples
        self.post_trigger_samples = num_samples - pre_trigger_samples
        self.timebase = timebase

        resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
        status = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, resolution)
        if status in [282, 286]: # Handle power-related issues
            status = ps.ps5000aChangePowerSource(self.chandle, status)
        assert_pico_ok(status)
        print("[PICO] PicoScope opened.")

        assert_pico_ok(ps.ps5000aSetChannel(self.chandle, PICO_CH_TRIG, 1, PICO_COUPLING, ch_trig_range_id, 0))
        assert_pico_ok(ps.ps5000aSetChannel(self.chandle, PICO_CH_MEAS, 1, PICO_COUPLING, ch_meas_range_id, 0))

        self.max_adc = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.max_adc)))
        threshold_adc = mV2adc(trigger_mv, ch_trig_range_id, self.max_adc)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.chandle, 1, PICO_CH_TRIG, threshold_adc,
            ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"], 0, 1000
        ))

        self.time_interval_ns = ctypes.c_float()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.chandle, self.timebase, self.num_samples, ctypes.byref(self.time_interval_ns), ctypes.byref(ctypes.c_int32()), 0
        ))

        self.buffer_meas = (ctypes.c_int16 * self.num_samples)()
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            self.chandle, PICO_CH_MEAS, ctypes.byref(self.buffer_meas), None, self.num_samples, 0, 0
        ))
        print("[PICO] Setup complete.")

    def arm(self):
        """Arm the oscilloscope for capture."""
        assert_pico_ok(ps.ps5000aRunBlock(
            self.chandle, self.pre_trigger_samples, self.post_trigger_samples, self.timebase, None, 0, None, None
        ))

    def wait_and_get_trace(self) -> np.ndarray:
        """Wait for capture to complete and return the raw ADC trace."""
        ready = ctypes.c_int16(0)
        while not ready.value:
            assert_pico_ok(ps.ps5000aIsReady(self.chandle, ctypes.byref(ready)))

        num_captured = ctypes.c_int32(self.num_samples)
        assert_pico_ok(ps.ps5000aGetValues(
            self.chandle, 0, ctypes.byref(num_captured), 1, 0, 0, ctypes.byref(ctypes.c_int16())
        ))

        return np.array(self.buffer_meas, dtype=np.int16)

    def close(self):
        """Close the connection to the oscilloscope."""
        ps.ps5000aStop(self.chandle)
        ps.ps5000aCloseUnit(self.chandle)
        print("[PICO] PicoScope closed.")


def read_ciphertext_from_target(ser: serial.Serial) -> bytes:
    """Robustly read the 16-byte ciphertext preceded by 'C'."""
    deadline = time.time() + UART_TIMEOUT_S
    while time.time() < deadline:
        header = ser.read(1)
        if not header:
            continue
        if header == b'C':
            ciphertext = ser.read(16)
            if len(ciphertext) != 16:
                raise IOError(f"Incomplete ciphertext. Got {len(ciphertext)} bytes.")
            return ciphertext
        # Optional: Handle unexpected printable lines from target for debugging
        if header in (b'\r', b'\n'):
            continue
        if 32 <= header[0] <= 126:
            line = (header + ser.readline()).decode(errors='replace').strip()
            if line:
                print(f"STM32: {line}")
            continue
    raise IOError("Timeout waiting for ciphertext from target.")

def acquire_one_trace(pico: PicoCapture, ser: serial.Serial, plaintext: bytes) -> tuple[np.ndarray, bytes]:
    """Arm scope, send plaintext, and retrieve trace and ciphertext."""
    pico.arm()
    ser.write(b'P' + plaintext)
    ser.flush()
    trace = pico.wait_and_get_trace()
    ciphertext = read_ciphertext_from_target(ser)
    return trace, ciphertext

def main():
    ap = argparse.ArgumentParser(description="CPA Data Acquisition Script")
    ap.add_argument("--n-traces", type=int, default=N_TRACES, help="Number of traces to capture")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples per trace")
    ap.add_argument("--output", default=OUTPUT_FILE, help="Output file name (.npz)")
    ap.add_argument("--serial-port", default=SERIAL_PORT, help="Serial port of the target")
    ap.add_argument("--baud", type=int, default=BAUD, help="Serial baud rate")
    ap.add_argument("--meas-range-mv", type=int, default=PICO_RANGE_MEAS_MV, help="PicoScope measurement range in mV")
    ap.add_argument("--trig-range-mv", type=int, default=PICO_RANGE_TRIG_MV, help="PicoScope trigger range in mV")
    ap.add_argument("--trig-mv", type=int, default=TRIGGER_THRESHOLD_MV, help="Trigger threshold in mV")
    args = ap.parse_args()

    # Initialize arrays to store data
    traces = np.zeros((args.n_traces, args.num_samples), dtype=np.int16)
    plaintexts = np.zeros((args.n_traces, 16), dtype=np.uint8)
    ciphertexts = np.zeros((args.n_traces, 16), dtype=np.uint8)

    pico = None
    target_serial = None
    try:
        # Initialize hardware
        pico = PicoCapture(
            num_samples=args.num_samples,
            timebase=TIMEBASE,
            ch_meas_range_id=mv_to_range_id(args.meas_range_mv),
            ch_trig_range_id=mv_to_range_id(args.trig_range_mv),
            trigger_mv=args.trig_mv,
            pre_trigger_samples=PRE_TRIGGER_SAMPLES,
        )
        target_serial = serial.Serial(args.serial_port, args.baud, timeout=UART_TIMEOUT_S)
        time.sleep(2) # Wait for device to reset
        startup_msg = target_serial.read_until(b"READY\r\n", timeout=5)
        if b"READY" in startup_msg:
            print("[UART] Target is ready.")
        else:
            print("[UART] Warning: Did not receive READY message from target.")

        print(f"--- Starting acquisition of {args.n_traces} traces ---")
        for i in range(args.n_traces):
            pt_bytes = os.urandom(16)
            trace, ct_bytes = acquire_one_trace(pico, target_serial, pt_bytes)

            traces[i] = trace
            plaintexts[i] = np.frombuffer(pt_bytes, dtype=np.uint8)
            ciphertexts[i] = np.frombuffer(ct_bytes, dtype=np.uint8)

            if (i + 1) % 100 == 0:
                print(f"Captured trace {i+1}/{args.n_traces}")
        print("--- Acquisition complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if pico:
            pico.close()
        if target_serial:
            target_serial.close()
            print("[UART] Serial port closed.")

    # Save data to .npz file
    if np.any(traces):
        print(f"Saving data to {args.output}...")
        np.savez(
            args.output,
            traces=traces,
            plaintexts=plaintexts,
            ciphertexts=ciphertexts,
            key=KNOWN_KEY,
        )

        # Save metadata to .json file
        meta = {
            "n_traces": args.n_traces, "num_samples": args.num_samples,
            "timebase": TIMEBASE, "dt_ns": float(pico.time_interval_ns.value),
            "meas_range_mv": args.meas_range_mv, "trig_range_mv": args.trig_range_mv,
        }
        json_path = args.output.replace(".npz", ".json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print("Data and metadata saved.")
    else:
        print("No traces captured. Not saving data.")

if __name__ == "__main__":
    main()
