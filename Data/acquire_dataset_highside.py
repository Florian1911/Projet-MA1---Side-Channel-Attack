"""
Acquisition AES SCA – mesure high-side différentielle, sync UART.

Câblage PicoScope :
  ChA  → côté 5V du shunt    (V_5V_rail / GND circuit)
  ChB  → côté carte du shunt (V_VCC_carte / GND circuit)
  Trace mesurée = ChA - ChB  ≈ tension aux bornes du shunt ∝ courant

Pas de trigger GPIO hardware. La carte STM32 envoie 'S' juste avant
de lancer l'AES ; côté PC l'armement du Pico est calé via ARM_DELAY_US.
"""

import os
import time
import json
import ctypes
import argparse
from pathlib import Path

import numpy as np
import serial


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
            f"Chemins vérifiés:\n{searched}"
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
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
UART_TIMEOUT_S = 2.0

N_TRACES = 1000
NUM_SAMPLES = 2000
OUTPUT_FILE = "dataset_aes_sca_highside.npz"

PICO_POWER_SUPPLY_NOT_CONNECTED = 286
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

PICO_CH_A = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]  # côté 5V – shunt (optionnel, monitoring)
PICO_CH_B = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]  # côté shunt – carte

# ChA : DC, grande range (monitoring alim, pas utilisé pour CPA)
PICO_COUPLING_A = ps.PS5000A_COUPLING["PS5000A_DC"]
PICO_RANGE_A    = ps.PS5000A_RANGE["PS5000A_5V"]

# ChB : DC couplé, low-side (shunt entre GND carte et GND circuit)
# Avec 2.4Ω et ~50mA → DC = ~120mV, variation AES ≈ ±10-20mV
# Range 200mV : résolution 0.05mV/LSB (50x mieux que high-side 5V)
PICO_COUPLING_B = ps.PS5000A_COUPLING["PS5000A_DC"]
PICO_RANGE_B    = ps.PS5000A_RANGE["PS5000A_200MV"]

PROBE_ATTENUATION_A = 1
PROBE_ATTENUATION_B = 1

TIMEBASE = 8               # ~48 ns/sample à 12 bits
PRE_TRIGGER_SAMPLES = 200  # échantillons avant le début estimé de l'AES

# Pas de trigger hardware : Pico capture dès l'armement.
# ARM_DELAY_US : délai entre l'armement du Pico et l'envoi du plaintext
# à calibrer pour que l'AES tombe dans la fenêtre.
# À 115200 baud, la transmission de 17 octets ('P'+PT) dure ~1,5 ms.
# 'S' est envoyé par la carte avant l'AES, soit +~87 µs supplémentaires.
# Exemple : ARM_DELAY_US = 0 (armement immédiat, la fenêtre commence dès l'envoi).
ARM_DELAY_US = 4700  # à calibrer : ~HAL_Delay_ms×1000 - latence_uart_us - marge

KEY = np.array([
    0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
    0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C
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


class PicoCapture:
    def __init__(self):
        print("[PICO] ouverture...", flush=True)
        self.chandle = ctypes.c_int16()
        resolution_dict = getattr(ps, "PS5000A_RESOLUTION", None) or ps.PS5000A_DEVICE_RESOLUTION
        status = ps.ps5000aOpenUnit(
            ctypes.byref(self.chandle),
            None,
            resolution_dict["PS5000A_DR_12BIT"],
        )
        if status == 3:
            raise RuntimeError("[PICO] PICO_NOT_FOUND: aucun PicoScope 5000A détecté.")
        if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
            status = ps.ps5000aChangePowerSource(self.chandle, status)
        assert_pico_ok(status)
        print("[PICO] ouvert", flush=True)

        # Configurer ChA (5V – shunt, DC, grande range)
        assert_pico_ok(ps.ps5000aSetChannel(
            self.chandle, PICO_CH_A, 1, PICO_COUPLING_A, PICO_RANGE_A, 0,
        ))
        # Configurer ChB (shunt – carte, AC couplé, petite range)
        assert_pico_ok(ps.ps5000aSetChannel(
            self.chandle, PICO_CH_B, 1, PICO_COUPLING_B, PICO_RANGE_B, 0,
        ))

        self.max_adc = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.max_adc)))

        # Pas de trigger hardware : désactivé (enabled=0).
        # Le Pico capture dès ps5000aRunBlock, la synchro se fait via ARM_DELAY_US.
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.chandle,
            0,           # disabled
            PICO_CH_A,   # canal ignoré si désactivé
            0,           # threshold ignoré
            ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
            0,
            0,
        ))
        print("[PICO] trigger désactivé (capture immédiate à l'armement)", flush=True)

        self.time_interval_ns = ctypes.c_float()
        returned = ctypes.c_int32()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self.chandle,
            TIMEBASE,
            NUM_SAMPLES,
            ctypes.byref(self.time_interval_ns),
            ctypes.byref(returned),
            0,
        ))
        print(f"[PICO] timebase ok dt_ns={self.time_interval_ns.value:.3f}", flush=True)

        self.buffer_a = (ctypes.c_int16 * NUM_SAMPLES)()
        self.buffer_b = (ctypes.c_int16 * NUM_SAMPLES)()

        assert_pico_ok(ps.ps5000aSetDataBuffers(
            self.chandle, PICO_CH_A,
            ctypes.byref(self.buffer_a), None, NUM_SAMPLES, 0, 0,
        ))
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            self.chandle, PICO_CH_B,
            ctypes.byref(self.buffer_b), None, NUM_SAMPLES, 0, 0,
        ))
        print("[PICO] buffers ok (ChA + ChB)", flush=True)

    def arm(self) -> None:
        assert_pico_ok(ps.ps5000aRunBlock(
            self.chandle,
            PRE_TRIGGER_SAMPLES,
            NUM_SAMPLES - PRE_TRIGGER_SAMPLES,
            TIMEBASE,
            None,
            0,
            None,
            None,
        ))

    def wait_capture(self, timeout_s: float = 2.0) -> tuple[np.ndarray, np.ndarray, int]:
        ready = ctypes.c_int16(0)
        deadline = time.time() + timeout_s
        while not ready.value:
            if time.time() >= deadline:
                raise RuntimeError("Timeout Pico: capture non prête")
            assert_pico_ok(ps.ps5000aIsReady(self.chandle, ctypes.byref(ready)))

        num = ctypes.c_int32(NUM_SAMPLES)
        overflow = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aGetValues(
            self.chandle, 0,
            ctypes.byref(num), 1, 0, 0,
            ctypes.byref(overflow),
        ))

        def to_mv(buf, pico_range, attenuation):
            adc = np.array(buf[:num.value], dtype=np.int32)
            mv = np.asarray(adc2mV(adc.tolist(), pico_range, self.max_adc), dtype=np.float32)
            mv *= attenuation
            if num.value != NUM_SAMPLES:
                padded = np.zeros(NUM_SAMPLES, dtype=np.float32)
                padded[:num.value] = mv
                return padded
            return mv

        mv_a = to_mv(self.buffer_a, PICO_RANGE_A, PROBE_ATTENUATION_A)
        mv_b = to_mv(self.buffer_b, PICO_RANGE_B, PROBE_ATTENUATION_B)

        # Tension aux bornes du shunt = signal de puissance
        trace_diff = mv_a - mv_b

        return trace_diff, mv_a, mv_b, int(overflow.value)

    def close(self) -> None:
        try:
            ps.ps5000aStop(self.chandle)
        finally:
            ps.ps5000aCloseUnit(self.chandle)


def compute_label(plaintext: np.ndarray, key: np.ndarray, byte_idx: int = 0) -> int:
    return int(AES_SBOX[plaintext[byte_idx] ^ key[byte_idx]])


def open_serial():
    print(f"[UART] ouverture {SERIAL_PORT} @ {BAUD}", flush=True)
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=UART_TIMEOUT_S)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[UART] ouvert", flush=True)
    return ser


def read_startup_lines(ser, duration: float = 1.0):
    t0 = time.time()
    while time.time() - t0 < duration:
        line = ser.readline()
        if line:
            print("STM32:", line.decode(errors="replace").strip())



def wait_for_sync(ser) -> None:
    """Lit et attend le byte 'S' envoyé par la carte avant l'AES."""
    deadline = time.time() + UART_TIMEOUT_S
    while time.time() < deadline:
        b = ser.read(1)
        if b == b"S":
            return
        if b in (b"\r", b"\n", b""):
            continue
        if b and 32 <= b[0] <= 126:
            tail = ser.readline()
            line = (b + tail).decode(errors="replace").strip()
            if line:
                print("STM32:", line)
    raise RuntimeError("Timeout: 'S' non reçu du STM32")


def acquire_one_trace(ser, pico: PicoCapture, plaintext: bytes):
    assert len(plaintext) == 16

    ser.reset_input_buffer()

    # 1. Envoyer le plaintext d'abord
    ser.write(b"P" + plaintext)
    ser.flush()

    # 2. Attendre 'S' (la carte a reçu le PT et va chiffrer dans ~5 ms)
    wait_for_sync(ser)

    # 3. Attendre que l'AES soit imminent avant d'armer le Pico.
    #    La carte attend HAL_Delay(5) = 5000 µs après avoir envoyé 'S'.
    #    On a reçu 'S' après ~87 µs de latence UART.
    #    → AES démarre ~4913 µs après réception de 'S'.
    #    ARM_DELAY_US doit être proche de 4913 µs pour centrer l'AES
    #    dans la fenêtre de capture (ex: 4700 µs = ~200 µs de marge).
    if ARM_DELAY_US > 0:
        time.sleep(ARM_DELAY_US / 1_000_000.0)
    pico.arm()

    # 4. Capturer la trace (le Pico démarre immédiatement, sans trigger)
    trace_diff, mv_a, mv_b, overflow = pico.wait_capture(timeout_s=UART_TIMEOUT_S)

    # 5. Lire le ciphertext ('C' + 16 octets)
    ciphertext = _read_ciphertext(ser)
    return trace_diff, mv_a, mv_b, ciphertext, overflow


def _read_ciphertext(ser) -> bytes:
    """Lit 'C' + 16 octets de ciphertext."""
    deadline = time.time() + UART_TIMEOUT_S
    while time.time() < deadline:
        header = ser.read(1)
        if not header:
            continue
        if header == b"C":
            ct = ser.read(16)
            if len(ct) != 16:
                raise RuntimeError("Ciphertext incomplet")
            return ct
        if header in (b"\r", b"\n"):
            continue
        if 32 <= header[0] <= 126:
            tail = ser.readline()
            line = (header + tail).decode(errors="replace").strip()
            if line:
                print("STM32:", line)
            continue
        raise RuntimeError(f"Réponse inattendue STM32: {header!r}")
    raise RuntimeError("Timeout: aucun ciphertext reçu")


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
        raise ValueError(f"Plage non supportée : {mv} mV. Valeurs possibles : {sorted(lut.keys())}")
    return lut[mv]


def main():
    global SERIAL_PORT, BAUD, N_TRACES, OUTPUT_FILE
    global NUM_SAMPLES, PRE_TRIGGER_SAMPLES
    global TIMEBASE, ARM_DELAY_US
    global PICO_RANGE_A, PICO_RANGE_B
    global PROBE_ATTENUATION_A, PROBE_ATTENUATION_B

    ap = argparse.ArgumentParser(description="AES SCA high-side diff (PicoScope + STM32 sync UART)")
    ap.add_argument("--serial-port",   default=SERIAL_PORT)
    ap.add_argument("--baud",          type=int, default=BAUD)
    ap.add_argument("--n-traces",      type=int, default=N_TRACES)
    ap.add_argument("--output",        default=OUTPUT_FILE)
    ap.add_argument("--num-samples",   type=int, default=NUM_SAMPLES)
    ap.add_argument("--timebase",      type=int, default=TIMEBASE)
    ap.add_argument("--pre-trigger",   type=int, default=PRE_TRIGGER_SAMPLES)
    ap.add_argument("--range-a-mv",    type=int, default=5000,
                    help="Plage ChA en mV (ex: 5000 pour 5V)")
    ap.add_argument("--range-b-mv",    type=int, default=5000,
                    help="Plage ChB en mV (ex: 5000 pour 5V)")
    ap.add_argument("--probe-att-a",   type=int, default=PROBE_ATTENUATION_A)
    ap.add_argument("--probe-att-b",   type=int, default=PROBE_ATTENUATION_B)
    ap.add_argument("--arm-delay-us",  type=int, default=ARM_DELAY_US)
    args = ap.parse_args()

    SERIAL_PORT          = str(args.serial_port)
    BAUD                 = int(args.baud)
    N_TRACES             = int(args.n_traces)
    OUTPUT_FILE          = str(args.output)
    NUM_SAMPLES          = int(args.num_samples)
    PRE_TRIGGER_SAMPLES  = int(args.pre_trigger)
    TIMEBASE             = int(args.timebase)
    ARM_DELAY_US         = max(0, int(args.arm_delay_us))
    PICO_RANGE_A         = mv_to_range_id(int(args.range_a_mv))
    PICO_RANGE_B         = mv_to_range_id(int(args.range_b_mv))
    PROBE_ATTENUATION_A  = int(args.probe_att_a)
    PROBE_ATTENUATION_B  = int(args.probe_att_b)

    print("[MAIN] start", flush=True)
    ser = open_serial()
    read_startup_lines(ser)
    pico = PicoCapture()

    traces      = np.zeros((N_TRACES, NUM_SAMPLES), dtype=np.float32)
    traces_a    = np.zeros((N_TRACES, NUM_SAMPLES), dtype=np.float32)
    traces_b    = np.zeros((N_TRACES, NUM_SAMPLES), dtype=np.float32)
    plaintexts  = np.zeros((N_TRACES, 16), dtype=np.uint8)
    ciphertexts = np.zeros((N_TRACES, 16), dtype=np.uint8)
    labels      = np.zeros((N_TRACES,), dtype=np.uint8)
    overflows   = np.zeros((N_TRACES,), dtype=np.int16)

    try:
        print(f"[MAIN] acquisition {N_TRACES} traces", flush=True)
        for i in range(N_TRACES):
            pt = os.urandom(16)
            _, mv_a, mv_b, ct, overflow = acquire_one_trace(ser, pico, pt)

            pt_arr = np.frombuffer(pt, dtype=np.uint8)
            ct_arr = np.frombuffer(ct, dtype=np.uint8)
            label  = compute_label(pt_arr, KEY, byte_idx=0)

            traces[i]      = mv_b          # ChB AC = signal de puissance pour CPA
            traces_a[i]    = mv_a
            traces_b[i]    = mv_b
            plaintexts[i]  = pt_arr
            ciphertexts[i] = ct_arr
            labels[i]      = label
            overflows[i]   = overflow

            if (i + 1) <= 3 or (i + 1) % 100 == 0:
                print(f"{i + 1}/{N_TRACES} | overflow={overflow} | "
                      f"chB_range=[{mv_b.min():.1f}, {mv_b.max():.1f}] mV",
                      flush=True)
    finally:
        print("[MAIN] fermeture devices", flush=True)
        pico.close()
        ser.close()

    np.savez(
        OUTPUT_FILE,
        traces=traces,         # ChB AC couplé (signal principal pour CPA)
        traces_a=traces_a,     # ChA brut
        traces_b=traces_b,     # ChB brut
        plaintexts=plaintexts,
        ciphertexts=ciphertexts,
        labels=labels,
        overflows=overflows,
        key=KEY,
    )

    meta = {
        "n_traces": int(N_TRACES),
        "trace_len": int(NUM_SAMPLES),
        "target_byte": 0,
        "label_definition": "SBox(plaintext[0] xor key[0])",
        "serial_port": SERIAL_PORT,
        "baud": BAUD,
        "timebase": TIMEBASE,
        "num_samples": NUM_SAMPLES,
        "pre_trigger_samples": PRE_TRIGGER_SAMPLES,
        "measure_mode": "high_side_differential",
        "trigger_mode": "none_uart_sync",
        "channel_a": "5V_side_of_shunt",
        "channel_b": "card_side_of_shunt",
        "trace_content": "ChA_mV_minus_ChB_mV",
        "probe_attenuation_a": PROBE_ATTENUATION_A,
        "probe_attenuation_b": PROBE_ATTENUATION_B,
        "pico_range_a_mv": int(args.range_a_mv),
        "pico_range_b_mv": int(args.range_b_mv),
        "arm_delay_us": ARM_DELAY_US,
        "dt_ns": float(pico.time_interval_ns.value),
        "n_traces_saved": int(traces.shape[0]),
        "overflow_ratio": float((overflows != 0).mean()),
    }

    with open(OUTPUT_FILE.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset sauvegardé dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
