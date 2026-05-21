#!/usr/bin/env python3
"""Acquisition AES UART high-side avec trigger materiel sur Pico ChB.

Cablage par defaut:
  - Pico ChA: tension apres shunt cote carte, referencee au GND Pico
  - Pico ChB: trigger PB8 STM32
  - trace sauvegardee: supply_mv - ChA, donc une image positive de la chute
    dans le shunt high-side.

Protocole UART cible:
  PC -> STM32: b'P' + 16 octets plaintext
  STM32 -> PC: b'C' + 16 octets ciphertext
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np
import serial


KEY_128 = np.array([
    0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
    0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C,
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)

RANGES = [
    "PS5000A_10MV", "PS5000A_20MV", "PS5000A_50MV", "PS5000A_100MV",
    "PS5000A_200MV", "PS5000A_500MV", "PS5000A_1V", "PS5000A_2V",
    "PS5000A_5V", "PS5000A_10V", "PS5000A_20V",
]

RANGE_MV = {
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


def ensure_pico_runtime() -> None:
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


def wait_ready(ser: serial.Serial, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    seen = []
    while time.time() < deadline:
        line = ser.readline()
        if not line:
            continue
        text = line.decode(errors="replace").strip()
        if text:
            print(f"STM32: {text}", flush=True)
            seen.append(text)
        if text == "READY":
            return
    print("[UART] READY non recu, on continue quand meme.", flush=True)


def read_ciphertext(ser: serial.Serial, timeout_s: float) -> bytes:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b"C":
            ct = ser.read(16)
            if len(ct) != 16:
                raise RuntimeError("Ciphertext incomplet")
            return ct
        if b in (b"\r", b"\n"):
            continue
        if 32 <= b[0] <= 126:
            tail = ser.readline()
            text = (b + tail).decode(errors="replace").strip()
            if text:
                print(f"STM32: {text}", flush=True)
            continue
        raise RuntimeError(f"Octet UART inattendu: {b!r}")
    raise RuntimeError("Timeout UART: ciphertext non recu")


def main() -> None:
    ap = argparse.ArgumentParser(description="AES UART high-side: ChA mesure, ChB trigger")
    ap.add_argument("--serial-port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--uart-timeout", type=float, default=2.0)
    ap.add_argument("--n-traces", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--pre-trigger", type=int, default=500)
    ap.add_argument("--timebase", type=int, default=8)
    ap.add_argument("--output", default="aes_uart_highside_chb_trigger.npz")
    ap.add_argument("--supply-mv", type=float, default=3300.0)
    ap.add_argument("--meas-channel", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--trigger-channel", choices=["A", "B", "C", "D"], default="B")
    ap.add_argument("--meas-range", choices=RANGES, default="PS5000A_5V")
    ap.add_argument("--trigger-range", choices=RANGES, default="PS5000A_5V")
    ap.add_argument("--trigger-mv-probe", type=int, default=1500)
    ap.add_argument("--probe-att-meas", type=int, default=1)
    ap.add_argument("--probe-att-trigger", type=int, default=1)
    ap.add_argument("--trigger-direction", choices=["rising", "falling"], default="rising")
    ap.add_argument("--clock-mhz", type=float, default=None, help="Frequence STM32 flashee, stockee seulement dans le JSON")
    ap.add_argument("--settle-s", type=float, default=2.0)
    args = ap.parse_args()

    if args.pre_trigger >= args.num_samples:
        raise ValueError("--pre-trigger doit etre strictement inferieur a --num-samples")
    if args.meas_channel == args.trigger_channel:
        print("[WARN] meme canal pour mesure et trigger; verifier que c'est volontaire.", flush=True)

    ensure_pico_runtime()
    from picosdk.ps5000a import ps5000a as ps
    from picosdk.functions import assert_pico_ok, mV2adc

    channel_id = {
        name: ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{name}"]
        for name in ("A", "B", "C", "D")
    }
    range_meas = ps.PS5000A_RANGE[args.meas_range]
    range_trigger = ps.PS5000A_RANGE[args.trigger_range]
    coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
    direction = ps.PS5000A_THRESHOLD_DIRECTION[
        "PS5000A_RISING" if args.trigger_direction == "rising" else "PS5000A_FALLING"
    ]

    ser = None
    chandle = ctypes.c_int16()
    try:
        print(f"[UART] ouverture {args.serial_port} @ {args.baud}", flush=True)
        ser = serial.Serial(args.serial_port, args.baud, timeout=args.uart_timeout)
        time.sleep(args.settle_s)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        wait_ready(ser, timeout_s=max(1.0, args.uart_timeout))

        print("[PICO] ouverture", flush=True)
        resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
        status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
        if status in (286, 282):
            assert_pico_ok(ps.ps5000aChangePowerSource(chandle, status))
        else:
            assert_pico_ok(status)

        max_adc = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))

        ch_meas = channel_id[args.meas_channel]
        ch_trig = channel_id[args.trigger_channel]
        assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_meas, 1, coupling, range_meas, 0.0))
        if ch_trig != ch_meas:
            assert_pico_ok(ps.ps5000aSetChannel(chandle, ch_trig, 1, coupling, range_trigger, 0.0))

        trig_mv_bnc = max(1, int(args.trigger_mv_probe) // int(args.probe_att_trigger))
        trig_adc = mV2adc(trig_mv_bnc, range_trigger, max_adc)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(chandle, 1, ch_trig, trig_adc, direction, 0, 1000))

        dt_ns = ctypes.c_float()
        returned = ctypes.c_int32()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            chandle, args.timebase, args.num_samples,
            ctypes.byref(dt_ns), ctypes.byref(returned), 0,
        ))
        print(f"[PICO] dt={dt_ns.value:.3f} ns, trigger Ch{args.trigger_channel}", flush=True)

        buf_meas = (ctypes.c_int16 * args.num_samples)()
        buf_trig = (ctypes.c_int16 * args.num_samples)()
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            chandle, ch_meas, ctypes.byref(buf_meas), None, args.num_samples, 0, 0,
        ))
        if ch_trig != ch_meas:
            assert_pico_ok(ps.ps5000aSetDataBuffers(
                chandle, ch_trig, ctypes.byref(buf_trig), None, args.num_samples, 0, 0,
            ))

        traces = np.zeros((args.n_traces, args.num_samples), dtype=np.float32)
        meas_mv = np.zeros_like(traces)
        trig_mv = np.zeros_like(traces)
        plaintexts = np.zeros((args.n_traces, 16), dtype=np.uint8)
        ciphertexts = np.zeros((args.n_traces, 16), dtype=np.uint8)
        labels = np.zeros(args.n_traces, dtype=np.uint8)
        overflows = np.zeros(args.n_traces, dtype=np.int16)

        post_trigger = args.num_samples - args.pre_trigger
        scale_meas = RANGE_MV[args.meas_range] * float(args.probe_att_meas) / float(max_adc.value)
        scale_trig = RANGE_MV[args.trigger_range] * float(args.probe_att_trigger) / float(max_adc.value)

        for i in range(args.n_traces):
            pt = os.urandom(16)
            pt_arr = np.frombuffer(pt, dtype=np.uint8)

            ser.reset_input_buffer()
            assert_pico_ok(ps.ps5000aRunBlock(
                chandle, args.pre_trigger, post_trigger,
                args.timebase, None, 0, None, None,
            ))

            ser.write(b"P" + pt)
            ser.flush()

            ready = ctypes.c_int16(0)
            deadline = time.time() + args.uart_timeout
            while not ready.value:
                if time.time() >= deadline:
                    raise RuntimeError("Timeout Pico: trigger non detecte")
                assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))

            count = ctypes.c_int32(args.num_samples)
            overflow = ctypes.c_int16(0)
            assert_pico_ok(ps.ps5000aGetValues(
                chandle, 0, ctypes.byref(count), 1, 0, 0, ctypes.byref(overflow),
            ))
            ct = read_ciphertext(ser, timeout_s=args.uart_timeout)

            meas = np.asarray(buf_meas[:count.value], dtype=np.float32) * scale_meas
            if ch_trig == ch_meas:
                trig = meas.copy()
            else:
                trig = np.asarray(buf_trig[:count.value], dtype=np.float32) * scale_trig
            if count.value != args.num_samples:
                raise RuntimeError(f"Capture incomplete: {count.value}/{args.num_samples}")

            traces[i] = float(args.supply_mv) - meas
            meas_mv[i] = meas
            trig_mv[i] = trig
            plaintexts[i] = pt_arr
            ciphertexts[i] = np.frombuffer(ct, dtype=np.uint8)
            labels[i] = AES_SBOX[int(pt_arr[0] ^ KEY_128[0])]
            overflows[i] = overflow.value

            if i < 3 or (i + 1) % 100 == 0:
                print(
                    f"{i + 1}/{args.n_traces} overflow={overflow.value} "
                    f"trace=[{traces[i].min():.2f},{traces[i].max():.2f}]mV "
                    f"meas_mean={meas.mean():.2f}mV",
                    flush=True,
                )

        np.savez(
            args.output,
            traces=traces,
            measure_mv=meas_mv,
            trigger_mv=trig_mv,
            plaintexts=plaintexts,
            ciphertexts=ciphertexts,
            labels=labels,
            overflows=overflows,
            key=KEY_128,
            meta_dt_ns=np.float32(dt_ns.value),
            meta_supply_mv=np.float32(args.supply_mv),
            meta_meas_channel=np.array(args.meas_channel),
            meta_trigger_channel=np.array(args.trigger_channel),
        )

        meta = {
            "n_traces": int(args.n_traces),
            "num_samples": int(args.num_samples),
            "pre_trigger": int(args.pre_trigger),
            "timebase": int(args.timebase),
            "dt_ns": float(dt_ns.value),
            "serial_port": args.serial_port,
            "baud": int(args.baud),
            "measure_channel": args.meas_channel,
            "trigger_channel": args.trigger_channel,
            "measure_definition": f"{args.supply_mv:.1f}mV - Ch{args.meas_channel}",
            "trigger_definition": f"Ch{args.trigger_channel} = PB8 STM32",
            "meas_range": args.meas_range,
            "trigger_range": args.trigger_range,
            "trigger_mv_probe": int(args.trigger_mv_probe),
            "probe_att_meas": int(args.probe_att_meas),
            "probe_att_trigger": int(args.probe_att_trigger),
            "clock_mhz": None if args.clock_mhz is None else float(args.clock_mhz),
            "overflow_ratio": float((overflows != 0).mean()),
        }
        with open(Path(args.output).with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[OK] dataset: {args.output}", flush=True)

    finally:
        if chandle.value:
            try:
                ps.ps5000aStop(chandle)
            finally:
                ps.ps5000aCloseUnit(chandle)
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
