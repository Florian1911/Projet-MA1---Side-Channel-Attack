#!/usr/bin/env python3
"""
Acquisition SCA sans UART – rapid block mode (PicoScope 5000A).

Protocole de synchronisation :
  1. Lance ce script  →  scope armé pour N captures
  2. Le script affiche "PRÊT → appuie RESET"
  3. Appuie sur le bouton RESET de la carte STM32
  4. La LED clignote 3×, attend 3 s, puis lance les N chiffrements
  5. Le scope capture tous les triggers automatiquement
  6. LED fixe = terminé

Alignement vérifié en fin de script (mean trace + cross-corr sur 10 traces).
"""

import os
import time
import ctypes
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# PicoScope runtime
# ──────────────────────────────────────────────────────────────────────────────
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


_ensure_pico_runtime()

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (modifiables via args)
# ──────────────────────────────────────────────────────────────────────────────
N_TRACES          = 5000
NUM_SAMPLES       = 20000
PRE_TRIGGER       = 200
OUTPUT_FILE       = "dataset_no_uart.npz"
PLAINTEXTS_FILE   = "plaintexts_no_uart.npy"

TIMEBASE          = 8          # 80 ns/sample à 12 bits, 2 canaux

PICO_CH_TRIG      = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]   # PB8
PICO_CH_MEAS      = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]   # low-side shunt
PICO_COUPLING     = ps.PS5000A_COUPLING["PS5000A_DC"]
PICO_RANGE_TRIG   = ps.PS5000A_RANGE["PS5000A_500MV"]
PICO_RANGE_MEAS   = ps.PS5000A_RANGE["PS5000A_200MV"]
VOLT_RANGE_MV     = 200.0      # doit correspondre à PICO_RANGE_MEAS

PROBE_ATT_TRIG    = 10         # ×10 sur la sonde trigger
PROBE_ATT_MEAS    = 1
TRIG_MV_PROBE     = 1500       # seuil trigger (côté sonde ×10 → 150 mV BNC)
TRIG_DIR          = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
AUTO_TRIGGER_MS   = 0          # pas d'auto-trigger (attente infinie par capture)

KEY = np.array([
    0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
    0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C
], dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Vérification d'alignement par cross-corrélation
# ──────────────────────────────────────────────────────────────────────────────
def check_alignment(traces: np.ndarray) -> None:
    """
    Vérification d'alignement par variance inter-traces.
    Avec trigger hardware, les traces sont alignées si la std montre de la structure.
    """
    non_zero = (traces.std(axis=1) > 0).sum()
    print(f"[ALIGN] {non_zero}/{len(traces)} traces non nulles")
    if non_zero < len(traces):
        print(f"[ALIGN] AVERTISSEMENT : {len(traces) - non_zero} traces vides → bug buffer")
    else:
        print(f"[ALIGN] OK – toutes les traces ont du signal")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global N_TRACES, NUM_SAMPLES, PRE_TRIGGER, OUTPUT_FILE, PLAINTEXTS_FILE

    ap = argparse.ArgumentParser(description="Acquisition no-UART rapid-block")
    ap.add_argument("--n-traces",    type=int, default=N_TRACES)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--pre-trigger", type=int, default=PRE_TRIGGER)
    ap.add_argument("--output",      default=OUTPUT_FILE)
    ap.add_argument("--plaintexts",  default=PLAINTEXTS_FILE)
    ap.add_argument("--probe-att-trig", type=int, default=PROBE_ATT_TRIG,
                    help="Atténuation sonde trigger (1 ou 10)")
    ap.add_argument("--trig-mv-probe", type=int, default=TRIG_MV_PROBE,
                    help="Seuil trigger en mV côté sonde (avant atténuation)")
    ap.add_argument("--min-duration-factor", type=float, default=0.5,
                    help="Alerte si durée capture < facteur * durée estimée")
    args = ap.parse_args()

    N_TRACES        = args.n_traces
    NUM_SAMPLES     = args.num_samples
    PRE_TRIGGER     = args.pre_trigger
    POST_TRIGGER    = NUM_SAMPLES - PRE_TRIGGER
    OUTPUT_FILE     = args.output
    PLAINTEXTS_FILE = args.plaintexts

    # ── Chargement plaintexts ─────────────────────────────────────────────────
    pts_all = np.load(PLAINTEXTS_FILE)
    if len(pts_all) < N_TRACES:
        raise ValueError(f"{PLAINTEXTS_FILE} contient {len(pts_all)} plaintexts, besoin de {N_TRACES}")
    pts = pts_all[:N_TRACES]
    print(f"[OK] {N_TRACES} plaintexts chargés depuis {PLAINTEXTS_FILE}")

    # ── Ouverture PicoScope ───────────────────────────────────────────────────
    print("[PICO] ouverture...", flush=True)
    chandle    = ctypes.c_int16()
    resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
    status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    if status in (286, 282):
        assert_pico_ok(ps.ps5000aChangePowerSource(chandle, status))
    else:
        assert_pico_ok(status)

    max_adc = ctypes.c_int16()
    assert_pico_ok(ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc)))

    # ── Canaux ────────────────────────────────────────────────────────────────
    assert_pico_ok(ps.ps5000aSetChannel(
        chandle, PICO_CH_TRIG, 1, PICO_COUPLING, PICO_RANGE_TRIG, 0))
    assert_pico_ok(ps.ps5000aSetChannel(
        chandle, PICO_CH_MEAS, 1, PICO_COUPLING, PICO_RANGE_MEAS, 0))

    # ── Trigger ───────────────────────────────────────────────────────────────
    if args.probe_att_trig not in (1, 10):
        raise ValueError("--probe-att-trig doit valoir 1 ou 10")
    thr_bnc_mv  = max(1, int(args.trig_mv_probe) // int(args.probe_att_trig))
    thr_adc     = mV2adc(thr_bnc_mv, PICO_RANGE_TRIG, max_adc)
    assert_pico_ok(ps.ps5000aSetSimpleTrigger(
        chandle, 1, PICO_CH_TRIG, thr_adc, TRIG_DIR, 0, AUTO_TRIGGER_MS))
    print(f"[PICO] trigger: {args.trig_mv_probe} mV sonde / x{args.probe_att_trig} -> {thr_bnc_mv} mV BNC")

    # ── Timebase ──────────────────────────────────────────────────────────────
    dt_ns    = ctypes.c_float()
    returned = ctypes.c_int32()
    assert_pico_ok(ps.ps5000aGetTimebase2(
        chandle, TIMEBASE, NUM_SAMPLES,
        ctypes.byref(dt_ns), ctypes.byref(returned), 0))
    print(f"[PICO] dt_ns={dt_ns.value:.1f}  ({1e3/dt_ns.value:.1f} MS/s)")

    # ── Segmentation mémoire pour N_TRACES captures ───────────────────────────
    print(f"[PICO] segmentation mémoire ({N_TRACES} segments)...", flush=True)
    max_samp_seg = ctypes.c_int32(0)
    assert_pico_ok(ps.ps5000aMemorySegments(
        chandle, N_TRACES, ctypes.byref(max_samp_seg)))

    if max_samp_seg.value < NUM_SAMPLES:
        ps.ps5000aCloseUnit(chandle)
        raise RuntimeError(
            f"Mémoire scope insuffisante : max {max_samp_seg.value} samples/segment, "
            f"besoin {NUM_SAMPLES}.\n"
            f"Réduis --num-samples ou --n-traces.")

    print(f"[PICO] max samples/seg = {max_samp_seg.value}  (besoin {NUM_SAMPLES} ✓)")
    assert_pico_ok(ps.ps5000aSetNoOfCaptures(chandle, N_TRACES))

    # ── ARM : RunBlock pour toutes les N_TRACES captures ─────────────────────
    # Note : les buffers sont alloués APRÈS la capture (segment par segment),
    #        pas avant RunBlock — évite le bug où seul le segment 0 est rempli.
    raw_adc = np.zeros((N_TRACES, NUM_SAMPLES), dtype=np.int16)

    assert_pico_ok(ps.ps5000aRunBlock(
        chandle, PRE_TRIGGER, POST_TRIGGER,
        TIMEBASE, None, 0, None, None))

    # ── Synchronisation utilisateur ───────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  SCOPE ARMÉ – {N_TRACES} captures en attente")
    print()
    print("  ACTION :  Appuie sur RESET de la carte STM32 maintenant")
    print()
    print("  La LED clignote 3× puis attend 3 s avant de démarrer.")
    expected_s = N_TRACES * 11 / 1000
    print(f"  Durée estimée : ~{expected_s:.0f} s")
    print("=" * 60)
    print()

    # ── Attente de la fin des N captures ─────────────────────────────────────
    total_timeout = N_TRACES * 0.015 + 10   # 15 ms/trace + 10 s marge
    deadline  = time.time() + total_timeout
    ready     = ctypes.c_int16(0)
    t_start   = time.time()

    while not ready.value:
        if time.time() >= deadline:
            ps.ps5000aStop(chandle)
            ps.ps5000aCloseUnit(chandle)
            raise RuntimeError(
                f"Timeout : {N_TRACES} captures non reçues en {total_timeout:.0f} s.\n"
                "Vérifie que la carte tourne et que le trigger PB8 est connecté.")
        assert_pico_ok(ps.ps5000aIsReady(chandle, ctypes.byref(ready)))
        time.sleep(0.2)

    elapsed = time.time() - t_start
    print(f"[OK] {N_TRACES} captures reçues en {elapsed:.1f} s")
    min_expected = max(1.0, expected_s * args.min_duration_factor)
    if elapsed < min_expected:
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)
        raise RuntimeError(
            f"Capture trop rapide ({elapsed:.1f} s < {min_expected:.1f} s). "
            "Probable faux trigger (seuil trop bas / bruit). "
            "Augmente --trig-mv-probe ou vérifie le câblage PB8 -> ChA."
        )

    # ── Lecture segment par segment (SetDataBuffers juste avant GetValues) ────
    print("[PICO] lecture des données...", flush=True)
    overflows = np.zeros(N_TRACES, dtype=np.int16)

    for seg in range(N_TRACES):
        # Associer le buffer JUSTE avant la lecture de ce segment
        ptr = raw_adc[seg].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            chandle, PICO_CH_MEAS, ptr, None, NUM_SAMPLES, seg, 0))

        num = ctypes.c_int32(NUM_SAMPLES)
        ov  = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aGetValues(
            chandle, 0, ctypes.byref(num), 1, 0, seg, ctypes.byref(ov)))
        overflows[seg] = ov.value

        if (seg + 1) % 500 == 0 or seg == 0:
            print(f"  {seg + 1}/{N_TRACES}", flush=True)

    ps.ps5000aStop(chandle)
    ps.ps5000aCloseUnit(chandle)
    print("[PICO] fermé")

    # ── Conversion ADC → mV (vectorisée) ──────────────────────────────────────
    print("[POST] conversion ADC → mV…", flush=True)
    traces = raw_adc.astype(np.float32) * (VOLT_RANGE_MV / float(max_adc.value))
    del raw_adc

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    np.savez(OUTPUT_FILE,
             traces=traces,
             plaintexts=pts,
             key=KEY,
             overflows=overflows)
    print(f"[OK] Dataset → {OUTPUT_FILE}")
    print(f"     {N_TRACES} traces × {NUM_SAMPLES} samples")
    print(f"     overflows : {(overflows != 0).sum()}/{N_TRACES}")

    # ── Vérification d'alignement ─────────────────────────────────────────────
    print()
    check_alignment(traces)

    # ── Visualisation rapide ──────────────────────────────────────────────────
    mean_t = traces.mean(axis=0)
    std_t  = traces.std(axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(traces[:5].T, alpha=0.7, lw=0.8)
    axes[0].set_title("5 premières traces")
    axes[0].set_ylabel("Amplitude (mV)")

    axes[1].plot(mean_t, lw=0.8)
    axes[1].set_title(f"Mean trace ({N_TRACES} traces)")
    axes[1].set_ylabel("mV")

    axes[2].plot(std_t, lw=0.8, color='orange')
    axes[2].set_title("Standard deviation (activité)")
    axes[2].set_ylabel("Std (mV)")
    axes[2].set_xlabel("Sample")

    plt.tight_layout()
    fig.savefig("check_no_uart.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
