#!/usr/bin/env python3
"""
Acquisition SCA sans UART pour PicoScope 2000 (driver ps2000 classique).
Mode: block séquentiel (une capture par trigger).

Important:
- Ce mode est moins robuste que le rapid-block segmenté (5000A/2000A).
- Si la ré-armement est trop lent, des triggers peuvent être ratés.
"""

import os
import time
import ctypes
import argparse
from pathlib import Path

import numpy as np


def _parse_key_hex(s: str) -> np.ndarray:
    k = s.strip().replace(" ", "").replace(":", "").replace(",", "")
    if len(k) != 32:
        raise ValueError("--key-hex doit contenir 32 caractères hex (16 bytes)")
    return np.array([int(k[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)


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


def _ok(ret: int, what: str) -> None:
    if int(ret) <= 0:
        raise RuntimeError(f"{what} a échoué (ret={ret})")


def _time_interval_ns(val: int, units: int) -> float:
    # ps2000: 0=fs,1=ps,2=ns,3=us,4=ms,5=s
    scale = {
        0: 1e-6,
        1: 1e-3,
        2: 1.0,
        3: 1e3,
        4: 1e6,
        5: 1e9,
    }.get(int(units), 1.0)
    return float(val) * scale


def main():
    _ensure_pico_runtime()

    from picosdk.ps2000 import ps2000 as ps

    ap = argparse.ArgumentParser(description="Acquisition no-UART (Pico 2000, ps2000)")
    ap.add_argument("--n-traces", type=int, default=5000)
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--output", default="dataset_no_uart_2000.npz")
    ap.add_argument("--plaintexts", default="plaintexts_no_uart.npy")
    ap.add_argument("--cycle-plaintexts", action="store_true",
                    help="Autorise n-traces > nb plaintexts en rejouant plaintexts en boucle (modulo)")
    ap.add_argument("--timebase", type=int, default=1, help="Timebase initiale (auto-ajustée si invalide)")
    ap.add_argument("--probe-att-trig", type=int, default=10, help="Atténuation sonde trigger (1 ou 10)")
    ap.add_argument("--trig-mv-probe", type=int, default=1500, help="Seuil trigger en mV côté sonde")
    ap.add_argument("--trig-channel", choices=["A", "B"], default="A",
                    help="Canal de trigger: A (PB8) ou B (shunt)")
    ap.add_argument("--trig-direction", choices=["rising", "falling"], default="rising",
                    help="Direction du trigger")
    ap.add_argument("--auto-trigger-ms", type=int, default=0,
                    help="Auto-trigger en ms (0 = désactivé)")
    ap.add_argument("--trace-timeout-s", type=float, default=8.0, help="Timeout par trace")
    ap.add_argument("--first-trace-timeout-s", type=float, default=20.0,
                    help="Timeout pour la 1re trace (boot + délais init après RESET)")
    ap.add_argument("--tb-max-scan", type=int, default=4096, help="Borne max de recherche timebase")
    ap.add_argument("--capture-trig", action="store_true",
                    help="Sauvegarder aussi la trace trigger ChA dans le .npz")
    ap.add_argument("--trig-p2p-min-mv-probe", type=float, default=0.0,
                    help="Si >0: rejette une capture si p2p(ChA) < seuil (mV côté sonde)")
    ap.add_argument("--max-extra-captures", type=int, default=2000,
                    help="Captures supplémentaires autorisées pour compenser des rejets trigger")
    ap.add_argument("--save-partial-on-timeout", action="store_true", default=True,
                    help="Sauvegarder les traces déjà capturées en cas de timeout")
    ap.add_argument("--key-hex", default="2b7e151628aed2a6abf7158809cf4f3c",
                    help="Clé AES-128 à stocker dans le .npz (hex, 32 caractères)")
    args = ap.parse_args()

    n_traces = int(args.n_traces)
    num_samples = int(args.num_samples)

    pts_all = np.load(args.plaintexts)
    n_pt = len(pts_all)
    if n_pt < n_traces and not args.cycle_plaintexts:
        raise ValueError(
            f"{args.plaintexts} contient {n_pt} plaintexts, besoin de {n_traces}.\n"
            "Utilise --cycle-plaintexts si le firmware rejoue les plaintexts en boucle."
        )
    if n_pt >= n_traces:
        pts = pts_all[:n_traces]
        print(f"[OK] {n_traces} plaintexts chargés depuis {args.plaintexts}")
    else:
        idx = np.arange(n_traces, dtype=np.int64) % n_pt
        pts = pts_all[idx]
        print(
            f"[OK] plaintexts cyclés: source={n_pt}, demandé={n_traces} "
            f"(mode modulo activé)"
        )

    print("[PICO2000] ouverture...", flush=True)
    handle = ps.ps2000_open_unit()
    if handle <= 0:
        raise RuntimeError("Impossible d'ouvrir le Pico 2000.")

    ch_a = ps.PS2000_CHANNEL["PS2000_CHANNEL_A"]  # trigger PB8
    ch_b = ps.PS2000_CHANNEL["PS2000_CHANNEL_B"]  # shunt
    dc = ps.PICO_COUPLING["DC"]
    range_a = ps.PS2000_VOLTAGE_RANGE["PS2000_500MV"]
    range_b = ps.PS2000_VOLTAGE_RANGE["PS2000_200MV"]
    range_b_mv = ps.PICO_VOLTAGE_RANGE[range_b] * 1000.0
    max_adc = 32767.0

    _ok(ps.ps2000_set_channel(handle, ch_a, 1, dc, range_a), "SetChannel A")
    _ok(ps.ps2000_set_channel(handle, ch_b, 1, dc, range_b), "SetChannel B")

    if args.probe_att_trig not in (1, 10):
        raise ValueError("--probe-att-trig doit valoir 1 ou 10")
    thr_bnc_mv = max(1, int(args.trig_mv_probe) // int(args.probe_att_trig))
    thr_adc = int((thr_bnc_mv / 500.0) * max_adc)  # range trigger = 500mV
    trig_ch = ch_a if args.trig_channel == "A" else ch_b
    trig_dir = 2 if args.trig_direction == "rising" else 3
    _ok(ps.ps2000_set_trigger(handle, trig_ch, thr_adc, trig_dir, 0, int(args.auto_trigger_ms)), "SetTrigger")
    print(
        f"[PICO2000] trigger: ch={args.trig_channel} dir={args.trig_direction} "
        f"thr={args.trig_mv_probe} mV sonde / x{args.probe_att_trig} -> {thr_bnc_mv} mV BNC "
        f"(auto={int(args.auto_trigger_ms)} ms)"
    )

    ti = ctypes.c_int32()
    tu = ctypes.c_int16()
    max_s = ctypes.c_int32()
    tb_start = int(args.timebase)
    tb_max = int(args.tb_max_scan)

    best_tb = None
    best_ti = None
    best_tu = None
    best_max = -1

    # 1) Pass "capacity probe": request 1 sample to discover valid timebases
    #    and their max sample capacity on this exact channel configuration.
    for tb in range(0, tb_max + 1):
        ret = ps.ps2000_get_timebase(
            handle, tb, 1,
            ctypes.byref(ti), ctypes.byref(tu), 1, ctypes.byref(max_s)
        )
        if ret > 0:
            if int(max_s.value) > best_max:
                best_tb = tb
                best_ti = int(ti.value)
                best_tu = int(tu.value)
                best_max = int(max_s.value)

    if best_tb is None:
        ps.ps2000_close_unit(handle)
        raise RuntimeError("Aucune timebase valide trouvée (scan vide).")

    if best_max < num_samples:
        print(f"[PICO2000] num-samples ajusté: {num_samples} -> {best_max} (limite hardware)")
        num_samples = best_max

    # 2) Pick a working timebase for the final num_samples.
    # Prefer requested tb if valid, otherwise nearest valid around it, then fallback.
    search_order = []
    if 0 <= tb_start <= tb_max:
        search_order.append(tb_start)
        for d in range(1, tb_max + 1):
            lo = tb_start - d
            hi = tb_start + d
            added = False
            if lo >= 0:
                search_order.append(lo)
                added = True
            if hi <= tb_max:
                search_order.append(hi)
                added = True
            if not added:
                break
    else:
        search_order = list(range(0, tb_max + 1))

    tb = None
    for cand in search_order:
        ret = ps.ps2000_get_timebase(
            handle, cand, num_samples,
            ctypes.byref(ti), ctypes.byref(tu), 1, ctypes.byref(max_s)
        )
        if ret > 0 and int(max_s.value) >= num_samples:
            tb = cand
            best_ti = int(ti.value)
            best_tu = int(tu.value)
            break
    if tb is None:
        ps.ps2000_close_unit(handle)
        raise RuntimeError(f"Aucune timebase valide pour {num_samples} samples.")

    if tb != args.timebase:
        print(f"[PICO2000] timebase ajustée: {args.timebase} -> {tb}")
    dt_ns = _time_interval_ns(best_ti, best_tu)
    if dt_ns >= 1.0:
        print(f"[PICO2000] dt_ns={dt_ns:.1f}  ({1e3/dt_ns:.1f} MS/s)")
    else:
        print(f"[PICO2000] dt_ns={dt_ns:.4f}  ({1e3/dt_ns:.1f} MS/s)")

    need_trig_buffer = bool(args.capture_trig) or float(args.trig_p2p_min_mv_probe) > 0.0
    raw_adc = np.zeros((n_traces, num_samples), dtype=np.int16)
    raw_adc_trig = np.zeros((n_traces, num_samples), dtype=np.int16) if need_trig_buffer else None
    overflows = np.zeros(n_traces, dtype=np.int16)
    trig_p2p_mv_probe = np.zeros(n_traces, dtype=np.float32) if need_trig_buffer else None

    print()
    print("=" * 60)
    print(f"  SCOPE ARMÉ (séquentiel) – {n_traces} captures en attente")
    print()
    print("  ACTION : Appuie sur RESET de la carte STM32 maintenant")
    print("=" * 60)
    print()

    t_start = time.time()
    captured = 0
    rejected = 0
    attempts = 0
    fired = 0  # nombre de triggers réellement capturés (acceptés + rejetés)
    accepted_pt_idx = np.zeros(n_traces, dtype=np.int32)
    max_attempts = n_traces + max(0, int(args.max_extra_captures))
    while captured < n_traces and attempts < max_attempts:
        attempts += 1
        indisposed = ctypes.c_int32()
        _ok(ps.ps2000_run_block(handle, num_samples, tb, 1, ctypes.byref(indisposed)), "RunBlock")

        timeout_s = float(args.first_trace_timeout_s if captured == 0 else args.trace_timeout_s)
        t_deadline = time.time() + timeout_s
        while ps.ps2000_ready(handle) == 0:
            if time.time() >= t_deadline:
                ps.ps2000_stop(handle)
                if args.save_partial_on_timeout:
                    print(f"[WARN] Timeout trace {captured+1}/{n_traces} (trigger manquant). Sauvegarde partielle.")
                    break
                ps.ps2000_close_unit(handle)
                raise RuntimeError(f"Timeout trace {captured+1}/{n_traces} (trigger manquant)")
            time.sleep(0.001)
        else:
            # Loop completed only if trigger received
            pass

        if ps.ps2000_ready(handle) == 0:
            # break path after timeout with partial save enabled
            break

        ptr_b = raw_adc[captured].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        ptr_a = (raw_adc_trig[captured].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
                 if need_trig_buffer else None)
        ov = ctypes.c_int16(0)
        got = ps.ps2000_get_values(handle, ptr_a, ptr_b, None, None, ctypes.byref(ov), num_samples)
        if got <= 0:
            ps.ps2000_stop(handle)
            ps.ps2000_close_unit(handle)
            raise RuntimeError(f"GetValues échoué à la trace {captured+1}")
        # Un trigger valide vient d'être consommé côté cible -> plaintext index avance.
        pt_idx = fired
        fired += 1
        if pt_idx >= n_traces:
            # La cible a déjà émis tous ses plaintexts; ne pas écrire hors plage.
            break

        if need_trig_buffer:
            # ChA est en range 500 mV (BNC). Conversion en mV côté sonde.
            p2p_adc = int(raw_adc_trig[captured].max()) - int(raw_adc_trig[captured].min())
            p2p_mv_bnc = (p2p_adc / max_adc) * 500.0
            p2p_mv_probe = p2p_mv_bnc * float(args.probe_att_trig)
            trig_p2p_mv_probe[captured] = np.float32(p2p_mv_probe)
            if float(args.trig_p2p_min_mv_probe) > 0.0 and p2p_mv_probe < float(args.trig_p2p_min_mv_probe):
                rejected += 1
                continue

        overflows[captured] = ov.value
        accepted_pt_idx[captured] = np.int32(pt_idx)
        captured += 1

        if (captured % 500 == 0) or captured == 1:
            print(f"  {captured}/{n_traces}", flush=True)

    elapsed = time.time() - t_start
    print(f"[OK] {captured}/{n_traces} captures reçues en {elapsed:.1f} s")
    if rejected > 0:
        print(f"[PICO2000] captures rejetées (trigger faible): {rejected}")
    if attempts >= max_attempts and captured < n_traces:
        print(f"[WARN] limite d'essais atteinte ({max_attempts}), dataset partiel")
    if fired < n_traces:
        print(f"[WARN] triggers réellement vus: {fired}/{n_traces}")

    ps.ps2000_stop(handle)
    ps.ps2000_close_unit(handle)
    print("[PICO2000] fermé")

    traces = raw_adc[:captured].astype(np.float32) * (range_b_mv / max_adc)

    key = _parse_key_hex(args.key_hex)
    print(f"[META] key dans NPZ: {''.join(f'{x:02X}' for x in key)}")

    payload = dict(
        traces=traces,
        plaintexts=pts[accepted_pt_idx[:captured]],
        key=key,
        overflows=overflows[:captured],
        requested_traces=np.int32(n_traces),
        captured_traces=np.int32(captured),
        fired_triggers=np.int32(fired),
        attempts=np.int32(attempts),
        rejected_trig=np.int32(rejected),
        plaintext_indices=accepted_pt_idx[:captured],
        num_samples=np.int32(num_samples),
        timebase=np.int32(tb),
    )
    if need_trig_buffer:
        range_a_mv = ps.PICO_VOLTAGE_RANGE[range_a] * 1000.0
        traces_trig = raw_adc_trig[:captured].astype(np.float32) * (range_a_mv / max_adc) * float(args.probe_att_trig)
        payload["traces_trigger"] = traces_trig
        payload["trig_p2p_mv_probe"] = trig_p2p_mv_probe[:captured]

    np.savez(
        args.output,
        **payload,
    )
    print(f"[OK] Dataset -> {args.output}")
    print(f"     {traces.shape[0]} traces x {traces.shape[1]} samples")
    if captured == 0:
        print(
            "[HINT] 0 trace capturée. Vérifie PB8->ChA, masse commune, "
            "et essaye --auto-trigger-ms 100 pour un diagnostic rapide."
        )
    print(f"     overflows : {(overflows[:captured] != 0).sum()}/{captured}")


if __name__ == "__main__":
    main()
