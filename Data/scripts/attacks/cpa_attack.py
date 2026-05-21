#!/usr/bin/env python3
"""CPA (Correlation Power Analysis) – AES-128 premier tour, modèle Hamming Weight."""

import numpy as np
import argparse

# AES S-box
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float32)

def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)

def cpa_byte(traces_f, plaintexts, byte_idx, t_mean, t_std, model='hw_sbox'):
    """CPA sur un byte de clé. Retourne (best_key, max_corr[256])."""
    N = traces_f.shape[0]
    pt_col = plaintexts[:, byte_idx].astype(np.uint8)  # (N,)

    max_corrs = np.zeros(256, dtype=np.float32)

    for k in range(256):
        if model == 'hw_sbox':
            h = HW[SBOX[pt_col ^ k]]
        elif model == 'hw_in':
            h = HW[pt_col ^ k]
        elif model == 'hd':
            h = HW[SBOX[pt_col ^ k] ^ pt_col]
        elif model == 'bit0':
            h = (SBOX[pt_col ^ k] & 1).astype(np.float32)
        elif model == 'bit7':
            h = ((SBOX[pt_col ^ k] >> 7) & 1).astype(np.float32)

        h_mean = h.mean()
        h_std  = h.std()
        if h_std < 1e-10:
            continue
        h_c = h - h_mean                  # (N,)
        # cov[t] = mean_i( h_c[i] * (trace[i,t] - t_mean[t]) )
        cov = (h_c @ (traces_f - t_mean)) / N   # (T,)
        corr = np.abs(cov / (h_std * t_std))     # (T,)
        max_corrs[k] = corr.max()

    best_key = int(max_corrs.argmax())
    return best_key, max_corrs


def main():
    parser = argparse.ArgumentParser(description="CPA AES-128 premier tour")
    parser.add_argument('--dataset',  default='dataset_final.npz')
    parser.add_argument('--n-traces', type=int, default=None,
                        help='Limiter le nombre de traces (ex: 1000)')
    parser.add_argument('--byte',     type=int, default=None,
                        help='Attaquer un seul byte (0-15)')
    parser.add_argument('--win-start', type=int, default=0,
                        help='Premier sample de la fenêtre (défaut: 0)')
    parser.add_argument('--win-end',   type=int, default=None,
                        help='Dernier sample (exclus) de la fenêtre (défaut: fin)')
    parser.add_argument('--lp-window', type=int, default=1,
                        help='Fenêtre du filtre passe-bas (moyenne glissante), 1 = désactivé')
    parser.add_argument('--model', choices=['hw_sbox', 'hw_in', 'hd', 'bit0', 'bit7'], default='hw_sbox',
                        help='Modèle de fuite à utiliser (défaut: hw_sbox)')
    parser.add_argument('--ignore-dataset-key', action='store_true',
                        help='Ignore la clé stockée dans le dataset (mode unknown-key)')
    args = parser.parse_args()

    d = np.load(args.dataset)
    traces     = d['traces'].astype(np.float32)
    plaintexts = d['plaintexts']
    true_key   = d['key'] if ('key' in d and not args.ignore_dataset_key) else None

    if args.n_traces:
        traces     = traces[:args.n_traces]
        plaintexts = plaintexts[:args.n_traces]

    win_start = args.win_start
    win_end   = args.win_end if args.win_end is not None else traces.shape[1]
    traces    = traces[:, win_start:win_end]

    # CORRECTION : Suppression de la dérive DC et de la pente (Detrending complet)
    traces = center_and_detrend(traces)

    # Filtre passe-bas optionnel (moyenne glissante) pour réduire le bruit HF
    if args.lp_window > 1:
        kernel = np.ones(args.lp_window, dtype=np.float32) / args.lp_window
        traces = np.apply_along_axis(
            lambda t: np.convolve(t, kernel, mode='same'), axis=1, arr=traces)
        print(f"[PRE] filtre passe-bas : fenêtre={args.lp_window} samples")

    N, T = traces.shape
    print(f"Dataset : {N} traces × {T} samples")
    if true_key is not None:
        print(f"Clé attendue : {' '.join(f'{b:02x}' for b in true_key)}")
    else:
        print("Mode unknown-key: aucune clé de référence utilisée")
    print()

    # Précalcul stats traces (une fois pour tous les bytes)
    t_mean = traces.mean(axis=0)       # (T,)
    t_std  = traces.std(axis=0)        # (T,)
    t_std[t_std < 1e-10] = 1e-10

    bytes_to_attack = [args.byte] if args.byte is not None else range(16)
    recovered = []

    for b in bytes_to_attack:
        print(f"  Byte {b:2d} ... ", end='', flush=True)
        best_k, max_corrs = cpa_byte(traces, plaintexts, b, t_mean, t_std, model=args.model)
        recovered.append(best_k)

        verdict = ""
        if true_key is not None:
            true_corr = max_corrs[true_key[b]]
            if best_k == true_key[b]:
                verdict = "  ✓"
            else:
                verdict = f"  ✗ (attendu {true_key[b]:02x} corr={true_corr:.4f})"
        print(f"best={best_k:02x}  corr={max_corrs[best_k]:.4f}{verdict}")

    if len(list(bytes_to_attack)) == 16:
        print()
        print(f"Clé récupérée : {' '.join(f'{k:02x}' for k in recovered)}")
        if true_key is not None:
            n_correct = sum(r == t for r, t in zip(recovered, true_key))
            print(f"Bytes corrects : {n_correct}/16")


if __name__ == '__main__':
    main()
