#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

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
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)
KEYS = np.arange(256, dtype=np.uint8)


def preprocess(traces):
    t = traces.astype(np.float64, copy=False)
    t -= t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t -= np.outer((t @ x) / np.dot(x, x), x)
    return t


def model_hyp(pbyte, model):
    x = np.bitwise_xor(pbyte[:, None], KEYS[None, :])
    s = SBOX[x]
    if model == 'hw_sbox':
        h = HW[s]
    elif model == 'hd_ps':
        h = HW[np.bitwise_xor(s, pbyte[:, None])]
    elif model == 'hw_in':
        h = HW[x]
    elif model == 'mix50':
        h = 0.5 * HW[s] + 0.5 * HW[np.bitwise_xor(s, pbyte[:, None])]
    elif model == 'mix75':
        h = 0.75 * HW[s] + 0.25 * HW[np.bitwise_xor(s, pbyte[:, None])]
    elif model == 'bit0':
        h = ((s & 1) != 0).astype(np.float64)
    elif model == 'bit7':
        h = ((s >> 7) & 1).astype(np.float64)
    else:
        raise ValueError(model)
    return h.T


def cpa_peak(traces, plains_byte, model):
    tc = traces - traces.mean(axis=0, keepdims=True)
    tstd = traces.std(axis=0, ddof=1) + 1e-15
    h = model_hyp(plains_byte, model)
    hc = h - h.mean(axis=1, keepdims=True)
    hs = h.std(axis=1, ddof=1, keepdims=True) + 1e-15
    corr = (hc @ tc) / ((tc.shape[0] - 1) * hs * tstd[None, :])
    ac = np.abs(corr)
    scores = ac.max(axis=1)
    pois = ac.argmax(axis=1)
    order = np.argsort(scores)[::-1]
    k1, k2 = int(order[0]), int(order[1])
    margin = float(scores[k1] - scores[k2])
    return k1, k2, margin, int(pois[k1]), float(scores[k1])


def corr_key_at_poi(traces_col, plains_byte, key, model):
    pb = plains_byte
    x = np.bitwise_xor(pb, np.uint8(key))
    s = SBOX[x]
    if model == 'hw_sbox':
        h = HW[s]
    elif model == 'hd_ps':
        h = HW[np.bitwise_xor(s, pb)]
    elif model == 'hw_in':
        h = HW[x]
    elif model == 'mix50':
        h = 0.5 * HW[s] + 0.5 * HW[np.bitwise_xor(s, pb)]
    elif model == 'mix75':
        h = 0.75 * HW[s] + 0.25 * HW[np.bitwise_xor(s, pb)]
    elif model == 'bit0':
        h = ((s & 1) != 0).astype(np.float64)
    elif model == 'bit7':
        h = ((s >> 7) & 1).astype(np.float64)
    else:
        raise ValueError(model)

    hc = h - h.mean()
    tc = traces_col - traces_col.mean()
    denom = (np.sqrt((hc * hc).sum()) * np.sqrt((tc * tc).sum()) + 1e-15)
    return float(abs((hc @ tc) / denom))


def load_npz(path):
    d = np.load(path)
    return preprocess(d['traces']), d['plaintexts'].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', default='unknown_B_calib5k_nokey.npz')
    ap.add_argument('--attacks', nargs='+', default=['unknown_B_attack15k_nokey.npz', 'unknown_B_attack18k_nokey.npz', 'unknown_B_20k_aligned_nokey.npz', 'unknown_B_20k_aligned_nokey_shiftp8.npz'])
    ap.add_argument('--probe-n', type=int, default=1800)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--out', default='full_blind_hypervote_fast_summary.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    trc, ptc = load_npz(args.calib)
    ncal = trc.shape[0]
    idx = rng.choice(ncal, size=min(args.probe_n, ncal), replace=False)
    trc_probe, ptc_probe = trc[idx], ptc[idx]

    windows = [(600, 1800), (800, 2200), (1000, 2600), (1200, 3000), (0, 3968)]
    models = ['hw_sbox', 'hd_ps', 'mix50', 'mix75', 'hw_in', 'bit0', 'bit7']

    # stage 1: candidates from calibration probe only
    per_byte_candidates = []
    for b in range(16):
        rows = []
        pb = ptc_probe[:, b]
        for m in models:
            for w0, w1 in windows:
                k1, k2, margin, poi, s1 = cpa_peak(trc_probe[:, w0:w1], pb, m)
                rows.append({
                    'model': m, 'window': [w0, w1], 'k1': k1, 'k2': k2,
                    'margin': margin, 'score_top1': s1, 'poi': poi + w0,
                })
        rows.sort(key=lambda r: r['margin'], reverse=True)
        per_byte_candidates.append(rows[:8])

    # stage 2: attack vote (probe re-detection + full score at POI)
    attacks = []
    for p in args.attacks:
        tr, pt = load_npz(p)
        ia = rng.choice(tr.shape[0], size=min(args.probe_n, tr.shape[0]), replace=False)
        attacks.append((p, tr, pt, ia))

    recovered = []
    byte_rows = []

    for b in range(16):
        cand = per_byte_candidates[b]
        vote = np.zeros(256, dtype=np.float64)
        evidence = []

        for p, tr, pt, ia in attacks:
            tr_probe = tr[ia]
            pb_probe = pt[ia, b]
            pb_full = pt[:, b]

            for c in cand:
                m = c['model']
                w0, w1 = c['window']
                # re-detect on this attack dataset (probe subset)
                k1, k2, margin, poi, s1 = cpa_peak(tr_probe[:, w0:w1], pb_probe, m)
                poi_abs = poi + w0

                # full-dataset rescore at detected POI for top candidates
                cf1 = corr_key_at_poi(tr[:, poi_abs], pb_full, k1, m)
                cf2 = corr_key_at_poi(tr[:, poi_abs], pb_full, k2, m)
                w = max(0.0, cf1 - cf2) * max(0.0, margin + 1e-9)
                vote[k1] += w

                evidence.append({
                    'dataset': p, 'model': m, 'window': [w0, w1],
                    'k1': k1, 'k2': k2, 'probe_margin': margin,
                    'full_corr_k1': cf1, 'full_corr_k2': cf2,
                    'poi': int(poi_abs), 'weight': w,
                })

        order = np.argsort(vote)[::-1]
        kf, k2 = int(order[0]), int(order[1])
        recovered.append(kf)
        tot = float(vote.sum() + 1e-12)
        byte_rows.append({
            'byte': b,
            'recovered_key': kf,
            'recovered_key_hex': f'{kf:02X}',
            'runner_up': k2,
            'runner_up_hex': f'{k2:02X}',
            'vote_top': float(vote[kf]),
            'vote_second': float(vote[k2]),
            'vote_share': float(vote[kf] / tot),
            'confidence_margin_ratio': float((vote[kf] - vote[k2]) / (vote[kf] + 1e-12)),
            'top3': [
                {'key': int(k), 'key_hex': f'{int(k):02X}', 'score': float(vote[k])}
                for k in order[:3]
            ],
            'evidence_count': len(evidence),
            'evidence': evidence,
            'calib_top_candidates': cand,
        })

    out = {
        'calib': args.calib,
        'attacks': args.attacks,
        'probe_n': args.probe_n,
        'seed': args.seed,
        'recovered_key': recovered,
        'recovered_key_hex': ''.join(f'{k:02X}' for k in recovered),
        'bytes': byte_rows,
    }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print('Recovered key (full blind, hypervote fast):', out['recovered_key_hex'])
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
