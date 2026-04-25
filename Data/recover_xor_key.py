#!/usr/bin/env python3
import argparse
import glob
import re
import numpy as np

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)


def preprocess(tr):
    t = tr.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    t = t - np.outer((t @ x) / np.dot(x, x), x)
    return t


def cpa_scores_hw_in(tr, pbyte):
    tc = tr - tr.mean(axis=0, keepdims=True)
    tstd = tr.std(axis=0, ddof=1) + 1e-15
    n = tr.shape[0]
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        h = HW[np.bitwise_xor(pbyte, k)]
        hc = h - h.mean()
        hstd = h.std(ddof=1) + 1e-15
        corr = (hc @ tc) / ((n - 1) * hstd * tstd)
        scores[k] = np.max(np.abs(corr))
    return scores


def byte_from_name(path):
    m = re.search(r'byte[_-]?(\d{1,2})', path, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:^|[_-])b(\d{1,2})(?:[_-]|\.|$)', path, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description='Recover XOR-byte key candidates from datasets')
    ap.add_argument('--glob', dest='glob_pat', default='xor_b*.npz')
    ap.add_argument('--n-traces', type=int, default=5000)
    ap.add_argument('--start-min', type=int, default=0)
    ap.add_argument('--start-max', type=int, default=1200)
    ap.add_argument('--start-step', type=int, default=100)
    ap.add_argument('--lengths', default='200,300,400,500,600,800,1000')
    ap.add_argument('--topk-vote', type=int, default=8,
                    help='Top-k keys per window used for rank vote (default: 8)')
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(',') if x.strip()]
    files = sorted(glob.glob(args.glob_pat))
    if not files:
        raise SystemExit(f'No files matching {args.glob_pat}')

    key = ['??'] * 16
    print('Per-byte results (k and k^FF candidates):')

    for f in files:
        d = np.load(f)
        tr_all = d['traces'][:args.n_traces]
        pt_all = d['plaintexts'][:args.n_traces]
        b = byte_from_name(f)
        if b is None or not (0 <= b < 16):
            print(f'  {f}: skip (cannot infer byte index from filename)')
            continue
        pbyte = pt_all[:, b].astype(np.uint8)

        best = None
        vote = np.zeros(256, dtype=np.float64)
        score_sum = np.zeros(256, dtype=np.float64)
        n_windows = 0
        for a in range(args.start_min, args.start_max + 1, args.start_step):
            for L in lengths:
                c = a + L
                if c > tr_all.shape[1]:
                    continue
                tr = preprocess(tr_all[:, a:c])
                scores = cpa_scores_hw_in(tr, pbyte)
                n_windows += 1
                score_sum += scores
                order = np.argsort(scores)[::-1]
                k_use = max(1, min(args.topk_vote, 256))
                # Rank vote: top-1 gets k_use points, top-k gets 1 point.
                for pos, kk in enumerate(order[:k_use]):
                    vote[int(kk)] += (k_use - pos)
                k = int(np.argmax(scores))
                sc = float(scores[k])
                if best is None or sc > best[0]:
                    best = (sc, k, a, c)

        sc, k, a, c = best
        k_best_vote = int(np.argmax(vote))
        k_best_mean = int(np.argmax(score_sum / max(1, n_windows)))
        k2 = k_best_vote ^ 0xFF
        key[b] = f'{k_best_vote:02x}/{k2:02x}'
        print(f'  b{b:02d}:')
        print(f'    peak-window: win[{a}:{c}] score={sc:.5f} candidates=0x{k:02x}/0x{k^0xFF:02x}')
        print(f'    vote-best:   key=0x{k_best_vote:02x}/0x{k2:02x}  vote={vote[k_best_vote]:.1f}')
        print(f'    mean-best:   key=0x{k_best_mean:02x}/0x{k_best_mean^0xFF:02x}')

    print('\nRecovered key candidates by byte (k/k^FF):')
    print(' '.join(key))


if __name__ == '__main__':
    main()
