#!/usr/bin/env python3
import argparse, json, numpy as np


def moving_avg(x, w):
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, k, mode='same')


def detrend_linear(y):
    x = np.linspace(-1.0, 1.0, y.size, dtype=np.float64)
    a = np.dot(y, x) / (np.dot(x, x) + 1e-15)
    return y - a * x


def pick_activity_window(env, q=97.0, pad=60):
    thr = np.percentile(env, q)
    idx = np.flatnonzero(env >= thr)
    if idx.size == 0:
        return 0, env.size
    s = max(0, int(idx[0]) - pad)
    e = min(env.size, int(idx[-1]) + pad + 1)
    return s, e


def two_means_threshold(v, iters=20):
    a = float(np.min(v)); b = float(np.max(v))
    if abs(b-a) < 1e-15:
        return a
    for _ in range(iters):
        da = np.abs(v-a); db = np.abs(v-b)
        ma = v[da <= db]; mb = v[da > db]
        if ma.size: a = float(np.mean(ma))
        if mb.size: b = float(np.mean(mb))
    return 0.5*(a+b)


def bits_to_int(bits):
    v=0
    for b in bits:
        v=(v<<1)|int(b)
    return v


def recover_bits_from_trace(m, bitlen=23, smooth=9, start=-1, end=-1):
    m = m - np.mean(m)
    m = detrend_linear(m)
    der = np.diff(m, prepend=m[0])
    env = moving_avg(np.abs(der), smooth)
    if start >= 0 and end > start:
        s,e = int(start), int(end)
    else:
        s,e = pick_activity_window(env)
    seg = m[s:e]
    env_seg = env[s:e]
    if seg.size < bitlen*8:
        return None
    edges = np.linspace(0, seg.size, bitlen+1).astype(int)
    feats=[]
    for i in range(bitlen):
        a,b = edges[i], edges[i+1]
        ch = env_seg[a:b]
        if ch.size < 4:
            feats.append(0.0); continue
        cut = int(0.6*ch.size)
        e1 = float(np.mean(ch[:cut]))
        e2 = float(np.mean(ch[cut:]))
        feats.append(e2-e1)
    feats = np.asarray(feats, dtype=np.float64)
    thr = two_means_threshold(feats)
    bits = (feats > thr).astype(int)
    bits[0]=1
    return bits, feats.tolist(), (s,e)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--trace-key', default='traces')
    ap.add_argument('--bitlen', type=int, default=23)
    ap.add_argument('--runs', type=int, default=31)
    ap.add_argument('--sub-traces', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--start', type=int, default=-1)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--out', default='rsa_spa_vote_result.json')
    args=ap.parse_args()

    d=np.load(args.npz)
    tr=d[args.trace_key].astype(np.float64)
    n=tr.shape[0]
    rng=np.random.default_rng(args.seed)

    smooth_grid=[5,7,9,11,13]
    bit_votes=np.zeros((args.bitlen,2), dtype=np.int64)
    run_out=[]

    for r in range(args.runs):
        mcount=min(args.sub_traces,n)
        idx=rng.choice(n,size=mcount,replace=False)
        m=np.median(tr[idx],axis=0)
        smooth=smooth_grid[r % len(smooth_grid)]
        rec=recover_bits_from_trace(m, bitlen=args.bitlen, smooth=smooth, start=args.start, end=args.end)
        if rec is None:
            continue
        bits, feats, win = rec
        for i,b in enumerate(bits.tolist()):
            bit_votes[i,b]+=1
        run_out.append({
            'run': r,
            'smooth': smooth,
            'window': {'start': int(win[0]), 'end': int(win[1])},
            'bits': ''.join(str(int(x)) for x in bits.tolist()),
            'd_est_hex': f"0x{bits_to_int(bits):X}"
        })

    final_bits=[]
    conf=[]
    for i in range(args.bitlen):
        z,o=int(bit_votes[i,0]),int(bit_votes[i,1])
        if o>=z:
            final_bits.append(1)
            conf.append(o/max(1,o+z))
        else:
            final_bits.append(0)
            conf.append(z/max(1,o+z))
    final_bits[0]=1
    d_est=bits_to_int(final_bits)

    out={
        'npz': args.npz,
        'runs_requested': args.runs,
        'runs_valid': len(run_out),
        'bitlen': args.bitlen,
        'bit_votes': bit_votes.tolist(),
        'bit_confidence': conf,
        'bits_msb_to_lsb': ''.join(str(int(b)) for b in final_bits),
        'd_est_int': int(d_est),
        'd_est_hex': f"0x{d_est:X}",
        'runs': run_out,
    }
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2)
    print('saved:',args.out)
    print('runs_valid:',len(run_out))
    print('bits:',out['bits_msb_to_lsb'])
    print('d_est:',out['d_est_int'],out['d_est_hex'])

if __name__=='__main__':
    main()
