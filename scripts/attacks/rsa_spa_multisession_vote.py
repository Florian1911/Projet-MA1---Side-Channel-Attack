#!/usr/bin/env python3
import argparse, glob, json, numpy as np


def moving_avg(x, w):
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, k, mode='same')


def detrend_linear(y):
    x = np.linspace(-1.0, 1.0, y.size, dtype=np.float64)
    a = np.dot(y, x) / (np.dot(x, x) + 1e-15)
    return y - a * x


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


def recover_one(npz_path, trace_key, bitlen, runs, sub_traces, seed, start, end):
    d=np.load(npz_path)
    tr=d[trace_key].astype(np.float64)
    n=tr.shape[0]
    rng=np.random.default_rng(seed)

    smooth_grid=[5,7,9,11,13]
    bit_votes=np.zeros((bitlen,2),dtype=np.int64)

    for r in range(runs):
        idx=rng.choice(n,size=min(sub_traces,n),replace=False)
        m=np.median(tr[idx],axis=0)
        m=m-np.mean(m)
        m=detrend_linear(m)
        der=np.diff(m,prepend=m[0])
        env=moving_avg(np.abs(der), smooth_grid[r % len(smooth_grid)])

        s,e=start,end
        if s<0 or e<=s:
            # fallback: centered high-energy zone
            thr=np.percentile(env,97.0)
            ii=np.flatnonzero(env>=thr)
            if ii.size==0:
                s,e=0,env.size
            else:
                s=max(0,int(ii[0])-60); e=min(env.size,int(ii[-1])+61)

        seg=env[s:e]
        if seg.size < bitlen*8:
            continue

        edges=np.linspace(0,seg.size,bitlen+1).astype(int)
        feats=[]
        for i in range(bitlen):
            a,b=edges[i],edges[i+1]
            ch=seg[a:b]
            if ch.size<4:
                feats.append(0.0); continue
            cut=int(0.6*ch.size)
            feats.append(float(np.mean(ch[cut:]) - np.mean(ch[:cut])))

        feats=np.asarray(feats,dtype=np.float64)
        thr=two_means_threshold(feats)
        bits=(feats>thr).astype(int)
        bits[0]=1
        for i,b in enumerate(bits.tolist()):
            bit_votes[i,b]+=1

    out_bits=[]
    out_conf=[]
    for i in range(bitlen):
        z,o=int(bit_votes[i,0]),int(bit_votes[i,1])
        if o>=z:
            out_bits.append(1); out_conf.append(o/max(1,o+z))
        else:
            out_bits.append(0); out_conf.append(z/max(1,o+z))
    out_bits[0]=1
    return out_bits, out_conf, bit_votes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--glob', required=True, help='e.g. "rsa_spa_s*_20k_al.npz"')
    ap.add_argument('--trace-key', default='traces')
    ap.add_argument('--bitlen', type=int, default=23)
    ap.add_argument('--runs', type=int, default=61)
    ap.add_argument('--sub-traces', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--start', type=int, default=1556)
    ap.add_argument('--end', type=int, default=1868)
    ap.add_argument('--out', default='rsa_spa_multisession_vote.json')
    args=ap.parse_args()

    files=sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit('no files matched')

    global_score=np.zeros((args.bitlen,2),dtype=np.float64)
    sessions=[]

    for i,f in enumerate(files):
        bits,conf,votes = recover_one(
            f,args.trace_key,args.bitlen,args.runs,args.sub_traces,args.seed+i,args.start,args.end
        )
        for b in range(args.bitlen):
            global_score[b, bits[b]] += conf[b]
        sessions.append({
            'file': f,
            'bits_msb_to_lsb': ''.join(str(int(x)) for x in bits),
            'd_est_hex': hex(bits_to_int(bits)),
            'bit_confidence': conf,
            'bit_votes': votes.tolist(),
        })

    final=[]
    final_conf=[]
    for b in range(args.bitlen):
        z,o=float(global_score[b,0]),float(global_score[b,1])
        if o>=z:
            final.append(1); final_conf.append(o/max(1e-15,o+z))
        else:
            final.append(0); final_conf.append(z/max(1e-15,o+z))
    final[0]=1

    out={
        'files': files,
        'trace_key': args.trace_key,
        'bitlen': args.bitlen,
        'final_bits_msb_to_lsb': ''.join(str(int(x)) for x in final),
        'final_d_est_int': int(bits_to_int(final)),
        'final_d_est_hex': hex(bits_to_int(final)),
        'final_bit_confidence': final_conf,
        'sessions': sessions,
    }
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2)
    print('saved:',args.out)
    print('files:',len(files))
    print('final_bits:',out['final_bits_msb_to_lsb'])
    print('final_d_est:',out['final_d_est_hex'])

if __name__=='__main__':
    main()
