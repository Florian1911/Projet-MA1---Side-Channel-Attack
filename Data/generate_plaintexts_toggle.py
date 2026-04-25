#!/usr/bin/env python3
import argparse
import numpy as np


def write_header(path: str, pts: np.ndarray) -> None:
    n = pts.shape[0]
    with open(path, 'w') as f:
        f.write('#ifndef PLAINTEXTS_DATA_H\n')
        f.write('#define PLAINTEXTS_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'#define N_PLAINTEXTS {n}u\n\n')
        f.write(f'static const uint8_t PLAINTEXTS[{n}][16] = {{\n')
        for i, pt in enumerate(pts):
            vals = ','.join(f'0x{b:02x}' for b in pt)
            sep = ',' if i < n - 1 else ''
            f.write(f'  {{{vals}}}{sep}\n')
        f.write('};\n\n')
        f.write('#endif /* PLAINTEXTS_DATA_H */\n')


def main():
    ap = argparse.ArgumentParser(description='Plaintexts toggles pour test leakage source')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--byte', type=int, default=0)
    ap.add_argument('--low', type=lambda x: int(x, 0), default=0x00)
    ap.add_argument('--high', type=lambda x: int(x, 0), default=0xFF)
    ap.add_argument('--out-npy', default='plaintexts_no_uart.npy')
    ap.add_argument('--out-h', default='plaintexts_data.h')
    args = ap.parse_args()

    if not (0 <= args.byte < 16):
        raise ValueError('--byte doit etre entre 0 et 15')

    pts = np.zeros((args.n, 16), dtype=np.uint8)
    vals = np.empty((args.n,), dtype=np.uint8)
    vals[0::2] = np.uint8(args.low & 0xFF)
    vals[1::2] = np.uint8(args.high & 0xFF)
    pts[:, args.byte] = vals

    np.save(args.out_npy, pts)
    write_header(args.out_h, pts)

    print(f'[OK] toggle plaintexts generes: n={args.n}, byte={args.byte}, low=0x{args.low&0xFF:02x}, high=0x{args.high&0xFF:02x}')
    print(f'[OK] npy: {args.out_npy}')
    print(f'[OK] header: {args.out_h}')


if __name__ == '__main__':
    main()
