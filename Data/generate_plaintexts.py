#!/usr/bin/env python3
"""
Génère N plaintexts aléatoires et les exporte :
  - plaintexts_no_uart.npy  → utilisé par acquire_no_uart.py / cpa_attack.py
  - plaintexts_data.h       → à copier dans Core/Inc/ du projet STM32CubeIDE

Usage : python generate_plaintexts.py --n 5000 --seed 1234
"""

import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',       type=int, default=5000,
                        help='Nombre de plaintexts (défaut: 5000)')
    parser.add_argument('--seed',    type=int, default=1234,
                        help='Graine RNG (défaut: 1234)')
    parser.add_argument('--out-npy', default='plaintexts_no_uart.npy')
    parser.add_argument('--out-h',   default='plaintexts_data.h')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pts = rng.integers(0, 256, size=(args.n, 16), dtype=np.uint8)

    np.save(args.out_npy, pts)
    print(f"[OK] {args.n} plaintexts → {args.out_npy}")

    with open(args.out_h, 'w') as f:
        f.write("#ifndef PLAINTEXTS_DATA_H\n")
        f.write("#define PLAINTEXTS_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define N_PLAINTEXTS {args.n}u\n\n")
        f.write(f"static const uint8_t PLAINTEXTS[{args.n}][16] = {{\n")
        for i, pt in enumerate(pts):
            vals = ','.join(f'0x{b:02x}' for b in pt)
            sep  = ',' if i < args.n - 1 else ''
            f.write(f"  {{{vals}}}{sep}\n")
        f.write("};\n\n")
        f.write("#endif /* PLAINTEXTS_DATA_H */\n")
    print(f"[OK] C header → {args.out_h}")
    print()
    print(f"  → Copie '{args.out_h}' dans Core/Inc/ de ton projet STM32CubeIDE")
    print(f"  → Compile et flashe main_no_uart.c")
    print(f"  → Lance : python acquire_no_uart.py --n-traces {args.n}")


if __name__ == '__main__':
    main()
