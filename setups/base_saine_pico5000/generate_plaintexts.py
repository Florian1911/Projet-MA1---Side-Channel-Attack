#!/usr/bin/env python3
"""Genere des plaintexts AES et un header C coherent pour la cible STM32."""

import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Generation plaintexts SCA")
    ap.add_argument("--n", type=int, default=5000, help="Nombre de plaintexts")
    ap.add_argument("--seed", type=int, default=1234, help="Graine RNG")
    ap.add_argument("--out-npy", default="plaintexts_no_uart.npy")
    ap.add_argument("--out-h", default="plaintexts_data.h")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pts = rng.integers(0, 256, size=(args.n, 16), dtype=np.uint8)

    np.save(args.out_npy, pts)

    with open(args.out_h, "w", encoding="ascii") as f:
        f.write("#ifndef PLAINTEXTS_DATA_H\n")
        f.write("#define PLAINTEXTS_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define N_PLAINTEXTS {args.n}u\n\n")
        f.write(f"static const uint8_t PLAINTEXTS[{args.n}][16] = {{\n")
        for i, row in enumerate(pts):
            values = ",".join(f"0x{b:02x}" for b in row)
            suffix = "," if i < args.n - 1 else ""
            f.write(f"  {{{values}}}{suffix}\n")
        f.write("};\n\n")
        f.write("#endif\n")

    print(f"[OK] plaintexts npy: {args.out_npy}")
    print(f"[OK] header C      : {args.out_h}")


if __name__ == "__main__":
    main()
