#!/usr/bin/env python3
"""Fusionne plusieurs datasets NPZ (traces/plaintexts/overflows) dans l'ordre."""

import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge datasets NPZ")
    ap.add_argument("inputs", nargs="+", help="Fichiers .npz a fusionner dans l'ordre")
    ap.add_argument("--output", required=True, help="Fichier .npz de sortie")
    args = ap.parse_args()

    traces_parts = []
    pts_parts = []
    ov_parts = []
    key_ref = None

    for p in args.inputs:
        d = np.load(p)
        traces_parts.append(d["traces"])
        pts_parts.append(d["plaintexts"])
        ov_parts.append(d["overflows"])
        key = d["key"]
        if key_ref is None:
            key_ref = key
        elif not np.array_equal(key_ref, key):
            raise ValueError(f"Cle differente detectee dans {p}")

    traces = np.concatenate(traces_parts, axis=0)
    pts = np.concatenate(pts_parts, axis=0)
    overflows = np.concatenate(ov_parts, axis=0)

    np.savez(args.output, traces=traces, plaintexts=pts, key=key_ref, overflows=overflows)

    print(f"[OK] output: {args.output}")
    print(f"[OK] shape : {traces.shape}")
    print(f"[OK] overflows nonzero: {int((overflows != 0).sum())}")


if __name__ == "__main__":
    main()
