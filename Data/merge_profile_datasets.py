import argparse
from pathlib import Path

import numpy as np


def parse_paths(raw: str):
    return [p.strip() for p in raw.split(',') if p.strip()]


def main():
    ap = argparse.ArgumentParser(description="Merge multiple aligned profiling datasets (.npz)")
    ap.add_argument("--inputs", required=True, help="Comma-separated .npz paths")
    ap.add_argument("--out", default="profile_merged_multisession.npz")
    ap.add_argument(
        "--key",
        default="",
        help="Optional AES-128 key in hex (32 chars) to store in output under 'key'",
    )
    args = ap.parse_args()

    paths = parse_paths(args.inputs)
    if not paths:
        raise ValueError("--inputs is empty")

    traces_list = []
    plains_list = []
    over_list = []
    has_over = True

    for p in paths:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(pp)
        d = np.load(str(pp))
        if "traces" not in d.files or "plaintexts" not in d.files:
            raise ValueError(f"{pp} must contain traces and plaintexts")
        traces_list.append(d["traces"])
        plains_list.append(d["plaintexts"])
        if "overflows" in d.files:
            over_list.append(d["overflows"])
        else:
            has_over = False

    traces = np.concatenate(traces_list, axis=0)
    plains = np.concatenate(plains_list, axis=0)

    save = {"traces": traces, "plaintexts": plains}

    if has_over and len(over_list) == len(paths):
        save["overflows"] = np.concatenate(over_list, axis=0)

    if args.key:
        k = args.key.strip().lower().replace("0x", "")
        for sep in [",", ":", "-", " "]:
            k = k.replace(sep, "")
        if len(k) != 32:
            raise ValueError("--key must be 16 bytes (32 hex chars)")
        save["key"] = np.array([int(k[i:i + 2], 16) for i in range(0, 32, 2)], dtype=np.uint8)

    np.savez(args.out, **save)
    print(f"saved: {args.out} | traces={traces.shape} plaintexts={plains.shape}")


if __name__ == "__main__":
    main()
