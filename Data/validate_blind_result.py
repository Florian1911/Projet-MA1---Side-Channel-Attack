#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_hex_key(s: str) -> str:
    t = s.strip().replace(" ", "").replace(":", "").replace(",", "").upper()
    if len(t) != 32:
        raise ValueError("Key must be 32 hex chars (16 bytes).")
    int(t, 16)
    return t


def load_pred_from_json(path: Path) -> str:
    j = json.loads(path.read_text())
    if "recovered_key_hex" in j:
        return parse_hex_key(j["recovered_key_hex"])
    if "recovered_key" in j and isinstance(j["recovered_key"], list):
        return "".join(f"{int(x) & 0xFF:02X}" for x in j["recovered_key"])
    raise ValueError(f"Unsupported JSON format for key extraction: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate predicted AES-128 key against ground truth.")
    ap.add_argument("--pred-json", required=True, help="JSON containing recovered key")
    ap.add_argument("--true-key-hex", required=True, help="Ground truth key (hex)")
    args = ap.parse_args()

    pred = load_pred_from_json(Path(args.pred_json))
    true = parse_hex_key(args.true_key_hex)
    ok = [pred[i:i + 2] == true[i:i + 2] for i in range(0, 32, 2)]
    n = sum(ok)

    print("TRUE :", true)
    print("PRED :", pred)
    print(f"MATCH: {n}/16")
    bad = [i // 2 for i in range(0, 32, 2) if pred[i:i + 2] != true[i:i + 2]]
    if bad:
        print("BAD_BYTES:", bad)
    else:
        print("BAD_BYTES: []")


if __name__ == "__main__":
    main()
