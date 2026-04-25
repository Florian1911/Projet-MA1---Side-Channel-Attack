#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def get_top5(path: Path) -> list[dict]:
    j = load_json(path)
    top5 = j.get("top5_final_scores_repeat1", [])
    if not isinstance(top5, list):
        return []
    return top5


def confidence_from_top5(top5: list[dict], mode: str) -> float:
    if len(top5) < 2:
        return float("-inf")
    s = [float(x["score"]) for x in top5]
    gap12 = s[0] - s[1]
    if mode == "margin":
        return gap12
    if mode == "margin_norm":
        m = sum(s) / len(s)
        var = sum((x - m) ** 2 for x in s) / len(s)
        std = var ** 0.5
        return gap12 / (std + 1e-12)
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Final blind attack by fusing two byte-wise campaigns.")
    ap.add_argument("--campaign-a", required=True, help="First campaign directory (e.g. HW)")
    ap.add_argument("--campaign-b", required=True, help="Second campaign directory (e.g. SBOX)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument(
        "--confidence-mode",
        choices=["margin", "margin_norm"],
        default="margin_norm",
        help="How to compare confidence between campaigns per byte.",
    )
    ap.add_argument("--out", default="blind_fusion_summary.json")
    args = ap.parse_args()

    ca = Path(args.campaign_a)
    cb = Path(args.campaign_b)
    sa = load_json(ca / "summary.json")
    sb = load_json(cb / "summary.json")
    rows_a = {int(x["byte"]): x for x in sa.get("bytes", [])}
    rows_b = {int(x["byte"]): x for x in sb.get("bytes", [])}
    common = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if common != list(range(16)):
        raise ValueError(f"expected 16 common bytes, got {common}")

    per_byte = []
    rec = []
    for b in common:
        ka = ca / f"byte_{b:02d}" / f"keyrank_b{b:02d}.json"
        kb = cb / f"byte_{b:02d}" / f"keyrank_b{b:02d}.json"
        t5a = get_top5(ka)
        t5b = get_top5(kb)
        conf_a = confidence_from_top5(t5a, args.confidence_mode)
        conf_b = confidence_from_top5(t5b, args.confidence_mode)

        top1_a = int(t5a[0]["key"]) if t5a else -1
        top1_b = int(t5b[0]["key"]) if t5b else -1

        if conf_a >= conf_b:
            pick = args.label_a
            chosen = top1_a
            chosen_conf = conf_a
        else:
            pick = args.label_b
            chosen = top1_b
            chosen_conf = conf_b

        rec.append(chosen)
        per_byte.append(
            {
                "byte": b,
                "pick": pick,
                "top1_key_a": top1_a,
                "top1_key_b": top1_b,
                "conf_a": conf_a,
                "conf_b": conf_b,
                "chosen_key": chosen,
                "chosen_conf": chosen_conf,
            }
        )

    out = {
        "campaign_a": str(ca),
        "campaign_b": str(cb),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "confidence_mode": args.confidence_mode,
        "bytes": per_byte,
        "recovered_key_hex": "".join(f"{k & 0xFF:02X}" for k in rec),
        "recovered_key_hex_spaced": " ".join(f"{k & 0xFF:02X}" for k in rec),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("saved:", args.out)
    print("recovered_key:", out["recovered_key_hex_spaced"])


if __name__ == "__main__":
    main()
