#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import median


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def score_gap_from_keyrank(path: Path) -> float:
    j = load_json(path)
    top5 = j.get("top5_final_scores_repeat1", [])
    if len(top5) < 2:
        return float("-inf")
    return float(top5[0]["score"]) - float(top5[1]["score"])


def top1_key_from_keyrank(path: Path) -> int:
    j = load_json(path)
    top5 = j.get("top5_final_scores_repeat1", [])
    if not top5:
        return -1
    return int(top5[0]["key"])


def summarize_ranks(ranks: list[int | None]) -> dict:
    valid = [r for r in ranks if r is not None]
    if not valid:
        return {
            "mean_rank": None,
            "median_rank": None,
            "rank1_count": None,
            "rank_le_5": None,
            "rank_le_10": None,
            "rank_le_50": None,
            "valid_bytes": 0,
        }
    return {
        "mean_rank": float(sum(valid) / len(valid)),
        "median_rank": float(median(valid)),
        "rank1_count": int(sum(1 for r in valid if r == 1)),
        "rank_le_5": int(sum(1 for r in valid if r <= 5)),
        "rank_le_10": int(sum(1 for r in valid if r <= 10)),
        "rank_le_50": int(sum(1 for r in valid if r <= 50)),
        "valid_bytes": int(len(valid)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Blend two multisession campaigns byte-wise.")
    ap.add_argument("--campaign-a", required=True, help="Path to campaign dir A (contains summary.json)")
    ap.add_argument("--campaign-b", required=True, help="Path to campaign dir B (contains summary.json)")
    ap.add_argument(
        "--mode",
        choices=["margin", "oracle"],
        default="margin",
        help="margin=blind-friendly (choose bigger top1-top2 gap), oracle=choose smaller final rank",
    )
    ap.add_argument("--out", default="blend_summary.json", help="Output JSON summary")
    args = ap.parse_args()

    camp_a = Path(args.campaign_a)
    camp_b = Path(args.campaign_b)
    sum_a = load_json(camp_a / "summary.json")
    sum_b = load_json(camp_b / "summary.json")
    rows_a = {int(x["byte"]): x for x in sum_a.get("bytes", [])}
    rows_b = {int(x["byte"]): x for x in sum_b.get("bytes", [])}
    bytes_common = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if not bytes_common:
        raise ValueError("No common byte entries between campaigns.")

    picks = []
    ranks = []
    rec = []
    for b in bytes_common:
        ra = rows_a[b]
        rb = rows_b[b]
        rank_a = ra.get("final_avg_rank")
        rank_b = rb.get("final_avg_rank")

        ka = camp_a / f"byte_{b:02d}" / f"keyrank_b{b:02d}.json"
        kb = camp_b / f"byte_{b:02d}" / f"keyrank_b{b:02d}.json"
        gap_a = score_gap_from_keyrank(ka) if ka.exists() else float("-inf")
        gap_b = score_gap_from_keyrank(kb) if kb.exists() else float("-inf")
        top1_a = top1_key_from_keyrank(ka) if ka.exists() else -1
        top1_b = top1_key_from_keyrank(kb) if kb.exists() else -1

        if args.mode == "oracle":
            if rank_a is None and rank_b is None:
                pick = "A"
            elif rank_b is None:
                pick = "A"
            elif rank_a is None:
                pick = "B"
            else:
                pick = "A" if int(rank_a) <= int(rank_b) else "B"
        else:
            pick = "A" if gap_a >= gap_b else "B"

        if pick == "A":
            chosen_rank = int(rank_a) if rank_a is not None else None
            chosen_key = top1_a
        else:
            chosen_rank = int(rank_b) if rank_b is not None else None
            chosen_key = top1_b

        picks.append(
            {
                "byte": b,
                "pick": pick,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_chosen": chosen_rank,
                "gap_a": gap_a,
                "gap_b": gap_b,
                "top1_key_a": top1_a,
                "top1_key_b": top1_b,
                "top1_key_chosen": chosen_key,
            }
        )
        ranks.append(chosen_rank)
        rec.append(chosen_key)

    out = {
        "campaign_a": str(camp_a),
        "campaign_b": str(camp_b),
        "mode": args.mode,
        "metrics": summarize_ranks(ranks),
        "bytes": picks,
        "recovered_key_hex": "".join(f"{k & 0xFF:02X}" for k in rec),
        "recovered_key_hex_spaced": " ".join(f"{k & 0xFF:02X}" for k in rec),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("saved:", args.out)
    print("recovered_key:", out["recovered_key_hex_spaced"])
    print("metrics:", out["metrics"])


if __name__ == "__main__":
    main()
