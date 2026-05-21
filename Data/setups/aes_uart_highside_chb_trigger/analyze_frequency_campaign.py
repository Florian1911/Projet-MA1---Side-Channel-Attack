#!/usr/bin/env python3
"""Compare plusieurs datasets de campagne frequence AES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def snr_by_class(traces: np.ndarray, classes: np.ndarray) -> np.ndarray:
    traces64 = traces.astype(np.float64, copy=False)
    classes = classes.astype(np.int16, copy=False)
    total_mean = traces64.mean(axis=0)
    between = np.zeros(traces64.shape[1], dtype=np.float64)
    within = np.zeros(traces64.shape[1], dtype=np.float64)
    n_total = float(len(traces64))

    for c in np.unique(classes):
        x = traces64[classes == c]
        if len(x) < 2:
            continue
        mu = x.mean(axis=0)
        between += len(x) * (mu - total_mean) ** 2
        within += ((x - mu) ** 2).sum(axis=0)

    between /= max(1.0, n_total - 1.0)
    within /= max(1.0, n_total - len(np.unique(classes)))
    return between / np.maximum(within, 1e-12)


def estimate_trigger_width(trigger_mv: np.ndarray, pre_trigger: int) -> dict:
    mean_trig = trigger_mv.mean(axis=0)
    lo = float(np.percentile(mean_trig[:max(10, pre_trigger)], 10))
    hi = float(np.percentile(mean_trig, 99))
    threshold = lo + 0.5 * (hi - lo)
    idx = np.flatnonzero(mean_trig > threshold)
    if len(idx) == 0:
        return {"trigger_width_samples": 0, "trigger_start_sample": None, "trigger_end_sample": None}
    return {
        "trigger_width_samples": int(idx[-1] - idx[0] + 1),
        "trigger_start_sample": int(idx[0]),
        "trigger_end_sample": int(idx[-1]),
    }


def summarize_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        traces = data["traces"].astype(np.float32)
        labels = data["labels"].astype(np.uint8)
        trigger_mv = data["trigger_mv"].astype(np.float32) if "trigger_mv" in data else None
        dt_ns = float(data["meta_dt_ns"]) if "meta_dt_ns" in data else None

    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pre_trigger = int(meta.get("pre_trigger", meta.get("pre_trigger_samples", traces.shape[1] // 8)))

    pre = traces[:, :max(10, pre_trigger)]
    active = traces[:, max(0, pre_trigger):]
    centered = traces - traces[:, :max(10, pre_trigger)].mean(axis=1, keepdims=True)
    snr = snr_by_class(centered, HW[labels])

    out = {
        "dataset": str(path),
        "clock_mhz": meta.get("clock_mhz"),
        "n_traces": int(traces.shape[0]),
        "num_samples": int(traces.shape[1]),
        "dt_ns": dt_ns,
        "pre_noise_std_mv": float(pre.std()),
        "active_std_mv": float(active.std()) if active.size else float(traces.std()),
        "mean_trace_peak_to_peak_mv": float(np.ptp(traces.mean(axis=0))),
        "snr_hw_max": float(np.max(snr)),
        "snr_hw_argmax_sample": int(np.argmax(snr)),
    }
    if dt_ns is not None:
        out["snr_hw_argmax_us"] = float(np.argmax(snr) * dt_ns / 1000.0)

    if trigger_mv is not None:
        trig = estimate_trigger_width(trigger_mv, pre_trigger)
        out.update(trig)
        if dt_ns is not None and trig["trigger_width_samples"]:
            out["trigger_width_us"] = float(trig["trigger_width_samples"] * dt_ns / 1000.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume une campagne de frequence AES")
    ap.add_argument("datasets", nargs="+", help="Fichiers .npz a comparer")
    ap.add_argument("--output-json", default="frequency_campaign_summary.json")
    args = ap.parse_args()

    rows = [summarize_dataset(Path(p)) for p in args.datasets]
    rows.sort(key=lambda r: (r["clock_mhz"] is None, r["clock_mhz"] or 0))

    print("clock_mhz,n_traces,noise_std_mv,active_std_mv,mean_ptp_mv,snr_hw_max,snr_sample,trigger_width_us")
    for row in rows:
        print(
            f"{row.get('clock_mhz')},{row['n_traces']},"
            f"{row['pre_noise_std_mv']:.5f},{row['active_std_mv']:.5f},"
            f"{row['mean_trace_peak_to_peak_mv']:.5f},{row['snr_hw_max']:.6f},"
            f"{row['snr_hw_argmax_sample']},{row.get('trigger_width_us')}"
        )

    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] resume: {args.output_json}")


if __name__ == "__main__":
    main()
