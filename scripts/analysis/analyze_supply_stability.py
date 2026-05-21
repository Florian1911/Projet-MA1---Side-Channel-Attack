#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metadata(npz_path: Path) -> dict:
    meta_path = npz_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def robust_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=np.float64)
    xc = x - x.mean()
    yc = y.astype(np.float64) - y.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def summarize_block(name: str, block: np.ndarray) -> dict:
    per_trace_mean = block.mean(axis=1)
    per_trace_std = block.std(axis=1)
    per_trace_p2p = np.ptp(block, axis=1)
    return {
        "name": name,
        "mean_mv": float(per_trace_mean.mean()),
        "trace_mean_std_mv": float(per_trace_mean.std(ddof=1)),
        "trace_mean_min_mv": float(per_trace_mean.min()),
        "trace_mean_max_mv": float(per_trace_mean.max()),
        "trace_mean_p2p_mv": float(np.ptp(per_trace_mean)),
        "within_trace_std_median_mv": float(np.median(per_trace_std)),
        "within_trace_p2p_median_mv": float(np.median(per_trace_p2p)),
        "within_trace_p2p_p95_mv": float(np.percentile(per_trace_p2p, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze supply stability during AES capture windows")
    ap.add_argument("--npz", default="dataset_aes_sca_highside.npz")
    ap.add_argument("--out-prefix", default="supply_stability_highside")
    ap.add_argument("--pre-end", type=int, default=200)
    ap.add_argument("--active-start", type=int, default=200)
    ap.add_argument("--active-end", type=int, default=1800)
    ap.add_argument("--post-start", type=int, default=1800)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path)
    if "traces_a" not in d.files or "traces_b" not in d.files:
        raise ValueError("Dataset must contain traces_a and traces_b voltage channels")

    ch_a = d["traces_a"].astype(np.float64)
    ch_b = d["traces_b"].astype(np.float64)
    diff = d["traces"].astype(np.float64) if "traces" in d.files else ch_a - ch_b
    overflows = d["overflows"] if "overflows" in d.files else np.zeros(ch_a.shape[0], dtype=np.int16)
    metadata = load_metadata(npz_path)
    dt_ns = float(metadata.get("dt_ns", 1.0))
    t_us = np.arange(ch_a.shape[1], dtype=np.float64) * dt_ns * 1e-3

    n, p = ch_a.shape
    pre = slice(0, min(args.pre_end, p))
    active = slice(max(0, args.active_start), min(args.active_end, p))
    post = slice(max(0, args.post_start), p)

    channels = {
        "channel_a_supply_side": ch_a,
        "channel_b_board_side": ch_b,
        "shunt_diff_a_minus_b": diff,
    }

    report: dict = {
        "npz": str(npz_path),
        "n_traces": int(n),
        "trace_len": int(p),
        "dt_ns": dt_ns,
        "duration_us": float(t_us[-1] + dt_ns * 1e-3),
        "windows": {
            "pre": [0, int(pre.stop)],
            "active": [int(active.start), int(active.stop)],
            "post": [int(post.start), int(post.stop)],
        },
        "overflow_count": int(np.count_nonzero(overflows)),
        "channels": {},
    }

    for name, x in channels.items():
        pre_mean = x[:, pre].mean(axis=1)
        active_mean = x[:, active].mean(axis=1)
        post_mean = x[:, post].mean(axis=1) if post.start < post.stop else np.full(n, np.nan)
        active_delta = active_mean - pre_mean
        post_delta = post_mean - pre_mean
        trace_mean = x.mean(axis=1)
        mean_wave = x.mean(axis=0)

        report["channels"][name] = {
            "full": summarize_block("full", x),
            "pre": summarize_block("pre", x[:, pre]),
            "active": summarize_block("active", x[:, active]),
            "post": summarize_block("post", x[:, post]) if post.start < post.stop else None,
            "active_minus_pre_mean_mv": float(active_delta.mean()),
            "active_minus_pre_std_mv": float(active_delta.std(ddof=1)),
            "active_minus_pre_min_mv": float(active_delta.min()),
            "active_minus_pre_max_mv": float(active_delta.max()),
            "post_minus_pre_mean_mv": float(np.nanmean(post_delta)),
            "trace_mean_slope_mv_per_trace": robust_slope(trace_mean),
            "trace_mean_drift_first_to_last_fit_mv": robust_slope(trace_mean) * (n - 1),
            "mean_wave_min_mv": float(mean_wave.min()),
            "mean_wave_max_mv": float(mean_wave.max()),
            "mean_wave_p2p_mv": float(np.ptp(mean_wave)),
        }

    out_json = Path(f"{args.out_prefix}_summary.json")
    out_json.write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t_us, ch_a.mean(axis=0), label="ChA supply side", lw=1.5)
    ax[0].plot(t_us, ch_b.mean(axis=0), label="ChB board side", lw=1.5)
    ax[0].axvspan(t_us[active.start], t_us[active.stop - 1], color="tab:orange", alpha=0.12, label="active window")
    ax[0].set_ylabel("Voltage (mV)")
    ax[0].set_title("Mean supply voltage during capture")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(t_us, diff.mean(axis=0), color="tab:red", label="ChA - ChB", lw=1.5)
    ax[1].axvspan(t_us[active.start], t_us[active.stop - 1], color="tab:orange", alpha=0.12)
    ax[1].set_xlabel("Time (us)")
    ax[1].set_ylabel("Shunt voltage (mV)")
    ax[1].set_title("Mean differential shunt signal")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    out_wave = Path(f"{args.out_prefix}_waveforms.png")
    fig.savefig(out_wave, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    idx = np.arange(n)
    for axis, (name, x) in zip(ax, channels.items()):
        pre_mean = x[:, pre].mean(axis=1)
        active_mean = x[:, active].mean(axis=1)
        axis.plot(idx, pre_mean, label="pre mean", lw=1.0)
        axis.plot(idx, active_mean, label="active mean", lw=1.0)
        axis.set_ylabel("mV")
        axis.set_title(name)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    ax[-1].set_xlabel("Trace index")
    fig.tight_layout()
    out_evolution = Path(f"{args.out_prefix}_trace_evolution.png")
    fig.savefig(out_evolution, dpi=180)
    plt.close(fig)

    print(f"saved: {out_json}")
    print(f"saved: {out_wave}")
    print(f"saved: {out_evolution}")
    for name, values in report["channels"].items():
        print(
            f"{name}: active-pre={values['active_minus_pre_mean_mv']:.4f} mV "
            f"+/- {values['active_minus_pre_std_mv']:.4f} mV, "
            f"fit drift={values['trace_mean_drift_first_to_last_fit_mv']:.4f} mV"
        )


if __name__ == "__main__":
    main()
