#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pyabf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StimWindow:
    start_s: float
    end_s: float
    mean_pA: float


def detect_stim_window(t, i, baseline_frac=0.1, sigma=5.0, min_abs=5.0):
    if i is None or len(i) == 0:
        return StimWindow(float(t[0]), float(t[-1]), float("nan"))
    n_base = max(1, int(baseline_frac * len(i)))
    baseline = np.median(i[:n_base])
    thr = baseline + sigma * np.std(i[:n_base])
    mask = np.abs(i - baseline) > max(min_abs, thr)
    if np.any(mask):
        idx = np.where(mask)[0]
        start = float(t[idx[0]])
        end = float(t[idx[-1]])
        mean = float(np.mean(i[idx]))
    else:
        start = float(t[0])
        end = float(t[-1])
        mean = float(np.mean(i))
    if end <= start:
        end = start + (t[1] - t[0] if len(t) > 1 else 1e-4)
    return StimWindow(start, end, mean)


def detect_spikes(t, v, dvdt_thresh=20.0, v_thresh=-20.0, refractory_ms=1.0):
    # Simple threshold crossing with dv/dt guard.
    if len(t) < 2:
        return []
    dt = float(t[1] - t[0])
    dv = np.diff(v) / dt
    refractory = int((refractory_ms / 1000.0) / dt)
    events = []
    last_idx = -refractory
    for i in range(1, len(v) - 1):
        if i - last_idx < refractory:
            continue
        if v[i] >= v_thresh and dv[i - 1] >= dvdt_thresh and v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            last_idx = i
            events.append(i)
    return events


def sweep_stats(t, v, i, stim):
    stats = {}
    stats["t_start_s"] = float(t[0])
    stats["t_end_s"] = float(t[-1])
    stats["dt_s"] = float(t[1] - t[0]) if len(t) > 1 else float("nan")
    stats["v_min_mV"] = float(np.nanmin(v))
    stats["v_max_mV"] = float(np.nanmax(v))
    stats["v_mean_mV"] = float(np.nanmean(v))
    if i is not None:
        stats["i_min_pA"] = float(np.nanmin(i))
        stats["i_max_pA"] = float(np.nanmax(i))
        stats["i_mean_pA"] = float(np.nanmean(i))
        stats["stim_start_s"] = stim.start_s
        stats["stim_end_s"] = stim.end_s
        stats["stim_mean_pA"] = stim.mean_pA
    return stats


def save_overview_plot(out_path, traces, ylabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    for t, y in traces:
        ax.plot(t, y, lw=0.6, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(os.path.basename(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ABF: metadata, sweep stats, events, plots.")
    parser.add_argument("abf_path", help="Path to .abf file")
    parser.add_argument("--out-dir", default="abf_analysis", help="Output directory")
    parser.add_argument("--v-channel", type=int, default=0, help="Voltage channel index")
    parser.add_argument("--i-channel", type=int, default=1, help="Current channel index")
    parser.add_argument("--dvdt-thresh", type=float, default=20.0, help="dV/dt threshold (mV/ms)")
    parser.add_argument("--v-thresh", type=float, default=-20.0, help="Voltage threshold (mV)")
    parser.add_argument("--refractory-ms", type=float, default=1.0, help="Refractory window (ms)")
    args = parser.parse_args()

    abf = pyabf.ABF(args.abf_path)
    os.makedirs(args.out_dir, exist_ok=True)

    # Metadata summary
    meta = {
        "abf_id": abf.abfID,
        "sweeps": abf.sweepCount,
        "channels": abf.channelCount,
        "data_rate_hz": abf.dataRate,
        "sweep_length_s": abf.sweepLengthSec,
        "sweep_point_count": abf.sweepPointCount,
        "protocol": abf.protocol,
        "creator": abf.creator,
        "comment": getattr(abf, "abfComment", None) or getattr(abf, "abfFileComment", None),
        "file_size": getattr(abf, "fileSize", None) or getattr(abf, "abfFileSize", None),
        "version": abf.abfVersionString,
        "units": {
            "channel_0": abf.adcUnits[0] if abf.channelCount > 0 else None,
            "channel_1": abf.adcUnits[1] if abf.channelCount > 1 else None,
        },
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "metadata.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    sweep_rows = []
    event_rows = []
    v_traces = []
    i_traces = []

    for sweep in range(abf.sweepCount):
        abf.setSweep(sweepNumber=sweep, channel=args.v_channel)
        t = abf.sweepX.copy()
        v = abf.sweepY.copy()

        i = None
        if args.i_channel is not None and args.i_channel < abf.channelCount:
            abf.setSweep(sweepNumber=sweep, channel=args.i_channel)
            i = abf.sweepY.copy()

        # Ensure increasing time base
        if t[0] > t[-1]:
            t = t[::-1]
            v = v[::-1]
            if i is not None:
                i = i[::-1]

        stim = detect_stim_window(t, i) if i is not None else StimWindow(float(t[0]), float(t[-1]), float("nan"))
        spike_idx = detect_spikes(t, v, dvdt_thresh=args.dvdt_thresh, v_thresh=args.v_thresh,
                                  refractory_ms=args.refractory_ms)

        stats = sweep_stats(t, v, i, stim)
        stats["sweep"] = sweep
        stats["spike_count"] = len(spike_idx)
        sweep_rows.append(stats)

        for idx in spike_idx:
            event_rows.append({
                "sweep": sweep,
                "t_s": float(t[idx]),
                "v_mV": float(v[idx]),
            })

        v_traces.append((t, v))
        if i is not None:
            i_traces.append((t, i))

    # Write sweep stats table
    if sweep_rows:
        with open(os.path.join(args.out_dir, "sweep_stats.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)

    # Write events table
    if event_rows:
        with open(os.path.join(args.out_dir, "events.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
            writer.writeheader()
            writer.writerows(event_rows)

    if v_traces:
        save_overview_plot(os.path.join(args.out_dir, "overview_channel_v.png"), v_traces, "Vm (mV)")
    if i_traces:
        save_overview_plot(os.path.join(args.out_dir, "overview_channel_i.png"), i_traces, "I (pA)")

    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
