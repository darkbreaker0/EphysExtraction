import json
from pathlib import Path

import numpy as np
import pyabf

from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
import ipfx.spike_train_features as stf
import ipfx.subthresh_features as sub


def detect_stim_window(t, i):
    baseline_samples = int(0.1 * len(i))
    baseline = np.median(i[:baseline_samples])
    thr = baseline + 5.0 * np.std(i[:baseline_samples])
    mask = np.abs(i - baseline) > max(5.0, thr)
    if np.any(mask):
        idx = np.where(mask)[0]
        stim_start = float(t[idx[0]])
        stim_end = float(t[idx[-1]])
        stim_mean = float(np.mean(i[idx]))
    else:
        stim_start = float(t[0])
        stim_end = float(t[-1])
        stim_mean = float(np.mean(i))
    if stim_end <= stim_start:
        stim_end = stim_start + (t[1] - t[0] if len(t) > 1 else 1e-4)
    return stim_start, stim_end, stim_mean


def summarize_spikes(spikes_df):
    summary = {}
    if spikes_df is None or len(spikes_df) == 0:
        return summary
    for col in spikes_df.columns:
        values = spikes_df[col]
        if values.dtype.kind not in "biufc":
            continue
        arr = values.to_numpy(dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        summary[f"{col}_mean"] = float(np.nanmean(arr))
        summary[f"{col}_std"] = float(np.nanstd(arr))
        summary[f"{col}_min"] = float(np.nanmin(arr))
        summary[f"{col}_max"] = float(np.nanmax(arr))
    return summary


def main():
    abf_path = Path("2020_06_25_0007.abf")
    abf = pyabf.ABF(str(abf_path))

    results = []
    level_groups = {}

    for sweep in range(abf.sweepCount):
        abf.setSweep(sweepNumber=sweep, channel=0)
        t = abf.sweepX.copy()  # seconds
        v = abf.sweepY  # mV

        abf.setSweep(sweepNumber=sweep, channel=1)
        i = abf.sweepY  # pA

        # Ensure time base is increasing and starts at 0 for IPFX.
        if t[0] > t[-1]:
            t = t[::-1]
            v = v[::-1]
            i = i[::-1]
        # Rebuild time base to avoid any unexpected non-monotonic values.
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / abf.dataRate
        t = np.arange(len(t)) * dt

        stim_start, stim_end, stim_mean = detect_stim_window(t, i)
        t_min = float(np.min(t))
        t_max = float(np.max(t))
        if stim_start < t_min or stim_start > t_max:
            stim_start = t_min
        if stim_end < t_min or stim_end > t_max:
            stim_end = t_max
        if stim_end <= stim_start:
            stim_end = t_max

        spike_extractor = SpikeFeatureExtractor(start=stim_start, end=stim_end, filter=5.0)
        spikes_df = spike_extractor.process(t, v, i)

        spike_train_extractor = SpikeTrainFeatureExtractor(start=stim_start, end=stim_end, filter_frequency=1.0)
        spike_train_features = spike_train_extractor.process(t, v, i, spikes_df)

        # Expand spike train features with extra IPFX utilities.
        extra_stf = stf.basic_spike_train_features(t, spikes_df, stim_start, stim_end)
        if len(spikes_df) > 0:
            extra_stf["delay"] = stf.delay(t, v, spikes_df, stim_start, stim_end)
            extra_stf.update(stf.burst(t, spikes_df))
            extra_stf.update(stf.pause(t, spikes_df))

        spike_train_features = {
            k: (float(v) if v is not None else None)
            for k, v in {**spike_train_features, **extra_stf}.items()
        }

        sub_features = {}
        try:
            sub_features["baseline_voltage"] = float(sub.baseline_voltage(t, v, stim_start))
        except Exception:
            sub_features["baseline_voltage"] = None
        try:
            sub_features["voltage_deflection"] = float(sub.voltage_deflection(t, v, i, stim_start, stim_end))
        except Exception:
            sub_features["voltage_deflection"] = None
        try:
            sub_features["sag"] = float(sub.sag(t, v, i, stim_start, stim_end))
        except Exception:
            sub_features["sag"] = None
        try:
            sub_features["time_constant"] = float(sub.time_constant(t, v, i, stim_start, stim_end))
        except Exception:
            sub_features["time_constant"] = None
        try:
            sub_features["input_resistance"] = float(
                sub.input_resistance([t], [i], [v], stim_start, stim_end)
            )
        except Exception:
            sub_features["input_resistance"] = None

        spike_summary = summarize_spikes(spikes_df)

        entry = {
            "sweep": sweep,
            "stim_start_s": stim_start,
            "stim_end_s": stim_end,
            "stim_mean_pA": stim_mean,
            "spike_train_features": spike_train_features,
            "spike_features": spikes_df.to_dict(orient="records"),
            "spike_feature_summary": spike_summary,
            "subthreshold_features": sub_features,
        }
        results.append(entry)

        level = int(round(stim_mean))
        level_groups.setdefault(level, []).append(entry)

    # Summarize by level
    level_summary = {}
    for level, entries in level_groups.items():
        summary = {"n_sweeps": len(entries)}

        def summarize_keys(entries_list, key_path):
            keys = set()
            for e in entries_list:
                keys.update(e[key_path].keys())
            out = {}
            for key in keys:
                vals = [e[key_path].get(key) for e in entries_list]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if vals:
                    out[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }
            return out

        summary["spike_train_features"] = summarize_keys(entries, "spike_train_features")
        summary["subthreshold_features"] = summarize_keys(entries, "subthreshold_features")
        summary["spike_feature_summary"] = summarize_keys(entries, "spike_feature_summary")
        level_summary[level] = summary

    out = {
        "file": str(abf_path),
        "per_sweep": results,
        "per_level_summary": level_summary,
    }

    Path("ipfx_features.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # CSV summary: spike train, subthreshold, and spike summary means.
    spike_train_keys = set()
    sub_keys = set()
    spike_summary_keys = set()
    for e in results:
        spike_train_keys.update(e["spike_train_features"].keys())
        sub_keys.update(e["subthreshold_features"].keys())
        spike_summary_keys.update(e["spike_feature_summary"].keys())

    spike_train_keys = sorted(spike_train_keys)
    sub_keys = sorted(sub_keys)
    spike_summary_keys = sorted(spike_summary_keys)

    lines = []
    header = (
        ["sweep", "stim_mean_pA"]
        + [f"spike_train_{k}" for k in spike_train_keys]
        + [f"sub_{k}" for k in sub_keys]
        + [f"spike_summary_{k}" for k in spike_summary_keys]
    )
    lines.append(",".join(header))
    for e in results:
        row = [str(e["sweep"]), f"{e['stim_mean_pA']:.3f}"]
        for k in spike_train_keys:
            v = e["spike_train_features"].get(k)
            row.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}")
        for k in sub_keys:
            v = e["subthreshold_features"].get(k)
            row.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}")
        for k in spike_summary_keys:
            v = e["spike_feature_summary"].get(k)
            row.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}")
        lines.append(",".join(row))

    Path("ipfx_features.csv").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote ipfx_features.json and ipfx_features.csv")


if __name__ == "__main__":
    main()
