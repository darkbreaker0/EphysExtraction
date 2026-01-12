import argparse
import csv
import json
import os

import numpy as np
from pynwb import NWBHDF5IO

import ephys_extractor as efex


def parse_description(series):
    try:
        return json.loads(series.description)
    except Exception:
        return {}


def get_sweep_number(series):
    if hasattr(series, "sweep_number") and series.sweep_number is not None:
        return int(series.sweep_number)
    # Match numeric tokens from either underscore or dash separated names.
    parts = series.name.replace("-", "_").split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def series_unit(series):
    try:
        unit = series.unit
    except Exception:
        unit = None
    if not unit:
        return ""
    return unit.lower()


def pick_series(series_list, preferred_name=None, name_key=None):
    if not series_list:
        return None
    if preferred_name:
        for s in series_list:
            meta = parse_description(s)
            if meta.get(name_key) == preferred_name:
                return s
    if name_key == "adc_name":
        for s in series_list:
            meta = parse_description(s)
            name = (meta.get("adc_name") or "").lower()
            if "vm" in name or "voltage" in name:
                return s
        # Fallback: pick voltage units.
        for s in series_list:
            if series_unit(s) in ("volts", "v"):
                return s
    if name_key == "dac_name":
        for s in series_list:
            meta = parse_description(s)
            name = (meta.get("dac_name") or "").lower()
            if "i" in name or "clamp" in name:
                return s
        # Fallback: pick current units.
        for s in series_list:
            if series_unit(s) in ("amperes", "a"):
                return s
    return series_list[0]


def series_to_trace(series):
    data = np.asarray(series.data[:], dtype=np.float64)
    conv = float(getattr(series, "conversion", 1.0))
    unit = (series.unit or "").lower()
    data_si = data * conv
    return data_si, unit


def to_millivolts(data_si, unit):
    if unit in ("volts", "v"):
        return data_si * 1e3
    return data_si


def to_picoamps(data_si, unit):
    if unit in ("amperes", "a"):
        return data_si * 1e12
    return data_si


def build_time_vector(series, n_points):
    rate = float(series.rate)
    t0 = float(series.starting_time)
    return t0 + (np.arange(n_points) / rate)


def average_window(v, t, start, end):
    start = max(start, t[0])
    end = min(end, t[-1])
    if end <= start:
        return np.nan
    start_idx = int(np.searchsorted(t, start, side="left"))
    end_idx = int(np.searchsorted(t, end, side="right"))
    if end_idx <= start_idx:
        return np.nan
    return float(np.mean(v[start_idx:end_idx]))


def find_stim_window(t, i_pA, stim_eps):
    if i_pA is None:
        return t[0], t[-1]
    active = np.flatnonzero(np.abs(i_pA) > stim_eps)
    if active.size == 0:
        return t[0], t[-1]
    return t[active[0]], t[active[-1]]


def safe_feature(ext, key):
    try:
        return ext.sweep_feature(key)
    except Exception:
        return np.nan


def estimate_stim_amp(t, i_pA, start):
    if i_pA is None:
        return np.nan
    idx = int(np.searchsorted(t, start, side="left"))
    idx = min(idx + 1, len(i_pA) - 1)
    return float(i_pA[idx])


def linear_slope(x, y):
    if len(x) < 2:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def choose_window(t, i_pA, stim_eps, fixed_start, fixed_end, min_stim_dur):
    if fixed_start is not None:
        start = fixed_start
        end = fixed_end if fixed_end is not None else t[-1]
        return start, end

    start, end = find_stim_window(t, i_pA, stim_eps)
    if (end - start) >= min_stim_dur:
        return start, end

    span = t[-1] - t[0]
    fallback_start = t[0] + 0.2 * span
    fallback_end = t[0] + 0.8 * span
    return fallback_start, fallback_end


def classify_sweep(i_pA, t, start, end, step_tol, short_thresh):
    if i_pA is None:
        return "unknown"
    start_idx = int(np.searchsorted(t, start, side="left"))
    end_idx = int(np.searchsorted(t, end, side="right"))
    if end_idx <= start_idx + 1:
        return "unknown"
    window = i_pA[start_idx:end_idx]
    if np.nanmax(window) - np.nanmin(window) <= step_tol:
        dur = end - start
        return "short_square" if dur <= short_thresh else "long_square"
    return "ramp"


def most_common_amp(amps):
    if len(amps) == 0:
        return np.nan
    vals, counts = np.unique(np.round(amps, 6), return_counts=True)
    max_count = counts.max()
    candidates = vals[counts == max_count]
    return float(np.min(candidates))


def estimate_current_noise(i_pA):
    # Use trace edges as baseline to estimate command noise.
    if i_pA is None or len(i_pA) < 10:
        return np.nan
    n = len(i_pA)
    edge = max(1, int(0.1 * n))
    baseline = np.concatenate([i_pA[:edge], i_pA[-edge:]])
    return float(np.nanstd(baseline))


def estimate_dvdt_cutoff(v_mV, t):
    # Estimate a dv/dt cutoff from high-percentile positive slopes.
    if v_mV is None or len(v_mV) < 10:
        return np.nan
    dvdt = np.diff(v_mV) / np.diff(t)
    dvdt = dvdt[np.isfinite(dvdt)]
    dvdt = dvdt[dvdt > 0]
    if dvdt.size == 0:
        return np.nan
    p995 = float(np.percentile(dvdt, 99.5))
    return max(5.0, min(40.0, 0.5 * p995))


def auto_parameters(nwbfile, acq_by_sweep, stim_by_sweep, adc_name, dac_name,
                    defaults):
    # Derive per-file heuristics from a few representative sweeps.
    rates = []
    noise_vals = []
    starts = []
    ends = []
    durations = []
    dvdt_cutoffs = []

    for sweep in sorted(acq_by_sweep.keys()):
        acq_series = pick_series(acq_by_sweep.get(sweep, []), adc_name, "adc_name")
        if acq_series is None:
            continue
        rates.append(float(acq_series.rate))

        stim_series = pick_series(stim_by_sweep.get(sweep, []), dac_name, "dac_name")
        i = None
        if stim_series is not None:
            i_si, i_unit = series_to_trace(stim_series)
            i = to_picoamps(i_si, i_unit)

        if i is not None:
            noise_vals.append(estimate_current_noise(i))

        v_si, v_unit = series_to_trace(acq_series)
        v = to_millivolts(v_si, v_unit)
        t = build_time_vector(acq_series, len(v))

        dvdt_cutoff = estimate_dvdt_cutoff(v, t)
        if np.isfinite(dvdt_cutoff):
            dvdt_cutoffs.append(dvdt_cutoff)

        if i is not None:
            stim_eps = max(1.0, 4.0 * np.nanmedian(noise_vals)) if noise_vals else defaults["stim_eps"]
            start, end = find_stim_window(t, i, stim_eps)
            if np.isfinite(start) and np.isfinite(end) and end > start:
                starts.append(start)
                ends.append(end)
                durations.append(end - start)

        if len(rates) >= 3:
            break

    rate = float(np.nanmedian(rates)) if rates else np.nan
    nyquist_khz = (rate / 2.0) / 1000.0 if np.isfinite(rate) else np.nan
    filter_khz = defaults["filter_khz"]
    if np.isfinite(nyquist_khz):
        filter_khz = min(4.0, 0.45 * nyquist_khz)
        filter_khz = max(1.0, filter_khz)

    stim_eps = defaults["stim_eps"]
    if noise_vals and np.isfinite(np.nanmedian(noise_vals)):
        stim_eps = max(1.0, 4.0 * float(np.nanmedian(noise_vals)))

    step_tol = defaults["step_tol"]
    if noise_vals and np.isfinite(np.nanmedian(noise_vals)):
        step_tol = max(1.0, 3.0 * float(np.nanmedian(noise_vals)))

    min_stim_dur = defaults["min_stim_dur"]
    short_thresh = defaults["short_thresh"]
    baseline_window = defaults["baseline_window"]
    fixed_start = defaults["fixed_start"]
    fixed_end = defaults["fixed_end"]
    if durations:
        med_dur = float(np.nanmedian(durations))
        min_stim_dur = max(0.01, 0.2 * med_dur)
        baseline_window = min(0.1, 0.25 * med_dur)
        baseline_window = max(0.02, baseline_window)
        if med_dur >= 0.2:
            short_thresh = 0.1
        else:
            short_thresh = max(0.02, 0.25 * med_dur)

        dt = 1.0 / rate if np.isfinite(rate) and rate > 0 else np.nan
        if np.isfinite(dt) and np.std(starts) <= 2 * dt and np.std(ends) <= 2 * dt:
            fixed_start = float(np.nanmedian(starts))
            fixed_end = float(np.nanmedian(ends))

    dv_cutoff = defaults["dv_cutoff"]
    if dvdt_cutoffs:
        dv_cutoff = float(np.nanmedian(dvdt_cutoffs))

    return dict(
        filter_khz=filter_khz,
        dv_cutoff=dv_cutoff,
        stim_eps=stim_eps,
        min_stim_dur=min_stim_dur,
        fixed_start=fixed_start,
        fixed_end=fixed_end,
        step_tol=step_tol,
        short_thresh=short_thresh,
        baseline_window=baseline_window,
    )


def run(nwb_path, adc_name, dac_name, filter_khz, dv_cutoff, stim_eps,
        min_stim_dur, fixed_start, fixed_end, step_tol, short_thresh,
        baseline_window, out_prefix, auto_params):
    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwbfile = io.read()

        acq_by_sweep = {}
        for series in nwbfile.acquisition.values():
            sweep = get_sweep_number(series)
            if sweep is None:
                continue
            acq_by_sweep.setdefault(sweep, []).append(series)

        stim_by_sweep = {}
        for series in nwbfile.stimulus.values():
            sweep = get_sweep_number(series)
            if sweep is None:
                continue
            stim_by_sweep.setdefault(sweep, []).append(series)

        if auto_params:
            defaults = dict(
                filter_khz=filter_khz,
                dv_cutoff=dv_cutoff,
                stim_eps=stim_eps,
                min_stim_dur=min_stim_dur,
                fixed_start=fixed_start,
                fixed_end=fixed_end,
                step_tol=step_tol,
                short_thresh=short_thresh,
                baseline_window=baseline_window,
            )
            auto = auto_parameters(
                nwbfile, acq_by_sweep, stim_by_sweep, adc_name, dac_name, defaults
            )
            filter_khz = auto["filter_khz"]
            dv_cutoff = auto["dv_cutoff"]
            stim_eps = auto["stim_eps"]
            min_stim_dur = auto["min_stim_dur"]
            fixed_start = auto["fixed_start"]
            fixed_end = auto["fixed_end"]
            step_tol = auto["step_tol"]
            short_thresh = auto["short_thresh"]
            baseline_window = auto["baseline_window"]
            print("Auto params:", auto)

        sweep_rows = []
        spike_rows = []

        for sweep in sorted(acq_by_sweep.keys()):
            acq_series = pick_series(acq_by_sweep.get(sweep, []), adc_name, "adc_name")
            if acq_series is None:
                continue
            stim_series = pick_series(stim_by_sweep.get(sweep, []), dac_name, "dac_name")

            acq_meta = parse_description(acq_series)
            stim_meta = parse_description(stim_series) if stim_series is not None else {}

            v_si, v_unit = series_to_trace(acq_series)
            v = to_millivolts(v_si, v_unit)
            t = build_time_vector(acq_series, len(v))

            i = None
            if stim_series is not None:
                i_si, i_unit = series_to_trace(stim_series)
                i = to_picoamps(i_si, i_unit)

            start, end = choose_window(t, i, stim_eps, fixed_start, fixed_end, min_stim_dur)
            sweep_type = classify_sweep(i, t, start, end, step_tol, short_thresh)

            ext = efex.EphysSweepFeatureExtractor(
                t=t,
                v=v,
                i=i,
                start=start,
                end=end,
                filter=filter_khz,
                dv_cutoff=dv_cutoff,
            )
            ext.process_spikes()

            spike_count = len(ext.spikes())
            if spike_count > 0:
                try:
                    pause_n, pause_frac = ext.pause_metrics()
                except Exception:
                    pause_n, pause_frac = 0, 0.0
                try:
                    burst_index, burst_count = ext.burst_metrics()
                except Exception:
                    burst_index, burst_count = 0.0, 0
                try:
                    delay_ratio, delay_tau = ext.delay_metrics()
                except Exception:
                    delay_ratio, delay_tau = np.nan, np.nan
            else:
                pause_n, pause_frac = 0, 0.0
                burst_index, burst_count = 0.0, 0
                delay_ratio, delay_tau = 0.0, 0.0

            stim_amp = estimate_stim_amp(t, i, start)
            peak_deflect = safe_feature(ext, "peak_deflect")
            if isinstance(peak_deflect, (list, tuple)) and len(peak_deflect) == 2:
                peak_deflect_v, peak_deflect_idx = peak_deflect
                peak_deflect_i = i[int(peak_deflect_idx)] if i is not None else np.nan
            else:
                peak_deflect_v = np.nan
                peak_deflect_i = np.nan

            v_baseline = safe_feature(ext, "v_baseline")
            v_steady = average_window(v, t, end - baseline_window, end)
            if np.isnan(v_steady) or np.isnan(v_baseline):
                deflect_ss = np.nan
            else:
                deflect_ss = v_steady - v_baseline

            sag_raw = safe_feature(ext, "sag")
            if isinstance(sag_raw, (list, tuple)) and len(sag_raw) == 2:
                sag_val = sag_raw[0]
                sag_ratio = sag_raw[1]
            else:
                sag_val = sag_raw
                sag_ratio = np.nan

            row = {
                "sweep_number": sweep,
                "adc_name": acq_meta.get("adc_name"),
                "dac_name": stim_meta.get("dac_name"),
                "start_s": start,
                "end_s": end,
                "sweep_type": sweep_type,
                "stim_amp": stim_amp,
                "peak_deflect_v": peak_deflect_v,
                "peak_deflect_i": peak_deflect_i,
                "v_steady": v_steady,
                "deflect_ss": deflect_ss,
                "spike_count": spike_count,
                "pause_n": pause_n,
                "pause_frac": pause_frac,
                "burst_index": burst_index,
                "burst_count": burst_count,
                "delay_ratio": delay_ratio,
                "delay_tau": delay_tau,
                "avg_rate": safe_feature(ext, "avg_rate"),
                "latency": safe_feature(ext, "latency"),
                "mean_isi": safe_feature(ext, "mean_isi"),
                "median_isi": safe_feature(ext, "median_isi"),
                "first_isi": safe_feature(ext, "first_isi"),
                "isi_cv": safe_feature(ext, "isi_cv"),
                "adaptation_index": safe_feature(ext, "adaptation_index"),
                "AP_amp_adapt": safe_feature(ext, "AP_amp_adapt"),
                "AP_amp_change": safe_feature(ext, "AP_amp_change"),
                "AP_fano_factor": safe_feature(ext, "AP_fano_factor"),
                "AP_cv": safe_feature(ext, "AP_cv"),
                "isis_change": safe_feature(ext, "isis_change"),
                "norm_sq_isis": safe_feature(ext, "norm_sq_isis"),
                "fano_factor": safe_feature(ext, "fano_factor"),
                "cv": safe_feature(ext, "cv"),
                "v_baseline": v_baseline,
                "tau": safe_feature(ext, "tau"),
                "sag": sag_raw,
                "sag_value": sag_val,
                "sag_ratio": sag_ratio,
            }
            sweep_rows.append(row)

            for spike in ext.spikes():
                spike_row = {"sweep_number": sweep}
                spike_row.update(spike)
                spike_rows.append(spike_row)

    if not out_prefix:
        out_prefix = os.path.splitext(nwb_path)[0]

    sweep_path = out_prefix + "_sweep_features.csv"
    spike_path = out_prefix + "_spike_features.csv"

    if sweep_rows:
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)

    cell_path = out_prefix + "_cell_features.csv"
    if sweep_rows:
        avg_rate_vals = np.array([r["avg_rate"] for r in sweep_rows], dtype=float)
        stim_amps = np.array([r["stim_amp"] for r in sweep_rows], dtype=float)
        v_bases = np.array([r["v_baseline"] for r in sweep_rows], dtype=float)
        v_peaks = np.array([r["peak_deflect_v"] for r in sweep_rows], dtype=float)
        i_peaks = np.array([r["peak_deflect_i"] for r in sweep_rows], dtype=float)
        sweep_types = np.array([r["sweep_type"] for r in sweep_rows], dtype=object)

        em_mask = ~np.isnan(v_bases)
        em_mV = float(np.nanmean(v_bases[em_mask])) if np.any(em_mask) else np.nan

        spiking_mask = (avg_rate_vals > 0) & ~np.isnan(stim_amps)
        if np.any(spiking_mask):
            i_rheo_pA = float(np.min(stim_amps[spiking_mask]))
        else:
            i_rheo_pA = np.nan

        sub_mask = (avg_rate_vals == 0) & (stim_amps < 0)
        if np.sum(sub_mask) >= 2:
            v = v_peaks[sub_mask]
            i = stim_amps[sub_mask]
            r_rest_mohm = linear_slope(i, v) * 1e3
        else:
            r_rest_mohm = np.nan

        long_mask = sweep_types == "long_square"
        short_mask = sweep_types == "short_square"
        ramp_mask = sweep_types == "ramp"

        long_spiking = long_mask & (avg_rate_vals > 0) & ~np.isnan(stim_amps)
        long_sub = long_mask & (avg_rate_vals == 0) & (stim_amps < 0)

        if np.any(long_spiking):
            i_rheo_long = float(np.min(stim_amps[long_spiking]))
        else:
            i_rheo_long = np.nan

        if np.sum(long_spiking) >= 2:
            x = stim_amps[long_spiking]
            y = avg_rate_vals[long_spiking]
            A = np.vstack([x, np.ones_like(x)]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            fi_slope = float(m)
        else:
            fi_slope = np.nan

        if np.any(long_mask):
            v_baseline_long = float(np.nanmean(v_bases[long_mask]))
        else:
            v_baseline_long = np.nan

        if np.sum(long_sub) >= 1:
            sag_vals = []
            sag_vm = []
            sag_ratios = []
            for r in [row for row, keep in zip(sweep_rows, long_sub) if keep]:
                sag = r.get("sag_value")
                sag_r = r.get("sag_ratio")
                peak_v = r.get("peak_deflect_v")
                sag_vals.append(sag if isinstance(sag, (int, float)) else np.nan)
                sag_ratios.append(sag_r if isinstance(sag_r, (int, float)) else np.nan)
                sag_vm.append(peak_v)
            sag_vals = np.array(sag_vals, dtype=float)
            sag_ratios = np.array(sag_ratios, dtype=float)
            sag_vm = np.array(sag_vm, dtype=float)
            target = -100.0
            if np.any(~np.isnan(sag_vm)):
                idx = int(np.nanargmin(np.abs(sag_vm - target)))
                sag_at_target = float(sag_vals[idx]) if idx < len(sag_vals) else np.nan
                vm_for_sag = float(sag_vm[idx]) if idx < len(sag_vm) else np.nan
                sag_ratio_at_target = float(sag_ratios[idx]) if idx < len(sag_ratios) else np.nan
            else:
                sag_at_target = np.nan
                vm_for_sag = np.nan
                sag_ratio_at_target = np.nan
        else:
            sag_at_target = np.nan
            vm_for_sag = np.nan
            sag_ratio_at_target = np.nan

        tau_vals_all = np.array([r["tau"] for r in sweep_rows], dtype=float)
        tau_mask = long_sub & ~np.isnan(tau_vals_all)
        if np.any(tau_mask):
            tau_mean = float(np.nanmean(tau_vals_all[tau_mask]))
        else:
            tau_mean = np.nan

        short_spiking = short_mask & (avg_rate_vals > 0) & ~np.isnan(stim_amps)
        short_common_amp = most_common_amp(stim_amps[short_spiking])

        ramp_spiking = ramp_mask & (avg_rate_vals > 0)

        deflect_peak = v_peaks - v_bases
        v_steady_vals = np.array([r["v_steady"] for r in sweep_rows], dtype=float)
        deflect_ss = np.array([r["deflect_ss"] for r in sweep_rows], dtype=float)

        peak_mask = sub_mask & ~np.isnan(deflect_peak)
        ss_mask = sub_mask & ~np.isnan(v_steady_vals)
        bs_mask = sub_mask & ~np.isnan(deflect_ss)

        r_rest_peak_mohm = linear_slope(stim_amps[peak_mask], deflect_peak[peak_mask]) * 1e3
        r_rest_ss_mohm = linear_slope(stim_amps[ss_mask], v_steady_vals[ss_mask]) * 1e3
        r_rest_bs_mohm = linear_slope(stim_amps[bs_mask], deflect_ss[bs_mask]) * 1e3

        depol_mask = (avg_rate_vals == 0) & (stim_amps > 0)
        depol_peak_mask = depol_mask & ~np.isnan(deflect_peak)
        depol_ss_mask = depol_mask & ~np.isnan(v_steady_vals)
        depol_bs_mask = depol_mask & ~np.isnan(deflect_ss)

        depol_resp_peak = float(np.nanmean(deflect_peak[depol_peak_mask])) if np.any(depol_peak_mask) else np.nan
        depol_resp_ss = float(np.nanmean(deflect_ss[depol_bs_mask])) if np.any(depol_bs_mask) else np.nan

        r_depol_peak_mohm = linear_slope(stim_amps[depol_peak_mask], deflect_peak[depol_peak_mask]) * 1e3
        r_depol_ss_mohm = linear_slope(stim_amps[depol_ss_mask], v_steady_vals[depol_ss_mask]) * 1e3
        r_depol_bs_mohm = linear_slope(stim_amps[depol_bs_mask], deflect_ss[depol_bs_mask]) * 1e3

        rect_peak = r_depol_peak_mohm / r_rest_peak_mohm if np.isfinite(r_depol_peak_mohm) and np.isfinite(r_rest_peak_mohm) and r_rest_peak_mohm != 0 else np.nan
        rect_ss = r_depol_ss_mohm / r_rest_ss_mohm if np.isfinite(r_depol_ss_mohm) and np.isfinite(r_rest_ss_mohm) and r_rest_ss_mohm != 0 else np.nan
        rect_bs = r_depol_bs_mohm / r_rest_bs_mohm if np.isfinite(r_depol_bs_mohm) and np.isfinite(r_rest_bs_mohm) and r_rest_bs_mohm != 0 else np.nan

        cell_row = {
            "Em_mV": em_mV,
            "I_rheo_pA": i_rheo_pA,
            "R_rest_MOhm": r_rest_mohm,
            "R_rest_peak_MOhm": r_rest_peak_mohm,
            "R_rest_ss_MOhm": r_rest_ss_mohm,
            "R_rest_bs_MOhm": r_rest_bs_mohm,
            "depol_resp_peak_mV": depol_resp_peak,
            "depol_resp_ss_mV": depol_resp_ss,
            "R_depol_peak_MOhm": r_depol_peak_mohm,
            "R_depol_ss_MOhm": r_depol_ss_mohm,
            "R_depol_bs_MOhm": r_depol_bs_mohm,
            "rectification_peak": rect_peak,
            "rectification_ss": rect_ss,
            "rectification_bs": rect_bs,
            "I_rheo_long_pA": i_rheo_long,
            "fi_slope": fi_slope,
            "v_baseline_long_mV": v_baseline_long,
            "tau_long_s": tau_mean,
            "sag_long": sag_at_target,
            "sag_ratio_long": sag_ratio_at_target,
            "IH_sag": sag_at_target,
            "IH_sag_ratio": sag_ratio_at_target,
            "vm_for_sag_mV": vm_for_sag,
            "short_common_amp_pA": short_common_amp,
            "n_sweeps": len(sweep_rows),
            "n_spiking_sweeps": int(np.sum(spiking_mask)),
            "n_long_sweeps": int(np.sum(long_mask)),
            "n_short_sweeps": int(np.sum(short_mask)),
            "n_ramp_sweeps": int(np.sum(ramp_mask)),
            "n_long_spiking": int(np.sum(long_spiking)),
            "n_long_subthreshold": int(np.sum(long_sub)),
            "n_short_spiking": int(np.sum(short_spiking)),
            "n_ramp_spiking": int(np.sum(ramp_spiking)),
        }

        # Aggregate numeric spike-level features
        if spike_rows:
            spike_keys = [k for k in spike_rows[0].keys() if k != "sweep_number"]
            for key in spike_keys:
                vals = []
                for row in spike_rows:
                    val = row.get(key)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        vals.append(val)
                if len(vals) == 0:
                    mean_val = np.nan
                    median_val = np.nan
                else:
                    arr = np.array(vals, dtype=float)
                    mean_val = float(np.nanmean(arr))
                    median_val = float(np.nanmedian(arr))
                cell_row[f"spike_mean_{key}"] = mean_val
                cell_row[f"spike_median_{key}"] = median_val

        # Aggregate numeric sweep-level features
        sweep_num_keys = [k for k in sweep_rows[0].keys()
                          if k not in ("sweep_number", "adc_name", "dac_name", "sweep_type")]
        for key in sweep_num_keys:
            vals = []
            for row in sweep_rows:
                val = row.get(key)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    vals.append(val)
            if len(vals) == 0:
                mean_val = np.nan
                median_val = np.nan
            else:
                arr = np.array(vals, dtype=float)
                mean_val = float(np.nanmean(arr))
                median_val = float(np.nanmedian(arr))
            cell_row[f"sweep_mean_{key}"] = mean_val
            cell_row[f"sweep_median_{key}"] = median_val

        with open(cell_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cell_row.keys()))
            writer.writeheader()
            writer.writerow(cell_row)

    if spike_rows:
        fieldnames = list(spike_rows[0].keys())
        # Ensure fieldnames cover any late-appearing spike keys.
        for row in spike_rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(spike_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(spike_rows)

    print(f"Wrote {sweep_path}")
    if sweep_rows:
        print(f"Wrote {cell_path}")
    if spike_rows:
        print(f"Wrote {spike_path}")
    else:
        print("No spikes detected; spike CSV not written.")


def main():
    parser = argparse.ArgumentParser(description="Extract sweep/spike features from NWB.")
    parser.add_argument("nwb_path", help="Path to NWB file.")
    parser.add_argument("--adc-name", default=None, help="ADC channel name to extract.")
    parser.add_argument("--dac-name", default=None, help="DAC channel name to extract.")
    parser.add_argument("--filter-khz", type=float, default=2.0, help="Bessel low-pass cutoff in kHz.")
    parser.add_argument("--dv-cutoff", type=float, default=20.0, help="dV/dt cutoff for spike detection.")
    parser.add_argument("--stim-eps", type=float, default=1.0, help="Stimulus threshold in pA.")
    parser.add_argument("--min-stim-dur", type=float, default=0.01, help="Minimum stimulus duration in seconds.")
    parser.add_argument("--fixed-start", type=float, default=None, help="Fixed start time in seconds.")
    parser.add_argument("--fixed-end", type=float, default=None, help="Fixed end time in seconds.")
    parser.add_argument("--step-tol", type=float, default=5.0, help="Max pA range to treat as step.")
    parser.add_argument("--short-thresh", type=float, default=0.02, help="Max duration for short square (sec).")
    parser.add_argument("--baseline-window", type=float, default=0.1, help="Window for steady-state voltage (sec).")
    parser.add_argument("--out-prefix", default=None, help="Output prefix for CSVs.")
    parser.add_argument("--auto-params", action="store_true",
                        help="Estimate analysis parameters from the NWB file.")
    args = parser.parse_args()

    run(
        nwb_path=args.nwb_path,
        adc_name=args.adc_name,
        dac_name=args.dac_name,
        filter_khz=args.filter_khz,
        dv_cutoff=args.dv_cutoff,
        stim_eps=args.stim_eps,
        min_stim_dur=args.min_stim_dur,
        fixed_start=args.fixed_start,
        fixed_end=args.fixed_end,
        step_tol=args.step_tol,
        short_thresh=args.short_thresh,
        baseline_window=args.baseline_window,
        out_prefix=args.out_prefix,
        auto_params=args.auto_params,
    )


if __name__ == "__main__":
    main()
