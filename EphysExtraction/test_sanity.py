import numpy as np

import ephys_extractor as efex


def make_test_trace():
    rate = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / rate)
    v = np.full_like(t, -70.0)
    spike = 80.0 * np.exp(-((t - 0.2) / 0.001) ** 2)
    v = v + spike
    i = np.zeros_like(t)
    return t, v, i


def make_hyperpol_trace():
    rate = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / rate)
    v = np.full_like(t, -70.0)
    i = np.zeros_like(t)

    start = 0.1
    end = 0.6
    i[(t >= start) & (t <= end)] = -100.0

    tau_fast = 0.02
    tau_slow = 0.2
    A = -5.0
    B = -10.0

    t_rel = t - start
    active = (t >= start) & (t <= end)
    v[active] = -70.0 + A * (1.0 - np.exp(-t_rel[active] / tau_fast)) + B * np.exp(-t_rel[active] / tau_slow)

    return t, v, i, start, end


def main():
    t, v, i = make_test_trace()
    ext = efex.EphysSweepFeatureExtractor(
        t=t,
        v=v,
        i=i,
        start=0.1,
        end=0.9,
        dv_cutoff=20.0,
        filter=2.0,
    )
    ext.process_spikes()

    spikes = ext.spikes()
    print(f"spikes_detected: {len(spikes)}")
    if len(spikes) == 0:
        raise RuntimeError("Expected at least one spike in synthetic trace.")

    spike_keys = ext.spike_feature_keys()
    if len(spike_keys) == 0:
        raise RuntimeError("Expected spike feature keys to be non-empty.")

    for key in spike_keys:
        if key not in spikes[0]:
            raise RuntimeError(f"Missing spike key in output: {key}")

    spike = spikes[0]
    required_keys = [
        "threshold_v",
        "peak_v",
        "width",
        "upstroke",
        "downstroke",
        "upstroke_downstroke_ratio",
    ]
    for key in required_keys:
        if key not in spike:
            raise RuntimeError(f"Missing spike feature: {key}")

    if not (spike["peak_v"] > spike["threshold_v"]):
        raise RuntimeError("Expected peak_v > threshold_v.")
    if not (spike["width"] > 0):
        raise RuntimeError("Expected positive spike width.")
    if not (spike["upstroke_downstroke_ratio"] > 0):
        raise RuntimeError("Expected positive upstroke/downstroke ratio.")

    sweep_keys = list(ext.sweep_feature_keys())
    if len(sweep_keys) == 0:
        raise RuntimeError("Expected sweep feature keys to be non-empty.")
    for key in sweep_keys:
        ext.sweep_feature(key)

    pause_n, pause_frac = ext.pause_metrics()
    burst_idx, burst_n = ext.burst_metrics()
    delay_ratio, delay_tau = ext.delay_metrics()

    if pause_n < 0 or pause_frac < 0:
        raise RuntimeError("Pause metrics out of bounds.")
    if burst_n < 0:
        raise RuntimeError("Burst count out of bounds.")
    if delay_ratio < 0 or delay_tau < 0:
        raise RuntimeError("Delay metrics out of bounds.")

    avg_rate = ext.sweep_feature("avg_rate")
    latency = ext.sweep_feature("latency")
    if not (avg_rate > 0):
        raise RuntimeError("Expected positive avg_rate.")
    if not (0.0 <= latency <= 1.0):
        raise RuntimeError("Latency out of expected bounds.")

    print("avg_rate:", avg_rate)
    print("latency:", latency)

    t, v, i, start, end = make_hyperpol_trace()
    ext = efex.EphysSweepFeatureExtractor(
        t=t,
        v=v,
        i=i,
        start=start,
        end=end,
        dv_cutoff=20.0,
        filter=2.0,
    )

    sweep_keys = list(ext.sweep_feature_keys())
    for key in sweep_keys:
        ext.sweep_feature(key)

    for key in ["v_baseline", "tau", "sag", "peak_deflect", "stim_amp"]:
        try:
            ext.sweep_feature(key)
        except Exception as exc:
            raise RuntimeError(f"Failed to compute sweep feature '{key}': {exc}")

    v_baseline = ext.sweep_feature("v_baseline")
    peak_deflect_v, _ = ext.sweep_feature("peak_deflect")
    sag, sag_ratio = ext.sweep_feature("sag")

    if not (-71.0 <= v_baseline <= -69.0):
        raise RuntimeError(f"Unexpected baseline voltage: {v_baseline}")
    if not (peak_deflect_v < v_baseline):
        raise RuntimeError("Expected hyperpolarizing peak deflection.")
    if not (0.0 <= sag <= 1.0):
        raise RuntimeError(f"Unexpected sag value: {sag}")
    if not (sag_ratio > 0.0):
        raise RuntimeError(f"Unexpected sag ratio: {sag_ratio}")

    print("v_baseline:", v_baseline)
    print("peak_deflect_v:", peak_deflect_v)
    print("sag:", sag)
    print("sag_ratio:", sag_ratio)


if __name__ == "__main__":
    main()
