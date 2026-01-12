# User Manual

This manual describes the typical workflows used in this workspace.

Workflow for ABF data
1) Summarize the ABF file.
2) Convert ABF to NWB (choose one approach).
3) Run feature extraction on the NWB file.

1) Report summary of an ABF file (abf_analyze.py)
Run the analyzer:

```powershell
python abf_analyze.py
```

Outputs (written by the script; see the paths configured inside it):
- ABF summary report (basic acquisition metadata and stimulus timing).
- Sweep table CSV (Allen-style sweep metadata if enabled).

2) Convert ABF to NWB (choose one)

Option A: abf2_to_nwb2.py

```powershell
python abf2_to_nwb2.py
```

Notes:
- Reads the ABF file path set in the script.
- Writes an NWB file alongside the ABF (or at the configured output path).

Option B: neuroconv
- Use the NeuroConv ABF interface and pass the ABF path plus optional metadata.
- This produces an NWB file that uses icephys tables.

3) Run feature extraction on an NWB file (run_features_on_nwb.py)
1) (Recommended) Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

2) Run the extractor:

```powershell
python run_features_on_nwb.py <path_to_nwb> ^
  --fixed-start 0.28125 --fixed-end 0.7812 ^
  --stim-eps 1.0 --min-stim-dur 0.1 ^
  --step-tol 1.0 --short-thresh 0.1 ^
  --filter-khz 2.0 --dv-cutoff 15.0 --baseline-window 0.1 ^
  --out-prefix <output_prefix>
```

Outputs:
- `<output_prefix>_sweep_features.csv`: per-sweep features (stimulus, baseline,
  deflections, spike counts, sweep classification).
- `<output_prefix>_cell_features.csv`: aggregated cell-level features (Em,
  rheobase, input resistance, sag/ih metrics).
- `<output_prefix>_spike_features.csv`: per-spike features (threshold, peak,
  widths). Written only if spikes are detected.

Notes on sweep metadata
- `run_features_on_nwb.py` expects sweep numbers and stimulus pairing to be
available from the NWB file. AIBS-style sweeps include `sweep_number` and
channel metadata in the series description.
- Neuroconv-generated NWB files use icephys tables; if spike detection fails,
the script may need to be adapted to the icephys tables.

Common troubleshooting
- No spikes detected: lower `--dv-cutoff`, increase `--filter-khz`, or remove
`--fixed-start/--fixed-end` to let the script auto-detect the stimulus window.
- Warnings about exponential fits: `tau` estimates may be unreliable; other
features are usually unaffected.
