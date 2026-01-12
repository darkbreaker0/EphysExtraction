# Ephys_extraction Workspace

This workspace packages the upstream `EphysExtraction/` codebase together
with local analysis scripts, converted data, and outputs used during
validation.

Upstream source:
- https://github.com/berenslab/EphysExtraction (see `NOTICE.md`)

Overview (from upstream README)
- Extract electrophysiological parameters from voltage traces.
- Core scripts:
  - `EphysExtraction/ephys_features.py`: feature extraction utilities
  - `EphysExtraction/ephys_extractor.py`: per-cell feature extraction
- Notebooks (data format conversion + sanity checks):
  - `GetFeaturesPipeline.ipynb` (M1, Tolias lab)
  - `GetFeaturesPipeline2.ipynb` (Hippocampus, Xiaolong lab)
  - `GetFeaturesPipelineL4.ipynb` (Layer 4 V1/S1, Tolias lab)
  - `GetFeaturesPipelineL4Neurolucida.ipynb` (Layer 4 V1)
- Upstream development: Yves Bernaerts (main); Supervisor: Philipp Berens.

Local usage
- For scripted NWB analysis, use `run_features_on_nwb.py`.
- Usage notes are in `user_manual.md`.

Installation
1) Create a virtual environment:

```powershell
python -m venv .venv
```

2) Activate it:

```powershell
.\.venv\Scripts\activate
```

3) Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install ipfx efel bluepyefe neuroconv nwbinspector pynwb
```

Before publishing or redistributing, review `NOTICE.md` and `CHANGES.md` for
attribution and local modifications.
