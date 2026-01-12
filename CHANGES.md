Changes vs upstream
-------------------

The upstream codebase is the contents of `EphysExtraction/` from:
https://github.com/berenslab/EphysExtraction 


Chnage summary
---------------------------------------------------------

Added:
- `run_features_on_nwb.py`
- `EphysExtraction/test_sanity.py`

Modified:
- `EphysExtraction/ephys_extractor.py`
- `EphysExtraction/ephys_features.py`

Deleted:
- None detected

Medium-detail notes
-------------------

`EphysExtraction/ephys_extractor.py`:
- Replaced deprecated pandas `.ix` indexing with `.loc` when populating spike feature DataFrames.
- Fixed baseline detection thresholding (`np.abs(dv) >= thresh` instead of `np.abs(dv >= thresh)`).
- Replaced `map(...)` usage with list comprehensions when building numpy arrays of sweep amplitudes.
- Made fixed window handling explicit (`fixed_start is not None` checks).

`EphysExtraction/ephys_features.py`:
- Added guard in `average_voltage` to return `NaN` when the requested window is empty.
- Corrected RMSE calculation to use squared residuals.
- In `fit_membrane_time_constant`, used the `_exp_curve_at_end` prediction for the fit quality check.

For a precise patch diff, compare the local `EphysExtraction/` folder against
the upstream repository.
