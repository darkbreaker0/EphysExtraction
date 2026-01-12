# ABF2 to NWB2 conversion script
# Uses pyabf + pynwb and writes minimal Icephys acquisition/stimulus series.

import argparse
import json
import os
import sys
from datetime import datetime

# Avoid importing local repo-level "pynwb" folder instead of installed package.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_SCRIPT_DIR, "pynwb")):
    sys.path = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]

import numpy as np
import pyabf

from pynwb import NWBFile, NWBHDF5IO, H5DataIO
from pynwb.icephys import (
    IntracellularElectrode,
    CurrentClampSeries,
    VoltageClampSeries,
    IZeroClampSeries,
    CurrentClampStimulusSeries,
    VoltageClampStimulusSeries,
)


CLAMP_V = 0
CLAMP_I = 1
CLAMP_I0 = 2


def unit_to_conversion(unit):
    if unit == "mV":
        return 1e-3, "V"
    if unit == "V":
        return 1.0, "V"
    if unit == "pA":
        return 1e-12, "A"
    if unit == "nA":
        return 1e-9, "A"
    if unit == "A":
        return 1.0, "A"
    return 1.0, unit


def clamp_from_unit(unit):
    if unit in ("V", "mV"):
        return CLAMP_I
    if unit in ("A", "pA", "nA"):
        return CLAMP_V
    return CLAMP_I


def acq_class_from_clamp(clamp_mode):
    if clamp_mode == CLAMP_V:
        return VoltageClampSeries
    if clamp_mode == CLAMP_I:
        return CurrentClampSeries
    if clamp_mode == CLAMP_I0:
        return IZeroClampSeries
    return CurrentClampSeries


def stim_class_from_clamp(clamp_mode):
    if clamp_mode == CLAMP_V:
        return VoltageClampStimulusSeries
    if clamp_mode == CLAMP_I:
        return CurrentClampStimulusSeries
    if clamp_mode == CLAMP_I0:
        return None
    return CurrentClampStimulusSeries


def create_nwbfile(abf):
    local_tz = datetime.now().astimezone().tzinfo
    session_start = abf.abfDateTime.replace(tzinfo=local_tz)

    session_description = abf.abfFileComment or abf.protocol or "ABF2 conversion"
    identifier = abf.abfID or os.path.basename(abf.abfFilePath)
    experiment_description = f"{abf.creator} {abf.creatorVersionString}".strip()

    return NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start,
        experiment_description=experiment_description,
        notes=abf.protocol,
    )


def select_indices(name_list, selected_names):
    if not selected_names:
        return list(range(len(name_list)))

    indices = []
    for name in selected_names:
        if name not in name_list:
            raise ValueError(f"Channel '{name}' not found. Available: {name_list}")
        indices.append(name_list.index(name))

    return indices


def convert_abf_to_nwb(in_path, out_path, acq_names=None, stim_names=None, compress=True):
    abf = pyabf.ABF(in_path)
    nwbfile = create_nwbfile(abf)

    device = nwbfile.create_device(name=f"{abf.creator} {abf.creatorVersionString}".strip())

    electrodes = []
    for idx, name in enumerate(abf.adcNames):
        elec = IntracellularElectrode(name=f"elec{idx}", device=device, description=name)
        nwbfile.add_icephys_electrode(elec)
        electrodes.append(elec)

    acq_indices = select_indices(abf.adcNames, acq_names)
    stim_indices = select_indices(abf.dacNames, stim_names) if abf.dacNames else []

    rate = float(abf.dataRate)

    for sweep in range(abf.sweepCount):
        for ch in acq_indices:
            abf.setSweep(sweep, channel=ch, absoluteTime=True)
            data = abf.sweepY.astype(np.float32)
            if compress:
                data = H5DataIO(data=data, compression=True, chunks=True, shuffle=True, fletcher32=True)

            conv, unit = unit_to_conversion(abf.sweepUnitsY)
            clamp_mode = None
            if hasattr(abf, "_adcSection"):
                clamp_mode = int(abf._adcSection.nTelegraphMode[ch])
            if clamp_mode is None:
                clamp_mode = clamp_from_unit(abf.sweepUnitsY)

            series_class = acq_class_from_clamp(clamp_mode)
            start_time = float(abf.sweepX[0])

            description = json.dumps({
                "file": abf.abfID,
                "protocol": abf.protocol,
                "sweep": sweep,
                "adc_name": abf.adcNames[ch],
                "adc_unit": abf.sweepUnitsY,
            })

            acq = series_class(
                name=f"acq_{sweep:03d}_{ch}",
                data=data,
                sweep_number=np.uint64(sweep),
                electrode=electrodes[ch],
                gain=float(abf._adcSection.fADCProgrammableGain[ch]) if hasattr(abf, "_adcSection") else 1.0,
                resolution=np.nan,
                conversion=conv,
                starting_time=start_time,
                rate=rate,
                unit=unit,
                description=description,
                stimulus_description=abf.protocol,
            )
            nwbfile.add_acquisition(acq)

        for ch in stim_indices:
            if hasattr(abf, "_dacSection") and not abf._dacSection.nWaveformEnable[ch]:
                continue

            abf.setSweep(sweep, channel=ch, absoluteTime=True)
            stim_data = abf.sweepC.astype(np.float32)
            if compress:
                stim_data = H5DataIO(data=stim_data, compression=True, chunks=True, shuffle=True, fletcher32=True)

            stim_conv, stim_unit = unit_to_conversion(abf.sweepUnitsC)
            clamp_mode = None
            if hasattr(abf, "_adcSection"):
                clamp_mode = int(abf._adcSection.nTelegraphMode[0])
            if clamp_mode is None:
                clamp_mode = clamp_from_unit(abf.sweepUnitsC)

            stim_class = stim_class_from_clamp(clamp_mode)
            if stim_class is None:
                continue

            stim_desc = json.dumps({
                "file": abf.abfID,
                "protocol": abf.protocol,
                "sweep": sweep,
                "dac_name": abf.dacNames[ch],
                "dac_unit": abf.sweepUnitsC,
            })

            stim = stim_class(
                name=f"stim_{sweep:03d}_{ch}",
                data=stim_data,
                sweep_number=np.uint64(sweep),
                electrode=electrodes[0],
                gain=float(abf._dacSection.fDACScaleFactor[ch]) if hasattr(abf, "_dacSection") else 1.0,
                resolution=np.nan,
                conversion=stim_conv,
                starting_time=float(abf.sweepX[0]),
                rate=rate,
                unit=stim_unit,
                description=stim_desc,
                stimulus_description=abf.protocol,
            )
            nwbfile.add_stimulus(stim)

    with NWBHDF5IO(out_path, "w") as io:
        io.write(nwbfile)


def main():
    parser = argparse.ArgumentParser(description="Convert ABF2 files to NWB2.")
    parser.add_argument("input", help="Path to ABF2 file or folder.")
    parser.add_argument("--output", help="Output file (for file input) or output folder (for folder input).")
    parser.add_argument("--acq", nargs="*", default=None, help="ADC channel names to include.")
    parser.add_argument("--stim", nargs="*", default=None, help="DAC channel names to include.")
    parser.add_argument("--no-compress", action="store_true", help="Disable HDF5 compression.")

    args = parser.parse_args()

    in_path = args.input
    compress = not args.no_compress

    if os.path.isfile(in_path):
        out_path = args.output
        if not out_path:
            root, _ = os.path.splitext(in_path)
            out_path = root + ".nwb"
        convert_abf_to_nwb(in_path, out_path, args.acq, args.stim, compress=compress)
        print(f"Wrote {out_path}")
        return

    if not os.path.isdir(in_path):
        raise ValueError(f"Input path does not exist: {in_path}")

    out_dir = args.output or in_path
    os.makedirs(out_dir, exist_ok=True)

    for name in os.listdir(in_path):
        if not name.lower().endswith(".abf"):
            continue
        src = os.path.join(in_path, name)
        root, _ = os.path.splitext(name)
        dst = os.path.join(out_dir, root + ".nwb")
        convert_abf_to_nwb(src, dst, args.acq, args.stim, compress=compress)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
