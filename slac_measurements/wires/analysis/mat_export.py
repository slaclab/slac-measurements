"""Export WireMeasurementAnalysisResult to MATLAB .mat format.

Produces files compatible with the MATLAB wirescan_gui (File > Open).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.io

if TYPE_CHECKING:
    from slac_measurements.wires.analysis.results import (
        WireMeasurementAnalysisResult,
    )


def datetime_to_matlab_datenum(dt: datetime) -> float:
    """Convert a Python datetime to a MATLAB datenum (days since Jan 0, 0000)."""
    MATLAB_EPOCH_OFFSET = 719529.0  # datenum('1970-01-01')
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return MATLAB_EPOCH_OFFSET + (dt - epoch).total_seconds() / 86400.0


def _build_name_map(mad_names: list[str]) -> dict[str, str]:
    """Build MAD→EPICS name map from slac_db YAML device configs."""
    import warnings

    try:
        import slac_db.config

        yaml_dir = slac_db.config.yaml()
    except (ImportError, Exception):
        warnings.warn(
            "slac_db not available; MAD names will not be converted to EPICS. "
            "Pass name_map explicitly to to_mat().",
            stacklevel=3,
        )
        return {}

    import pathlib

    import yaml

    mad_set = set(mad_names)
    name_map: dict[str, str] = {}

    yaml_path = pathlib.Path(yaml_dir)
    if not yaml_path.is_dir():
        warnings.warn(
            f"slac_db YAML directory not found: {yaml_path}. "
            "Pass name_map explicitly to to_mat().",
            stacklevel=3,
        )
        return {}

    for yaml_file in yaml_path.glob("*.yaml"):
        if not mad_set - set(name_map.keys()):
            break
        try:
            with open(yaml_file) as f:
                area_data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(area_data, dict):
            continue
        for device_type_data in area_data.values():
            if not isinstance(device_type_data, dict):
                continue
            for mad_name, device_info in device_type_data.items():
                if mad_name not in mad_set or mad_name in name_map:
                    continue
                if not isinstance(device_info, dict):
                    continue
                ctrl = device_info.get("controls_information", {})
                if ctrl.get("control_name"):
                    name_map[mad_name] = ctrl["control_name"]
                    continue
                pvs = ctrl.get("PVs", {})
                if pvs:
                    first_pv = next(iter(pvs.values()), "")
                    if first_pv and ":" in first_pv:
                        parts = first_pv.split(":")
                        if len(parts) >= 3:
                            name_map[mad_name] = ":".join(parts[:3])

    missing = mad_set - set(name_map.keys())
    if missing:
        warnings.warn(
            f"Could not find EPICS names for: {sorted(missing)}. "
            "These will use MAD names. Pass name_map to override.",
            stacklevel=3,
        )
    return name_map


def _load_wirescan_config() -> dict | None:
    """Load wirescan_config.json from known locations."""
    import json
    import os
    import pathlib

    config_locations = [
        pathlib.Path("/usr/local/lcls/tools/matlab/toolbox/wirescan_config.json"),
        pathlib.Path(os.environ.get("MATLABPATH", "")) / "wirescan_config.json",
        pathlib.Path.home() / "Documents" / "toolbox" / "wirescan_config.json",
    ]
    for config_path in config_locations:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _wire_to_sector(wire_mad: str, cfg: dict | None) -> str | None:
    """Look up which MATLAB GUI sector a wire belongs to."""
    if cfg is None:
        return None
    sectors = cfg.get("sectors", {})
    for sector_name, sector_data in sectors.items():
        if not isinstance(sector_data, dict):
            continue
        wire_list = sector_data.get("wireMADList", [])
        if wire_mad in wire_list:
            return sector_name
    return None


def _get_sector_pmt_list(area: str, cfg: dict | None = None) -> list[str]:
    """Get the full PMT/detector list for a sector from wirescan_config.json."""
    if cfg is None:
        cfg = _load_wirescan_config()

    if cfg is not None:
        sectors = cfg.get("sectors", {})
        if isinstance(sectors, dict) and area in sectors:
            pmt_list = sectors[area].get("PMTMADList", [])
            if pmt_list:
                return pmt_list

    # Fallback: wire_area_metadata.yaml from slac_db
    try:
        import slac_db.config

        import yaml

        meta_path = slac_db.config.package_data() / "wire_area_metadata.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                area_meta = yaml.safe_load(f)
            if area in area_meta and "detectors" in area_meta[area]:
                return [d.split(":")[0] for d in area_meta[area]["detectors"]]
    except Exception:
        pass
    return []


def analysis_result_to_mat(
    result: WireMeasurementAnalysisResult,
    filepath: str,
    *,
    name_map: dict[str, str] | None = None,
    include_rmat: bool = False,
    physics_model: str = "BLEM",
) -> str:
    """Export a WireMeasurementAnalysisResult as a MATLAB .mat file.

    The output file contains a single ``data`` struct variable matching
    the layout produced by the MATLAB ``wirescan_gui`` ``dataSave``
    function, so it can be loaded directly with ``File > Open``.

    Parameters
    ----------
    result : WireMeasurementAnalysisResult
        The analysis result to export.
    filepath : str
        Output ``.mat`` file path.
    name_map : dict[str, str], optional
        Mapping from MAD names (as stored in the Python result) to EPICS
        names (as expected by MATLAB).  E.g.
        ``{"WS28744": "WIRE:LI28:744", "PMT29150": "PMT:LI29:150"}``.
        If None, attempts to look up names via ``slac_db`` (requires DB
        access); falls back to using MAD names unchanged.
    include_rmat : bool
        Whether to fetch R-matrices from the optics model.  Requires
        network access.  Default False.
    physics_model : str
        Model source for R-matrix retrieval.  Default ``"BLEM"``.

    Returns
    -------
    str
        Path to the saved ``.mat`` file.
    """
    from slac_measurements.wires.analysis.coordinates import stage_to_beam

    metadata = result.collection_result.metadata
    raw_data = result.collection_result.raw_data

    # Build or auto-detect MAD→EPICS name map
    if name_map is None:
        all_mad_names = [metadata.wire_name]
        if metadata.detectors:
            all_mad_names.extend(metadata.detectors)
        all_mad_names.extend(k for k in raw_data.keys() if k != metadata.wire_name)
        # Also include base MAD names (strip beampath suffix) for lookup
        base_names = {n.rsplit(":", 1)[0] for n in all_mad_names}
        all_mad_names = list(set(all_mad_names) | base_names)
        name_map = _build_name_map(all_mad_names)

    def _epics(mad_name: str) -> str:
        return name_map.get(mad_name, mad_name)

    data: dict[str, Any] = {}

    # --- Scalar / string fields ---
    wire_epics = _epics(metadata.wire_name)
    data["name"] = wire_epics
    data["wireName"] = wire_epics
    data["wireMode"] = "wire"
    data["beampath"] = metadata.beampath or ""
    data["status"] = np.bool_(True)

    if metadata.timestamp is not None:
        data["ts"] = datetime_to_matlab_datenum(metadata.timestamp)
    else:
        data["ts"] = np.float64(0.0)

    # --- Wire geometry ---
    data["wireAngle"] = np.float64(metadata.install_angle)
    data["wireScanDir"] = np.float64(0.0)

    wire_dir = {}
    wire_limit = {}
    wire_center = {}
    wire_size = {}
    for tag in ("x", "y", "u"):
        wire_dir[tag] = np.bool_(tag in metadata.active_profiles)
        if tag in metadata.scan_ranges:
            lo, hi = metadata.scan_ranges[tag]
            wire_limit[tag] = np.array([lo, hi], dtype=np.float64)
        else:
            wire_limit[tag] = np.array([0.0, 0.0], dtype=np.float64)
        wire_center[tag] = np.float64(0.0)
        wire_size[tag] = np.float64(12.5)

    data["wireDir"] = wire_dir
    data["wireLimit"] = wire_limit
    data["wireCenter"] = wire_center
    data["wireSize"] = wire_size

    # --- Device lists and raw data ---
    pmt_keys, bpm_keys, toro_keys = _classify_raw_data_keys(raw_data, metadata)

    # Get the full sector PMT list so indices match the GUI dropdown.
    # Python detector names carry a beampath suffix (e.g. "PMT29150:LI29")
    # while the config uses short MAD names (e.g. "PMT29150").  Build a
    # lookup from base MAD name → collected (suffixed) name for matching.
    base_to_collected = {k.rsplit(":", 1)[0]: k for k in pmt_keys}
    # metadata.area comes from the device DB (e.g. "L3") which doesn't match
    # the MATLAB GUI sector names (e.g. "LI28").  Look up the wire's sector
    # from the config's wireMADList (handles e.g. WS27644 → LI28).
    cfg = _load_wirescan_config()
    sector = _wire_to_sector(metadata.wire_name, cfg) or metadata.area
    full_pmt_keys = _get_sector_pmt_list(sector, cfg)
    if full_pmt_keys:
        pmt_keys_ordered = full_pmt_keys
        # Ensure name_map covers config-only detectors (e.g. PMT28750)
        unmapped = [k for k in full_pmt_keys if k not in name_map]
        if unmapped:
            name_map.update(_build_name_map(unmapped))
    else:
        pmt_keys_ordered = pmt_keys
        base_to_collected = {k: k for k in pmt_keys}

    pmt_epics = [_epics(k) for k in pmt_keys_ordered]
    bpm_epics = [_epics(k) for k in bpm_keys]
    toro_epics = [_epics(k) for k in toro_keys]

    data["PMTList"] = (
        np.array(pmt_epics, dtype=object) if pmt_epics else np.array([], dtype=object)
    )
    data["BPMList"] = (
        np.array(bpm_epics, dtype=object) if bpm_epics else np.array([], dtype=object)
    )
    data["toroList"] = (
        np.array(toro_epics, dtype=object) if toro_epics else np.array([], dtype=object)
    )

    # Use the longest active profile to determine n_pulses for raw arrays.
    # Export unique positions so MATLAB's reprocessing (interp1) won't fail.
    wire_key = metadata.wire_name
    if wire_key in raw_data:
        wire_arr = np.asarray(raw_data[wire_key], dtype=np.float64)
        # Deduplicate: keep only unique positions (sorted), averaging
        # signal at duplicate positions — matches what Python analysis does.
        unique_pos, inverse = np.unique(wire_arr, return_inverse=True)
        data["wireData"] = unique_pos.reshape(1, -1)
        data["wireMask"] = np.ones_like(data["wireData"], dtype=np.bool_)
    else:
        unique_pos = np.zeros(0, dtype=np.float64)
        inverse = np.array([], dtype=int)
        data["wireData"] = np.zeros((1, 0), dtype=np.float64)
        data["wireMask"] = np.zeros((1, 0), dtype=np.bool_)

    n_pulses = data["wireData"].shape[1]

    if pmt_keys_ordered:
        pmt_arrays = []
        for key in pmt_keys_ordered:
            collected_key = base_to_collected.get(key)
            if collected_key:
                arr = np.asarray(
                    raw_data.get(collected_key, np.zeros(len(inverse))),
                    dtype=np.float64,
                ).ravel()
                if len(inverse) > 0 and len(arr) == len(inverse):
                    deduped = np.zeros(n_pulses, dtype=np.float64)
                    counts = np.zeros(n_pulses, dtype=np.float64)
                    np.add.at(deduped, inverse, arr)
                    np.add.at(counts, inverse, 1.0)
                    counts[counts == 0] = 1.0
                    pmt_arrays.append(deduped / counts)
                else:
                    pmt_arrays.append(
                        arr[:n_pulses] if len(arr) >= n_pulses else np.zeros(n_pulses)
                    )
            else:
                pmt_arrays.append(np.zeros(n_pulses, dtype=np.float64))
        data["PMTData"] = np.array(pmt_arrays).reshape(len(pmt_keys_ordered), n_pulses)
    else:
        data["PMTData"] = np.zeros((1, n_pulses), dtype=np.float64)

    def _dedup_array(arr_raw, inv, n_out):
        """Average raw array values at duplicate wire positions."""
        if len(inv) > 0 and len(arr_raw) == len(inv):
            deduped = np.zeros(n_out, dtype=np.float64)
            counts = np.zeros(n_out, dtype=np.float64)
            np.add.at(deduped, inv, arr_raw)
            np.add.at(counts, inv, 1.0)
            counts[counts == 0] = 1.0
            return deduped / counts
        return arr_raw[:n_out] if len(arr_raw) >= n_out else np.zeros(n_out)

    if bpm_keys:
        bpmx_arrays = []
        bpmy_arrays = []
        for key in bpm_keys:
            bpm_val = raw_data.get(key, {})
            if isinstance(bpm_val, dict):
                raw_x = np.asarray(
                    bpm_val.get("x", np.zeros(len(inverse))), dtype=np.float64
                ).ravel()
                raw_y = np.asarray(
                    bpm_val.get("y", np.zeros(len(inverse))), dtype=np.float64
                ).ravel()
                bpmx_arrays.append(_dedup_array(raw_x, inverse, n_pulses))
                bpmy_arrays.append(_dedup_array(raw_y, inverse, n_pulses))
            else:
                bpmx_arrays.append(np.zeros(n_pulses, dtype=np.float64))
                bpmy_arrays.append(np.zeros(n_pulses, dtype=np.float64))
        data["BPMXData"] = np.array(bpmx_arrays).reshape(len(bpm_keys), n_pulses)
        data["BPMYData"] = np.array(bpmy_arrays).reshape(len(bpm_keys), n_pulses)
    else:
        data["BPMXData"] = np.zeros((0, n_pulses), dtype=np.float64)
        data["BPMYData"] = np.zeros((0, n_pulses), dtype=np.float64)

    if toro_keys:
        toro_arrays = []
        for key in toro_keys:
            raw_t = np.asarray(
                raw_data.get(key, np.zeros(len(inverse))), dtype=np.float64
            ).ravel()
            toro_arrays.append(_dedup_array(raw_t, inverse, n_pulses))
        data["toroData"] = np.array(toro_arrays).reshape(len(toro_keys), n_pulses)
    else:
        # GUI always expects at least one toroid row
        area = metadata.area or "LI21"
        toro_keys = [f"TORO:{area}:1"]
        data["toroList"] = np.array(toro_keys, dtype=object)
        data["toroData"] = np.ones((1, n_pulses), dtype=np.float64) * 1e9

    data["selectToro"] = np.float64(1.0)
    data["selectBPM"] = (
        np.ones((1, len(bpm_keys)), dtype=np.float64)
        if bpm_keys
        else np.zeros((1, 0), dtype=np.float64)
    )

    # --- R-matrices (optional) ---
    if include_rmat and bpm_keys:
        try:
            data["rMatList"] = _fetch_rmat(
                metadata.wire_name, bpm_keys, metadata.beampath, physics_model
            )
        except Exception:
            data["rMatList"] = np.zeros((4, 6, len(bpm_keys)), dtype=np.float64)
    else:
        n_bpm = len(bpm_keys)
        data["rMatList"] = (
            np.zeros((4, 6, n_bpm), dtype=np.float64)
            if n_bpm
            else np.zeros((4, 6, 0), dtype=np.float64)
        )

    # --- Calibrated profiles (pos / signal) ---
    default_det = metadata.default_detector
    pos = {}
    signal = {}
    for tag in ("x", "y", "u"):
        if tag in result.profiles and default_det:
            prof = result.profiles[tag]
            positions_beam = stage_to_beam(prof.positions, tag, metadata.install_angle)
            pos[tag] = positions_beam.astype(np.float64)
            det_data = prof.detectors.get(default_det)
            signal[tag] = (
                det_data.values.astype(np.float64)
                if det_data is not None
                else np.zeros_like(positions_beam)
            )
        else:
            pos[tag] = np.zeros(0, dtype=np.float64)
            signal[tag] = np.zeros(0, dtype=np.float64)

    data["pos"] = pos
    data["signal"] = signal

    if default_det and pmt_keys_ordered:
        default_det_base = (
            default_det.rsplit(":", 1)[0] if full_pmt_keys else default_det
        )
        try:
            data["selectPMT"] = np.float64(pmt_keys_ordered.index(default_det_base) + 1)
        except ValueError:
            data["selectPMT"] = np.float64(1.0)
    else:
        data["selectPMT"] = np.float64(1.0)

    # --- Beam struct (fit results) ---
    data["beam"] = _build_beam_struct(result, pos, signal)

    # --- beamPV ---
    data["beamPV"] = _build_beam_pv(wire_epics, data["beam"])

    scipy.io.savemat(filepath, {"data": data}, do_compression=True, oned_as="row")
    return filepath


def _classify_raw_data_keys(
    raw_data: dict[str, Any], metadata: Any
) -> tuple[list[str], list[str], list[str]]:
    """Separate raw_data keys into PMT, BPM, and toroid lists."""
    wire_name = metadata.wire_name
    detector_set = set(metadata.detectors) if metadata.detectors else set()

    pmt_keys: list[str] = []
    bpm_keys: list[str] = []
    toro_keys: list[str] = []

    for key in sorted(raw_data.keys()):
        if key == wire_name:
            continue
        if key in detector_set:
            pmt_keys.append(key)
        elif "BPM" in key.upper():
            bpm_keys.append(key)
        elif "TORO" in key.upper():
            toro_keys.append(key)
        elif isinstance(raw_data[key], dict) and "x" in raw_data[key]:
            bpm_keys.append(key)

    # Ensure pmt_keys matches detector order from metadata
    if metadata.detectors:
        ordered_pmt = [d for d in metadata.detectors if d in set(pmt_keys)]
        pmt_keys = ordered_pmt

    return pmt_keys, bpm_keys, toro_keys


def _build_beam_struct(
    result: WireMeasurementAnalysisResult,
    pos: dict[str, np.ndarray],
    signal: dict[str, np.ndarray],
) -> np.ndarray:
    """Build MATLAB-compatible beam struct array.

    Returns a numpy object array of shape (1, 1) containing a dict,
    which scipy.io.savemat renders as a 1x1 struct array.
    """
    from slac_measurements.wires.analysis.coordinates import stage_to_beam

    metadata = result.collection_result.metadata
    default_det = metadata.default_detector
    fit_result = result.fit_result

    method_name = result.fitting_method.capitalize()
    if method_name == "Gaussian":
        method_name = "Gaussian"

    stats = np.zeros(6, dtype=np.float64)
    x_stat = np.zeros(5, dtype=np.float64)
    y_stat = np.zeros(5, dtype=np.float64)
    u_stat = np.zeros(5, dtype=np.float64)
    stat_map = {"x": x_stat, "y": y_stat, "u": u_stat}

    for i, tag in enumerate(("x", "y")):
        if tag in fit_result and default_det:
            det_fit = fit_result[tag].detectors.get(default_det)
            if det_fit is not None:
                mean_beam = stage_to_beam(
                    np.array([det_fit.mean]), tag, metadata.install_angle
                )[0]
                stats[i] = mean_beam
                stats[i + 2] = det_fit.sigma
                stat_map[tag][1] = mean_beam
                stat_map[tag][2] = det_fit.sigma

    # XY correlation from u-plane if available
    if "u" in fit_result and default_det:
        u_fit = fit_result["u"].detectors.get(default_det)
        if u_fit is not None:
            sigma_u = u_fit.sigma
            sigma_x = stats[2] if stats[2] > 0 else 0.0
            sigma_y = stats[3] if stats[3] > 0 else 0.0
            stats[4] = (sigma_x**2 + sigma_y**2 - sigma_u**2) / 2.0
            stat_map["u"][1] = stage_to_beam(
                np.array([u_fit.mean]), "u", metadata.install_angle
            )[0]
            stat_map["u"][2] = sigma_u

    # SUM: integrate signal for each profile
    for tag in ("x", "y", "u"):
        if tag in result.profiles and default_det:
            det_data = result.profiles[tag].detectors.get(default_det)
            if det_data is not None and len(det_data.values) > 1:
                p = pos[tag]
                if len(p) > 1:
                    dx = np.mean(np.abs(np.diff(p)))
                    stat_map[tag][0] = float(np.sum(det_data.values) * dx)

    stats[5] = stat_map["x"][0]

    # Build prof arrays: [3 x N] = [positions; signal; fit_curve]
    beam_entry: dict[str, Any] = {
        "method": method_name,
        "stats": stats.reshape(1, 6),
        "statsStd": np.zeros((1, 6), dtype=np.float64),
        "xStat": x_stat.reshape(1, 5),
        "xStatStd": np.zeros((1, 5), dtype=np.float64),
        "yStat": y_stat.reshape(1, 5),
        "yStatStd": np.zeros((1, 5), dtype=np.float64),
        "uStat": u_stat.reshape(1, 5),
        "uStatStd": np.zeros((1, 5), dtype=np.float64),
    }

    for tag in ("x", "y", "u"):
        prof_key = f"prof{tag}"
        if tag in result.profiles and tag in fit_result and default_det:
            det_fit = fit_result[tag].detectors.get(default_det)
            p = pos[tag]
            s = signal[tag]
            if det_fit is not None and len(p) > 0:
                fit_curve = np.interp(p, det_fit.positions, det_fit.curve)
                beam_entry[prof_key] = np.vstack([p, s, fit_curve])
            else:
                beam_entry[prof_key] = np.zeros((3, 0), dtype=np.float64)
        else:
            beam_entry[prof_key] = np.zeros((3, 0), dtype=np.float64)

    # scipy.io.savemat represents struct arrays as object arrays of dicts
    beam_array = np.empty((1, 1), dtype=object)
    beam_array[0, 0] = beam_entry
    return beam_array


def _build_beam_pv(wire_name: str, beam: np.ndarray) -> np.ndarray:
    """Build MATLAB-compatible beamPV struct array."""
    beam_entry = beam[0, 0]
    stats = beam_entry["stats"].ravel()

    names = [
        f"{wire_name}:X",
        f"{wire_name}:Y",
        f"{wire_name}:XRMS",
        f"{wire_name}:YRMS",
        f"{wire_name}:XY",
        f"{wire_name}:SUM",
    ]
    descs = [
        "X position",
        "Y position",
        "X rms",
        "Y rms",
        "XY corr",
        "profile intensity",
    ]
    egus = ["um", "um", "um", "um", "um^2", "cts"]

    pv_array = np.empty((6, 1), dtype=object)
    for i in range(6):
        pv_array[i, 0] = {
            "name": names[i],
            "val": np.float64(stats[i]),
            "desc": descs[i],
            "egu": egus[i],
        }
    return pv_array


def _fetch_rmat(
    wire_name: str,
    bpm_keys: list[str],
    beampath: str | None,
    physics_model: str,
) -> np.ndarray:
    """Fetch R-matrices from the optics model (requires network)."""
    from lcls_tools.common.devices.reader import create_wire

    device = create_wire(wire_name)
    rmat_list = []
    for bpm in bpm_keys:
        try:
            rmat = device.get_rmat(bpm, model=physics_model)
            rmat_list.append(rmat[:4, :6])
        except Exception:
            rmat_list.append(np.zeros((4, 6), dtype=np.float64))

    return (
        np.stack(rmat_list, axis=2)
        if rmat_list
        else np.zeros((4, 6, 0), dtype=np.float64)
    )
