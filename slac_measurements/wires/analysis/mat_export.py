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


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def datetime_to_matlab_datenum(dt: datetime) -> float:
    """Convert a Python datetime to a MATLAB datenum (days since Jan 0, 0000)."""
    MATLAB_EPOCH_OFFSET = 719529.0  # datenum('1970-01-01')
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return MATLAB_EPOCH_OFFSET + (dt - epoch).total_seconds() / 86400.0


def _dedup_array(raw: np.ndarray, inverse: np.ndarray, n_out: int) -> np.ndarray:
    """Average raw array values at duplicate wire positions."""
    if len(inverse) > 0 and len(raw) == len(inverse):
        deduped = np.zeros(n_out, dtype=np.float64)
        counts = np.zeros(n_out, dtype=np.float64)
        np.add.at(deduped, inverse, raw)
        np.add.at(counts, inverse, 1.0)
        counts[counts == 0] = 1.0
        return deduped / counts
    return raw[:n_out] if len(raw) >= n_out else np.zeros(n_out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Name map (MAD → EPICS)
# ---------------------------------------------------------------------------


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


def _build_auto_name_map(metadata: Any, raw_data: dict) -> dict[str, str]:
    """Auto-detect MAD→EPICS name map from metadata + raw_data keys."""
    all_mad_names = [metadata.wire_name]
    if metadata.detectors:
        all_mad_names.extend(metadata.detectors)
    all_mad_names.extend(k for k in raw_data.keys() if k != metadata.wire_name)
    base_names = {n.rsplit(":", 1)[0] for n in all_mad_names}
    return _build_name_map(list(set(all_mad_names) | base_names))


# ---------------------------------------------------------------------------
# Sector / config lookup
# ---------------------------------------------------------------------------


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
        if wire_mad in sector_data.get("wireMADList", []):
            return sector_name
    return None


def _get_sector_pmt_list(area: str, cfg: dict | None = None) -> list[str]:
    """Get the full PMT/detector list for a sector."""
    if cfg is None:
        cfg = _load_wirescan_config()

    if cfg is not None:
        sectors = cfg.get("sectors", {})
        if isinstance(sectors, dict) and area in sectors:
            pmt_list = sectors[area].get("PMTMADList", [])
            if pmt_list:
                return pmt_list

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


# ---------------------------------------------------------------------------
# Device classification and resolution
# ---------------------------------------------------------------------------


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

    if metadata.detectors:
        pmt_keys = [d for d in metadata.detectors if d in set(pmt_keys)]

    return pmt_keys, bpm_keys, toro_keys


def _resolve_devices(
    metadata: Any, raw_data: dict, name_map: dict[str, str]
) -> tuple[list[str], dict[str, str], list[str], list[str], bool]:
    """Resolve device lists and align PMT ordering with the MATLAB config.

    Returns (pmt_keys_ordered, base_to_collected, bpm_keys, toro_keys, has_sector_config).
    """
    pmt_keys, bpm_keys, toro_keys = _classify_raw_data_keys(raw_data, metadata)

    base_to_collected = {k.rsplit(":", 1)[0]: k for k in pmt_keys}

    cfg = _load_wirescan_config()
    sector = _wire_to_sector(metadata.wire_name, cfg) or metadata.area
    full_pmt_keys = _get_sector_pmt_list(sector, cfg)

    if full_pmt_keys:
        pmt_keys_ordered = full_pmt_keys
        unmapped = [k for k in full_pmt_keys if k not in name_map]
        if unmapped:
            name_map.update(_build_name_map(unmapped))
    else:
        pmt_keys_ordered = pmt_keys
        base_to_collected = {k: k for k in pmt_keys}

    return pmt_keys_ordered, base_to_collected, bpm_keys, toro_keys, bool(full_pmt_keys)


# ---------------------------------------------------------------------------
# Raw data array builders
# ---------------------------------------------------------------------------


def _build_wire_data(
    raw_data: dict, wire_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Deduplicate wire positions. Returns (wireData, wireMask, inverse, n_pulses)."""
    if wire_key in raw_data:
        wire_arr = np.asarray(raw_data[wire_key], dtype=np.float64)
        unique_pos, inverse = np.unique(wire_arr, return_inverse=True)
        wire_data = unique_pos.reshape(1, -1)
        wire_mask = np.ones_like(wire_data, dtype=np.bool_)
    else:
        inverse = np.array([], dtype=int)
        wire_data = np.zeros((1, 0), dtype=np.float64)
        wire_mask = np.zeros((1, 0), dtype=np.bool_)

    return wire_data, wire_mask, inverse, wire_data.shape[1]


def _build_pmt_data(
    raw_data: dict,
    pmt_keys_ordered: list[str],
    base_to_collected: dict[str, str],
    inverse: np.ndarray,
    n_pulses: int,
) -> np.ndarray:
    """Build PMTData array aligned to sector config order."""
    if not pmt_keys_ordered:
        return np.zeros((1, n_pulses), dtype=np.float64)

    rows = []
    for key in pmt_keys_ordered:
        collected_key = base_to_collected.get(key)
        if collected_key:
            arr = np.asarray(
                raw_data.get(collected_key, np.zeros(len(inverse))), dtype=np.float64
            ).ravel()
            rows.append(_dedup_array(arr, inverse, n_pulses))
        else:
            rows.append(np.zeros(n_pulses, dtype=np.float64))

    return np.array(rows).reshape(len(pmt_keys_ordered), n_pulses)


def _build_bpm_data(
    raw_data: dict,
    bpm_keys: list[str],
    inverse: np.ndarray,
    n_pulses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BPMXData and BPMYData arrays."""
    if not bpm_keys:
        return (
            np.zeros((0, n_pulses), dtype=np.float64),
            np.zeros((0, n_pulses), dtype=np.float64),
        )

    x_rows, y_rows = [], []
    for key in bpm_keys:
        bpm_val = raw_data.get(key, {})
        if isinstance(bpm_val, dict):
            raw_x = np.asarray(
                bpm_val.get("x", np.zeros(len(inverse))), dtype=np.float64
            ).ravel()
            raw_y = np.asarray(
                bpm_val.get("y", np.zeros(len(inverse))), dtype=np.float64
            ).ravel()
            x_rows.append(_dedup_array(raw_x, inverse, n_pulses))
            y_rows.append(_dedup_array(raw_y, inverse, n_pulses))
        else:
            x_rows.append(np.zeros(n_pulses, dtype=np.float64))
            y_rows.append(np.zeros(n_pulses, dtype=np.float64))

    return (
        np.array(x_rows).reshape(len(bpm_keys), n_pulses),
        np.array(y_rows).reshape(len(bpm_keys), n_pulses),
    )


def _build_toro_data(
    raw_data: dict,
    toro_keys: list[str],
    inverse: np.ndarray,
    n_pulses: int,
    area: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build toroData and toroList. Returns (toroData, toroList)."""
    if toro_keys:
        rows = []
        for key in toro_keys:
            raw_t = np.asarray(
                raw_data.get(key, np.zeros(len(inverse))), dtype=np.float64
            ).ravel()
            rows.append(_dedup_array(raw_t, inverse, n_pulses))
        return np.array(rows).reshape(len(toro_keys), n_pulses), None

    # GUI always expects at least one toroid row
    fallback_name = f"TORO:{area or 'LI21'}:1"
    return np.ones((1, n_pulses), dtype=np.float64) * 1e9, np.array(
        [fallback_name], dtype=object
    )


# ---------------------------------------------------------------------------
# Profile and beam struct builders
# ---------------------------------------------------------------------------


def _build_profiles(
    result: WireMeasurementAnalysisResult, metadata: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build calibrated pos/signal dicts from analysis profiles."""
    from slac_measurements.wires.analysis.coordinates import stage_to_beam

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
    return pos, signal


def _build_beam_struct(
    result: WireMeasurementAnalysisResult,
    pos: dict[str, np.ndarray],
    signal: dict[str, np.ndarray],
) -> np.ndarray:
    """Build MATLAB-compatible beam struct array (1x1 object array)."""
    from slac_measurements.wires.analysis.coordinates import stage_to_beam

    metadata = result.collection_result.metadata
    default_det = metadata.default_detector
    fit_result = result.fit_result

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

    for tag in ("x", "y", "u"):
        if tag in result.profiles and default_det:
            det_data = result.profiles[tag].detectors.get(default_det)
            if det_data is not None and len(det_data.values) > 1:
                p = pos[tag]
                if len(p) > 1:
                    dx = np.mean(np.abs(np.diff(p)))
                    stat_map[tag][0] = float(np.sum(det_data.values) * dx)

    stats[5] = stat_map["x"][0]

    beam_entry: dict[str, Any] = {
        "method": result.fitting_method.capitalize(),
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


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


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
        Mapping from MAD names to EPICS names.  If None, attempts auto-lookup
        via ``slac_db``; falls back to using MAD names unchanged.
    include_rmat : bool
        Whether to fetch R-matrices from the optics model.  Default False.
    physics_model : str
        Model source for R-matrix retrieval.  Default ``"BLEM"``.

    Returns
    -------
    str
        Path to the saved ``.mat`` file.
    """
    metadata = result.collection_result.metadata
    raw_data = result.collection_result.raw_data

    if name_map is None:
        name_map = _build_auto_name_map(metadata, raw_data)

    def _epics(mad_name: str) -> str:
        return name_map.get(mad_name, mad_name)

    wire_epics = _epics(metadata.wire_name)

    # --- Resolve device lists ---
    pmt_ordered, base_map, bpm_keys, toro_keys, has_sector = _resolve_devices(
        metadata, raw_data, name_map
    )

    # --- Wire positions (dedup) ---
    wire_data, wire_mask, inverse, n_pulses = _build_wire_data(
        raw_data, metadata.wire_name
    )

    # --- Raw signal arrays ---
    pmt_data = _build_pmt_data(raw_data, pmt_ordered, base_map, inverse, n_pulses)
    bpmx_data, bpmy_data = _build_bpm_data(raw_data, bpm_keys, inverse, n_pulses)
    toro_data, toro_fallback_list = _build_toro_data(
        raw_data, toro_keys, inverse, n_pulses, metadata.area
    )

    # --- EPICS name lists ---
    pmt_epics = [_epics(k) for k in pmt_ordered]
    bpm_epics = [_epics(k) for k in bpm_keys]
    toro_epics = [_epics(k) for k in toro_keys] if toro_keys else None

    # --- R-matrices ---
    n_bpm = len(bpm_keys)
    if include_rmat and bpm_keys:
        try:
            rmat = _fetch_rmat(
                metadata.wire_name, bpm_keys, metadata.beampath, physics_model
            )
        except Exception:
            rmat = np.zeros((4, 6, n_bpm), dtype=np.float64)
    else:
        rmat = np.zeros((4, 6, n_bpm), dtype=np.float64)

    # --- Profiles ---
    pos, signal = _build_profiles(result, metadata)

    # --- selectPMT ---
    default_det = metadata.default_detector
    if default_det and pmt_ordered:
        det_base = default_det.rsplit(":", 1)[0] if has_sector else default_det
        try:
            select_pmt = np.float64(pmt_ordered.index(det_base) + 1)
        except ValueError:
            select_pmt = np.float64(1.0)
    else:
        select_pmt = np.float64(1.0)

    # --- Assemble data struct ---
    data: dict[str, Any] = {
        "name": wire_epics,
        "wireName": wire_epics,
        "wireMode": "wire",
        "beampath": metadata.beampath or "",
        "status": np.bool_(True),
        "ts": (
            datetime_to_matlab_datenum(metadata.timestamp)
            if metadata.timestamp is not None
            else np.float64(0.0)
        ),
        "wireAngle": np.float64(metadata.install_angle),
        "wireScanDir": np.float64(0.0),
        "wireDir": {
            t: np.bool_(t in metadata.active_profiles) for t in ("x", "y", "u")
        },
        "wireLimit": {
            t: (
                np.array(metadata.scan_ranges[t], dtype=np.float64)
                if t in metadata.scan_ranges
                else np.array([0.0, 0.0], dtype=np.float64)
            )
            for t in ("x", "y", "u")
        },
        "wireCenter": {t: np.float64(0.0) for t in ("x", "y", "u")},
        "wireSize": {t: np.float64(12.5) for t in ("x", "y", "u")},
        "PMTList": np.array(pmt_epics, dtype=object)
        if pmt_epics
        else np.array([], dtype=object),
        "BPMList": np.array(bpm_epics, dtype=object)
        if bpm_epics
        else np.array([], dtype=object),
        "toroList": (
            toro_fallback_list
            if toro_fallback_list is not None
            else (
                np.array(toro_epics, dtype=object)
                if toro_epics
                else np.array([], dtype=object)
            )
        ),
        "wireData": wire_data,
        "wireMask": wire_mask,
        "PMTData": pmt_data,
        "BPMXData": bpmx_data,
        "BPMYData": bpmy_data,
        "toroData": toro_data,
        "selectToro": np.float64(1.0),
        "selectBPM": (
            np.ones((1, n_bpm), dtype=np.float64)
            if n_bpm
            else np.zeros((1, 0), dtype=np.float64)
        ),
        "rMatList": rmat,
        "pos": pos,
        "signal": signal,
        "selectPMT": select_pmt,
        "beam": _build_beam_struct(result, pos, signal),
    }
    data["beamPV"] = _build_beam_pv(wire_epics, data["beam"])

    scipy.io.savemat(filepath, {"data": data}, do_compression=True, oned_as="row")
    return filepath
