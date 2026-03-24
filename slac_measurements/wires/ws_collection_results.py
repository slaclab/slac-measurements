from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict

from slac_measurements.beam_profile import BeamProfileCollectionResult


class MeasurementMetadata(BaseModel):
    wire_name: str
    area: str
    beampath: str
    detectors: list[str]
    default_detector: str
    scan_ranges: Dict[str, Tuple[int, int]]
    timestamp: datetime
    active_profiles: list[str]
    install_angle: float
    notes: Optional[str] = None


class WireMeasurementCollectionResult(BeamProfileCollectionResult):
    """
    Stores the results of a wire beam profile collection.

    Attributes:
        model_config: Allows use of non-standard types
                      like NDArrayAnnotatedType.
        metadata (MeasurementMetadata): Metadata information related to
                                        the measurement.

    Inherited Attributes:
        raw_data (dict): Dictionary of device data as np.ndarrays.
                         Keys are device names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_data: Dict[str, Any]
    metadata: MeasurementMetadata

    def __repr__(self) -> str:
        """Return a string representation of the WireMeasurementCollectionResult."""
        meta = self.metadata
        num_devices = len(self.raw_data)
        return (
            f"WireMeasurementCollectionResult("
            f"wire_name='{meta.wire_name}', "
            f"area='{meta.area}', "
            f"beampath='{meta.beampath}', "
            f"devices={num_devices}, "
            f"timestamp={meta.timestamp.isoformat()})"
        )

    def save_to_h5(self, filepath: str) -> None:
        """
        Save wire beam profile collection results to an HDF5 file.

        The file structure is organized as follows:
        - /metadata: Measurement metadata (wire_name, area, beampath, etc.)
        - /raw_data/{device_name}: Raw detector data

        Parameters
        ----------
        filepath : str
            Path where the HDF5 file will be saved.
        """
        with h5py.File(filepath, "w") as f:
            # Save metadata
            metadata_group = f.create_group("metadata")
            self._save_metadata(metadata_group)

            # Save raw data
            raw_data_group = f.create_group("raw_data")
            self._save_raw_data(raw_data_group)

    def _save_metadata(self, group: h5py.Group) -> None:
        """Save measurement metadata as HDF5 attributes and datasets."""
        meta = self.metadata

        # Store scalar metadata as attributes
        group.attrs["wire_name"] = meta.wire_name
        group.attrs["area"] = meta.area
        group.attrs["beampath"] = meta.beampath
        group.attrs["default_detector"] = meta.default_detector
        group.attrs["timestamp"] = meta.timestamp.isoformat()
        group.attrs["active_profiles"] = meta.active_profiles
        group.attrs["install_angle"] = meta.install_angle

        if meta.notes:
            group.attrs["notes"] = meta.notes

        # Store list of detectors
        group.create_dataset(
            "detectors",
            data=np.array(meta.detectors, dtype="S"),
        )

        # Store scan ranges as a structured dataset
        scan_ranges_group = group.create_group("scan_ranges")
        for axis_name, (start, end) in meta.scan_ranges.items():
            scan_ranges_group.attrs[f"{axis_name}_start"] = start
            scan_ranges_group.attrs[f"{axis_name}_end"] = end

    def _save_raw_data(self, group: h5py.Group) -> None:
        """Save raw detector and wire data."""
        for device_name, data in self.raw_data.items():
            if isinstance(data, np.ndarray):
                group.create_dataset(device_name, data=data)
            else:
                # Try to convert to numpy array
                try:
                    group.create_dataset(device_name, data=np.array(data))
                except (TypeError, ValueError):
                    # Store as string representation if conversion fails
                    group.attrs[f"{device_name}_unsupported"] = str(data)


def load_from_h5(filepath: str) -> WireMeasurementCollectionResult:
    """
    Load wire beam profile measurement results from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to load.

    Returns
    -------
    WireBeamProfileMeasurementResult
        The loaded measurement results.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is missing required groups or data.
    """
    with h5py.File(filepath, "r") as f:
        # Load metadata
        metadata = _load_metadata(f["metadata"])

        # Load raw data
        raw_data = _load_raw_data(f["raw_data"])

    return WireMeasurementCollectionResult(
        raw_data=raw_data,
        metadata=metadata,
    )


def _load_metadata(group: h5py.Group) -> MeasurementMetadata:
    """Load measurement metadata from HDF5 group."""
    # Load scalar attributes
    wire_name = group.attrs["wire_name"]
    area = group.attrs["area"]
    beampath = group.attrs["beampath"]
    default_detector = group.attrs["default_detector"]
    timestamp_str = group.attrs["timestamp"]
    timestamp = datetime.fromisoformat(timestamp_str)
    active_profiles = group.attrs["active_profiles"]
    install_angle = group.attrs["install_angle"]
    notes = group.attrs.get("notes", None)

    # Load detectors list
    detectors = [d.decode() if isinstance(d, bytes) else d for d in group["detectors"]]

    # Load scan ranges
    scan_ranges = {}
    scan_ranges_group = group["scan_ranges"]
    for axis_name in set(k.rsplit("_", 1)[0] for k in scan_ranges_group.attrs.keys()):
        start = scan_ranges_group.attrs[f"{axis_name}_start"]
        end = scan_ranges_group.attrs[f"{axis_name}_end"]
        scan_ranges[axis_name] = (start, end)

    return MeasurementMetadata(
        wire_name=wire_name,
        area=area,
        beampath=beampath,
        detectors=detectors,
        default_detector=default_detector,
        scan_ranges=scan_ranges,
        timestamp=timestamp,
        active_profiles=active_profiles,
        install_angle=install_angle,
        notes=notes,
    )


def _load_raw_data(group: h5py.Group) -> Dict[str, Any]:
    """Load raw detector data from HDF5 group."""
    raw_data = {}

    for device_name in group.keys():
        raw_data[device_name] = group[device_name][:]

    return raw_data


class MultiWireMeasurementCollectionResult(BaseModel):
    """
    Stores the results of multiple wire beam profile collections.

    Attributes:
        model_config: Allows use of non-standard types.
        wire_results (Dict[str, WireMeasurementCollectionResult]):
            Dictionary mapping wire names to measurement results.
        timestamp: Overall measurement timestamp.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    wire_results: Dict[str, WireMeasurementCollectionResult]
    timestamp: datetime

    def save_to_h5(self, filepath: str) -> None:
        """
        Save multi-wire beam profile collection results to an HDF5 file.

        The file structure is organized as follows:
        - /multi_wire_metadata: Overall measurement metadata
        - /wire_{wire_name}/metadata: Individual wire metadata
        - /wire_{wire_name}/raw_data: Individual wire raw data

        Parameters
        ----------
        filepath : str
            Path where the HDF5 file will be saved.
        """
        with h5py.File(filepath, "w") as f:
            # Save overall metadata
            f.attrs["timestamp"] = self.timestamp.isoformat()
            f.attrs["num_wires"] = len(self.wire_results)
            f.attrs["wire_names"] = list(self.wire_results.keys())

            # Save each wire's results
            for wire_name, result in self.wire_results.items():
                wire_group = f.create_group(f"wire_{wire_name}")

                # Save metadata
                metadata_group = wire_group.create_group("metadata")
                self._save_wire_metadata(metadata_group, result.metadata)

                # Save raw data
                raw_data_group = wire_group.create_group("raw_data")
                self._save_wire_raw_data(raw_data_group, result.raw_data)

    def _save_wire_metadata(
        self, group: h5py.Group, metadata: MeasurementMetadata
    ) -> None:
        """Save individual wire measurement metadata."""
        group.attrs["wire_name"] = metadata.wire_name
        group.attrs["area"] = metadata.area
        group.attrs["beampath"] = metadata.beampath
        group.attrs["default_detector"] = metadata.default_detector
        group.attrs["timestamp"] = metadata.timestamp.isoformat()
        group.attrs["active_profiles"] = metadata.active_profiles
        group.attrs["install_angle"] = metadata.install_angle

        if metadata.notes:
            group.attrs["notes"] = metadata.notes

        # Store list of detectors
        group.create_dataset(
            "detectors",
            data=np.array(metadata.detectors, dtype="S"),
        )

        # Store scan ranges
        scan_ranges_group = group.create_group("scan_ranges")
        for axis_name, (start, end) in metadata.scan_ranges.items():
            scan_ranges_group.attrs[f"{axis_name}_start"] = start
            scan_ranges_group.attrs[f"{axis_name}_end"] = end

    def _save_wire_raw_data(self, group: h5py.Group, raw_data: Dict[str, Any]) -> None:
        """Save individual wire raw detector data."""
        for device_name, data in raw_data.items():
            if isinstance(data, np.ndarray):
                group.create_dataset(device_name, data=data)
            else:
                try:
                    group.create_dataset(device_name, data=np.array(data))
                except (TypeError, ValueError):
                    group.attrs[f"{device_name}_unsupported"] = str(data)


def load_multi_wire_from_h5(
    filepath: str,
) -> MultiWireMeasurementCollectionResult:
    """
    Load multi-wire beam profile measurement results from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to load.

    Returns
    -------
    MultiWireMeasurementCollectionResult
        The loaded measurement results.
    """
    with h5py.File(filepath, "r") as f:
        timestamp = datetime.fromisoformat(f.attrs["timestamp"])
        wire_names = f.attrs["wire_names"]

        wire_results = {}
        for wire_name in wire_names:
            wire_group = f[f"wire_{wire_name}"]

            # Load metadata
            metadata = _load_wire_metadata_multi(wire_group["metadata"])

            # Load raw data
            raw_data = _load_raw_data(wire_group["raw_data"])

            wire_results[wire_name] = WireMeasurementCollectionResult(
                raw_data=raw_data,
                metadata=metadata,
            )

    return MultiWireMeasurementCollectionResult(
        wire_results=wire_results,
        timestamp=timestamp,
    )


def _load_wire_metadata_multi(group: h5py.Group) -> MeasurementMetadata:
    """Load individual wire metadata from multi-wire HDF5 group."""
    wire_name = group.attrs["wire_name"]
    area = group.attrs["area"]
    beampath = group.attrs["beampath"]
    default_detector = group.attrs["default_detector"]
    timestamp_str = group.attrs["timestamp"]
    timestamp = datetime.fromisoformat(timestamp_str)
    active_profiles = group.attrs["active_profiles"]
    install_angle = group.attrs["install_angle"]
    notes = group.attrs.get("notes", None)

    # Load detectors list
    detectors = [d.decode() if isinstance(d, bytes) else d for d in group["detectors"]]

    # Load scan ranges
    scan_ranges = {}
    scan_ranges_group = group["scan_ranges"]
    for axis_name in set(k.rsplit("_", 1)[0] for k in scan_ranges_group.attrs.keys()):
        start = scan_ranges_group.attrs[f"{axis_name}_start"]
        end = scan_ranges_group.attrs[f"{axis_name}_end"]
        scan_ranges[axis_name] = (start, end)

    return MeasurementMetadata(
        wire_name=wire_name,
        area=area,
        beampath=beampath,
        detectors=detectors,
        default_detector=default_detector,
        scan_ranges=scan_ranges,
        timestamp=timestamp,
        active_profiles=active_profiles,
        install_angle=install_angle,
        notes=notes,
    )
