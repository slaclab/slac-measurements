from datetime import datetime
from typing import Any

import h5py
import slac_tools.pydantic_h5
from pydantic import BaseModel, ConfigDict

from slac_measurements.beam_profile import BeamProfileCollectionResult


class MeasurementMetadata(BaseModel):
    wire_name: str
    buffer_number: int | None = None
    area: str
    beampath: str | None = None
    detectors: list[str] | None = None
    default_detector: str | None = None
    rms_detector: str | None = None
    scan_ranges: dict[str, tuple[int, int]]
    timestamp: datetime | None = None
    active_profiles: list[str]
    install_angle: float
    charge_toroids: list[str] | None = None
    notes: str | None = None


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
    raw_data: dict[str, Any]
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
            f"timestamp={meta.timestamp.isoformat() if meta.timestamp is not None else None})"
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
            slac_tools.pydantic_h5.save_model(
                self, f, manual=dict(scan_ranges=_write_scan_ranges)
            )


def load_from_h5(filepath: str) -> WireMeasurementCollectionResult:
    """
    Load wire beam profile measurement results from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to load.

    Returns
    -------
    WireMeasurementCollectionResult
        The loaded measurement results.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is missing required groups or data.
    """
    with h5py.File(filepath, "r") as f:
        return slac_tools.pydantic_h5.load_model(
            WireMeasurementCollectionResult,
            f,
            manual=dict(scan_ranges=_read_scan_ranges),
        )

def _read_scan_ranges(group: h5py.Group) -> dict[str, tuple[int, int]]:
    """Backward compatibility for existing hdf5 files:
    Inverse of `_write_scan_ranges`."""
    sub = group["scan_ranges"]
    axes = {k.rsplit("_", 1)[0] for k in sub.attrs.keys()}
    return {
        axis: (int(sub.attrs[f"{axis}_start"]), int(sub.attrs[f"{axis}_end"]))
        for axis in axes
    }

def _write_scan_ranges(value: dict[str, tuple[int, int]], group: h5py.Group) -> None:
    """Backward compatibility for existing hdf5 files:
    Flatten scan_ranges into a scan_ranges/ group of paired *_start/*_end attrs."""
    sub = group.create_group("scan_ranges")
    for axis_name, (start, end) in value.items():
        sub.attrs[f"{axis_name}_start"] = start
        sub.attrs[f"{axis_name}_end"] = end
