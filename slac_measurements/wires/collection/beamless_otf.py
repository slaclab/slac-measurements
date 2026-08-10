"""Buffer-less OTF collection — validates motor travel by polling RBV."""

import logging
from datetime import datetime

import numpy as np

from slac_devices.wire import Wire
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)
from slac_measurements.wires.motion.otf import initialize_otf_with_retry
from slac_measurements.wires.motion.utils import POLL_INTERVAL_S, poll_motor_rbv

logger = logging.getLogger(__name__)


def run_beamless_otf_scan(
    device: Wire,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> WireMeasurementCollectionResult:
    """
    Execute OTF-style wire motion without buffer and record motor RBV.

    Parameters
    ----------
    device : Wire
        Initialized wire device (from slac_devices.reader.create_wire).
    poll_interval_s : float
        Seconds between RBV reads (default 0.05 = 20 Hz).

    Returns
    -------
    WireMeasurementCollectionResult
        Position array in raw_data[device.name], metadata with scan ranges.
    """
    initialize_otf_with_retry(device, logger)
    positions = poll_motor_rbv(device, poll_interval_s)

    metadata = MeasurementMetadata(
        wire_name=device.name,
        area=device.area,
        beampath=None,
        detectors=None,
        default_detector=None,
        scan_ranges={
            "x": tuple(device.x_range),
            "y": tuple(device.y_range),
            "u": tuple(device.u_range),
        },
        timestamp=datetime.now(),
        active_profiles=device.active_profiles(),
        install_angle=device.install_angle,
        notes="beamless otf motion test",
    )

    return WireMeasurementCollectionResult(
        raw_data={device.name: np.array(positions)},
        metadata=metadata,
    )
