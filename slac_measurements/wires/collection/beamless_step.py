"""Buffer-less step-scan collection — validates motor travel at discrete positions."""

import logging
import time
from datetime import datetime
from threading import Thread

import numpy as np

from slac_devices.wire import Wire
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)
from slac_measurements.wires.motion.step import (
    _WIRE_RETRACT_WAIT,
    get_step_positions,
    initialize_step_with_retry,
    move_to_step_position,
)
from slac_measurements.wires.motion.utils import poll_motor_rbv

logger = logging.getLogger(__name__)

STEP_SCAN_TIMEOUT = 60


def run_beamless_step_scan(
    device: Wire,
) -> WireMeasurementCollectionResult:
    """
    Execute step-scan style wire motion without buffer and record motor RBV.

    Moves the wire to each inner/outer position for all active profiles,
    verifying that the motor reaches each target within tolerance before
    advancing.

    Parameters
    ----------
    device : Wire
        Initialized wire device (from slac_devices.reader.create_wire).

    Returns
    -------
    WireMeasurementCollectionResult
        Position array in raw_data[device.name], metadata with scan ranges.
    """
    initialize_step_with_retry(device, logger)
    positions = get_step_positions(device)

    def motion_sequence():
        _step_through_positions(device, positions)
        time.sleep(_WIRE_RETRACT_WAIT)
        device.retract()

    recorded = None

    def capture_poll():
        nonlocal recorded
        recorded = poll_motor_rbv(device)

    poll_thread = Thread(target=capture_poll)
    motion_sequence()
    poll_thread.start()
    poll_thread.join(timeout=STEP_SCAN_TIMEOUT)

    if poll_thread.is_alive():
        raise TimeoutError(
            f"Poll thread did not finish within {STEP_SCAN_TIMEOUT}s timeout"
        )

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
        notes="beamless step motion test",
    )

    return WireMeasurementCollectionResult(
        raw_data={device.name: np.array(recorded)},
        metadata=metadata,
    )


def _step_through_positions(
    device: Wire,
    positions: list[int],
) -> None:
    """Move to each position using the same motion as step collection."""
    total = len(positions)

    for i, target in enumerate(positions):
        move_to_step_position(
            device,
            logger,
            position=target,
            position_index=i,
            total_positions=total,
            positions=positions,
        )

    logger.info("Step motion complete: %d positions", total)
