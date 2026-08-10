"""Shared motor-polling utilities for wire motion tests."""

import logging
import time

from slac_devices.wire import Wire

POLL_INTERVAL_S = 0.05  # 20 Hz
SETTLE_THRESHOLD_UM = 50  # position change below this = stopped
SETTLE_COUNT = 20  # consecutive "stopped" readings to confirm done

logger = logging.getLogger(__name__)


def poll_motor_rbv(
    device: Wire,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> list[float]:
    """Poll motor RBV until motion completes. Returns list of positions."""
    positions = []
    settle_count = 0
    start = time.monotonic()

    while True:
        pos = device.motor_rbv
        positions.append(pos)

        if len(positions) > 1:
            if abs(positions[-1] - positions[-2]) < SETTLE_THRESHOLD_UM:
                settle_count += 1
            else:
                settle_count = 0

            if settle_count >= SETTLE_COUNT:
                elapsed = time.monotonic() - start
                logger.info("Wire settled at %.1f um after %.1fs", pos, elapsed)
                break

        time.sleep(poll_interval_s)

    elapsed = time.monotonic() - start
    logger.info("Collected %d position samples over %.1fs", len(positions), elapsed)
    return positions
