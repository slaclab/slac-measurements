"""OTF motion primitives: initialize wire for on-the-fly scan."""

import logging

import slac_measurements.utils

logger = logging.getLogger(__name__)


def initialize_otf_with_retry(device, logger=logger, max_attempts=3):
    """Start OTF scan and retry until wire is homed and on status.

    start_scan must always be called to arm the wire for OTF motion,
    even if homed/on_status are already True from a prior run.
    """
    for attempt in range(1, max_attempts + 1):
        logger.info(
            "Starting OTF scan on %s (Attempt %s/%s)...",
            device.name,
            attempt,
            max_attempts,
        )
        device.start_scan()

        if slac_measurements.utils.wait_until(
            lambda: device.homed and device.on_status
        ):
            logger.info("%s is homed and on.", device.name)
            return

        logger.warning(
            "%s did not become homed and on - retrying...",
            device.name,
        )

    raise RuntimeError(
        f"Failed to initialize {device.name} after {max_attempts} attempts."
    )
