"""Step-scan motion primitives: initialize, compute positions, and move."""

import logging

import slac_measurements.utils

_WIRE_TOLERANCE = 250  # microns
_WIRE_RETRACT_WAIT = 2  # seconds

logger = logging.getLogger(__name__)


def initialize_step_with_retry(device, logger=logger, max_attempts=3):
    """Initialize wire for step scan mode with retries until enabled.

    initialize is idempotent — skip if wire is already enabled.
    """
    if device.enabled:
        logger.info("%s is already enabled.", device.name)
        return

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "Initializing %s for step scan (Attempt %s/%s)...",
            device.name,
            attempt,
            max_attempts,
        )
        device.initialize()

        if slac_measurements.utils.wait_until(lambda: device.enabled):
            logger.info("%s initialized (enabled is True).", device.name)
            return

        logger.warning("%s did not enable - retrying...", device.name)

    raise RuntimeError(
        f"Failed to initialize {device.name} after {max_attempts} attempts."
    )


def get_step_positions(device):
    """Return sorted inner and outer positions for active profiles."""
    positions = []
    for profile in device.active_profiles():
        for mode in ["inner", "outer"]:
            positions.append(getattr(device, f"{profile}_wire_{mode}"))
    return sorted(positions)


def move_to_step_position(
    device, logger=logger, *, position, position_index, total_positions, positions
):
    """Move wire to a step position and wait for arrival."""

    def calculate_speed():
        if position_index % 2 == 0:
            return int(device.speed_max)
        position_delta = positions[position_index] - positions[position_index - 1]
        speed = (position_delta / device.scan_pulses) * device.beam_rate
        return int(speed)

    logger.info(
        "Moving wire to %s (step %s/%s)...",
        position,
        position_index + 1,
        total_positions,
    )

    device.speed = calculate_speed()
    device.motor = position

    if not slac_measurements.utils.wait_until(
        lambda: abs(device.motor_rbv - position) < _WIRE_TOLERANCE,
    ):
        raise RuntimeError(
            f"{device.name} did not reach position {position} after 10s."
        )
