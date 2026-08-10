"""Step-scan wire collection: buffered data acquisition with discrete motion."""

import time

from slac_measurements.wires.collection.base import BaseWireMeasurementCollection
from slac_measurements.wires.motion.step import (
    get_step_positions,
    initialize_step_with_retry,
    move_to_step_position,
    _WIRE_RETRACT_WAIT,
)


class StepWireMeasurementCollection(BaseWireMeasurementCollection):
    """Collect wire scan data using discrete step motion."""

    def _run_collection_scan(self) -> None:
        """Run a step scan: init wire, start buffer, move positions, retract, wait."""
        self.logger.info("Performing step scan mode")

        initialize_step_with_retry(self.beam_profile_device, self.logger)

        self.logger.info("Starting buffer acquisition for step scan...")
        acquisition_start = time.monotonic()
        acquisition_timeout_s = self._calculate_acquisition_timeout_s()
        self.buffer.start()

        positions = get_step_positions(self.beam_profile_device)
        total_positions = len(positions)
        for index, position in enumerate(positions):
            move_to_step_position(
                self.beam_profile_device,
                self.logger,
                position=position,
                position_index=index,
                total_positions=total_positions,
                positions=positions,
            )

        self.logger.info("Retracting wire...")
        time.sleep(_WIRE_RETRACT_WAIT)
        self.beam_profile_device.retract()
        time.sleep(_WIRE_RETRACT_WAIT)

        wire_position = self.beam_profile_device.motor_rbv
        self.logger.info(
            "Wire retraction command issued. Motor position: %s",
            wire_position,
        )

        self.logger.info("Waiting for buffer acquisition to complete...")

        while not self.buffer.is_complete():
            elapsed_s = time.monotonic() - acquisition_start
            if elapsed_s > acquisition_timeout_s:
                raise TimeoutError(
                    f"Timing buffer {self.buffer.number} did not complete after "
                    f"{elapsed_s:.1f}s (timeout={acquisition_timeout_s:.1f}s)."
                )
            time.sleep(0.1)

        self.logger.info(
            "Timing buffer %s acquisition complete after %.1f seconds",
            self.buffer.number,
            time.monotonic() - acquisition_start,
        )
