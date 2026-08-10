"""OTF wire collection: buffered data acquisition with continuous motion."""

import time

from slac_measurements.wires.collection.base import BaseWireMeasurementCollection
from slac_measurements.wires.motion.otf import initialize_otf_with_retry


class OTFWireMeasurementCollection(BaseWireMeasurementCollection):
    """Collect wire scan data using on-the-fly wire motion."""

    def _run_collection_scan(self) -> None:
        """Run an OTF scan: init wire, start buffer."""

        def _start_timing_buffer() -> None:
            """Start BSA buffer and wait for completion while logging wire position."""

            self.logger.info("Starting buffer acquisition for on-the-fly scan...")
            acquisition_start = time.monotonic()
            acquisition_timeout_s = self._calculate_acquisition_timeout_s()
            self.buffer.start()

            time.sleep(0.5)

            i = 0
            while not self.buffer.is_complete():
                elapsed_s = time.monotonic() - acquisition_start
                if elapsed_s > acquisition_timeout_s:
                    raise TimeoutError(
                        f"Timing buffer {self.buffer.number} did not complete after "
                        f"{elapsed_s:.1f}s (timeout={acquisition_timeout_s:.1f}s)."
                    )
                time.sleep(0.1)
                if i % 10 == 0:
                    wire_position = self.beam_profile_device.motor_rbv
                    self.logger.info("Wire position: %s", wire_position)
                i += 1

            self.logger.info(
                "Timing buffer %s acquisition complete after %.1f seconds",
                self.buffer.number,
                time.monotonic() - acquisition_start,
            )

        self.logger.info("Performing on-the-fly scan mode")
        initialize_otf_with_retry(self.beam_profile_device, self.logger, max_attempts=3)
        _start_timing_buffer()
