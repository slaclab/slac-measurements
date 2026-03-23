import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from slac_devices.wire import Wire
import slac_measurements.beam_profile
import slac_measurements.utils
from slac_measurements.ws_collection_results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)

_DATE = datetime.now().strftime("%Y%m%d")
_LOG_FILENAME = f"ws_log_{_DATE}.txt"
_LOGGER_NAME = "wire_scan_logger"
_WIRE_LBLMS_LOCATION = Path(__file__).resolve().parent.parent / "devices" / "yaml" / "wire_lblms.yaml"
_WIRE_TOLERANCE = 250  # microns


class WireMeasurementCollection(slac_measurements.beam_profile.BeamProfileMeasurement):
    """
    Collects wire scan measurement data via motor motion and BSA buffer.

    Moves the wire and acquires synchronized detector data without organizing
    or fitting. Raw data is returned for downstream analysis.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier for buffer and device selection.
        my_buffer: BSA buffer managing data acquisition.
        devices (dict): Device objects (wire, detectors) used in the scan.
        data (dict): Raw buffered data by device name.
        logger (logging.Logger): File-based measurement logger.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str
    my_buffer: Optional[object] = None
    devices: Optional[dict] = None
    detectors: Optional[list] = None
    data: Optional[dict] = None
    logger: Optional[logging.Logger] = None

    # alias so beam_profile_device can also be accessed with name my_wire
    @property
    def my_wire(self) -> Wire:
        return self.beam_profile_device

    @my_wire.setter
    def my_wire(self, value):
        self.beam_profile_device = value

    def measure(self, scan_type: str = "step") -> WireMeasurementCollectionResult:
        """
        Execute wire scan: move wire, acquire detector data from BSA buffer.

        Two scan modes are supported:
        - ``on_the_fly`` : use the wire's built-in start_scan command and
          collect data while the wire moves continuously.
        - ``step`` : perform a discrete (step) scan by moving the motor to each
          inner/outer position in sequence with the buffer acquiring the whole
          time.

        The desired mode can be selected by passing ``scan_type``.

        Parameters
        ----------
        scan_type : str, optional
            ``"on_the_fly"`` or ``"step"``.  Any other value will raise a
            ``ValueError``.

        Returns
        -------
        WireMeasurementCollectionResult
            Raw data and metadata, including:
            - raw_data: Buffered position and detector values by device name
            - metadata: Timestamp, wire name, area, beampath, and detector list
        """
        if scan_type not in ("on_the_fly", "step"):
            raise ValueError(
                f"Unknown scan_type '{scan_type}'. ``on_the_fly`` or ``step`` expected."
            )

        self.my_buffer = self._reserve_buffer()
        metadata = self._create_metadata()
        self._scan_with_wire(scan_type=scan_type)

        # For on‑the‑fly scans we must start the timing buffer here; the step
        # implementation already handles the buffer start and wait internally.
        if scan_type == "on_the_fly":
            self._start_timing_buffer()

        # Get position and detector data from the buffer
        self.data = self._get_data_from_buffer()

        # Release EDEF/BSA
        self.logger.info("Releasing BSA buffer.")
        self.my_buffer.release()
        self.my_buffer = None

        # Turn off motor after scan only if retract was successful
        if self.my_wire.motor_rbv < 500:
            self.my_wire.torque_enable = 0

        return WireMeasurementCollectionResult(
            raw_data=self.data,
            metadata=metadata,
        )

    @model_validator(mode="after")
    def run_setup(self) -> Self:
        import slac_measurements.logger.file_logger
        # Configure  logger
        self.logger = slac_measurements.logger.file_logger.custom_logger(
            log_file=_LOG_FILENAME,
            name=_LOGGER_NAME,
        )
        self.logger.propagate = False

        # Reserve BSA buffer
        self.my_buffer = self._reserve_buffer()

        # Get list of detector names from wire metadata
        self.detectors = [d.split(":")[0] for d in self.my_wire.metadata.detectors]

        # Generate dictionary of all requried lcls-tools device objects
        self.devices = self.create_device_dictionary()
        return self

    def _create_device_dictionary(self) -> dict:
        """
        Creates a device dictionary for a wire scan setup.  Includes the wire
        device and any associated detectors from metadata.

        Returns:
            dict: A mapping of device names to device objects.
        """

        def _instantiate_device(name: str, area: str):
            """
            Instantiate a single device by name and area
            """
            import slac_devices.reader
            import slac_measurements.tmit_loss

            if name == "TMITLOSS":
                return slac_measurements.tmit_loss.TMITLoss(
                    my_buffer=self.my_buffer,
                    my_wire=self.my_wire,
                    beampath=self.beampath,
                    region=self.my_wire.area,
                )

            create_by_prefix = {
                "LBLM": slac_devices.reader.create_lblm,
                "PMT": slac_devices.reader.create_pmt,
            }

            creator = next(
                (f for prefix, f in create_by_prefix.items() if name.startswith(prefix)),
                None,
            )

            if creator is None:
                self.logger.warning("Unknown device type '%s'. Skipping.", name)
                return None

            device = creator(area=area, name=name)
            if device is None:
                self.logger.warning("Device creation for %s returned None.", name)

            return device

        self.logger.info("Creating device dictionary...")

        # Instantiate device dictionary with wire device
        devices = {self.my_wire.name: self.my_wire}

        # ds is a colon-separated detector string from metadata
        # e.g. "LBLM:TEST" -> name = "LBLM", area = "TEST"
        for ds in self.my_wire.metadata.detectors:
            name, area = ds.split(":")
            detector = _instantiate_device(name, area)
            if detector is not None:
                devices[name] = detector

        self.logger.info("Device dictionary built.")
        return devices

    def _create_metadata(self) -> MeasurementMetadata:
        """
        Make additional metadata.
        """
        def _load_yaml_config() -> Optional[dict]:
            import yaml

            file_to_open = _WIRE_LBLMS_LOCATION

            if file_to_open.exists() is False:
                msg = f"YAML config file {file_to_open} not found."
                self.logger.error(msg)
                return None

            with open(file_to_open, "r") as f:
                wire_lblms = yaml.safe_load(f)
                return wire_lblms

        def _get_default_detector() -> str:
            lblm_config = _load_yaml_config()
            if lblm_config is None:
                return self.detectors[0]
            else:
                default_detector = lblm_config[self.my_wire.name]
                return default_detector

        def _get_scan_ranges():
            return {
                "x": self.my_wire.x_range,
                "y": self.my_wire.y_range,
                "u": self.my_wire.u_range,
            }

        return MeasurementMetadata(
            wire_name=self.my_wire.name,
            buffer_number=self.my_buffer.number,
            area=self.my_wire.area,
            beampath=self.beampath,
            detectors=self.detectors,
            default_detector=_get_default_detector(),
            scan_ranges=_get_scan_ranges(),
            timestamp=datetime.now(),
            active_profiles=self.my_wire.active_profiles(),
            install_angle=self.my_wire.install_angle,
            notes=None,
        )

    def _get_data_from_buffer(self) -> dict:
        """
        Collects wire scan and detector data after buffer completes.

        Returns:
            dict: Collected data keyed by device name.
        """
        def _get_buffer_collection_method(device_name: str) -> Optional[str]:
            """
            Determine the buffer collection method for a given device based on its name.
            Returns None for devices that don't collect data this way (e.g., TMITLOSS).
            """
            if device_name == self.my_wire.name:
                return "position_buffer"
            elif device_name.startswith("LBLM"):
                return "fast_buffer"
            elif device_name.startswith("PMT"):
                return "qdcraw_buffer"
            else:
                return None

        def _collect_device_data(device_name: str) -> np.ndarray:
            """Collect data for a given device using the appropriate method."""
            import slac_measurements.utils

            device = self.devices[device_name]
            buffer_method = _get_buffer_collection_method(device_name)

            if buffer_method is None:
                return (
                    device.measure()
                )  # For devices like TMITLOSS that don't use buffer collection

            return slac_measurements.utils.collect_with_size_check(
                device, buffer_method, self.my_buffer, self.logger
            )

        self.logger.info("Getting data from BSA buffer...")
        data = {name: _collect_device_data(name) for name in self.devices.keys()}
        self.logger.info("Data retrieved from buffer. Scan complete.")
        return data

    def _initialize_wire_with_retry(self, wire_action: str, max_attempts: int = 3) -> None:
        """Call start_scan/initialize with retries until wire.enabled.

        wire_action must be 'start_scan' or 'initialize'; raises on failure.
        """
        if wire_action not in ("start_scan", "initialize"):
            raise ValueError(
                f"Unknown wire_action '{wire_action}'. Expected 'start_scan' or 'initialize'."
            )

        # Skip initialization if wire is already enabled
        if self.my_wire.enabled:
            self.logger.info(f"{self.my_wire.name} is already enabled.")
            return

        # Choose the appropriate method to call
        action_method = (
            self.my_wire.start_scan
            if wire_action == "start_scan"
            else self.my_wire.initialize
        )
        action_desc = (
            "for on-the-fly scan" if wire_action == "start_scan" else "for step scan"
        )

        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                f"Initializing {self.my_wire.name} {action_desc}: "
                f"(Attempt {attempt}/{max_attempts})..."
            )
            action_method()

            # If returns True within timeout, proceed
            if slac_measurements.utils.wait_until(lambda: self.my_wire.enabled):
                self.logger.info(f"{self.my_wire.name} initialized.")
                return

            # After timeout, log and iterate through for loop again
            else:
                self.logger.warning(
                    "%s did not enable - retrying...", self.my_wire.name
                )

        raise RuntimeError(
            f"Failed to initialize {self.my_wire.name} after {max_attempts} attempts."
        )

    def _perform_step_scan(self) -> None:
        """Run a step scan: init wire, start buffer, move positions, retract, wait."""

        def _move_to_step_position(
            self, position: int, position_index: int, total_positions: int
        ) -> None:
            """Move wire to a step position, waiting up to 15s or raise error."""
            self.logger.info(
                f"Moving wire to {position} (step {position_index + 1}/{total_positions})..."
            )

            # Set speed and move
            positions = _get_step_positions()
            speed = _calculate_step_speed(position_index, positions)
            self.my_wire.speed = speed
            self.my_wire.motor = position

            # Wait for position with 250 um tolerance
            if not slac_measurements.utils._wait_until(
                lambda: abs(self.my_wire.motor_rbv - position) < _WIRE_TOLERANCE,
            ):
                raise RuntimeError(
                    f"{self.my_wire.name} did not reach position {position} after 15s."
                )

        def _calculate_step_speed(position_index: int, positions: list) -> int:
            """Return speed for a step position: max for inner, computed for outer.

            Even indices use speed_max; odd indices use calc speed.
            """
            if position_index % 2 == 0:
                # inner position – use maximum speed
                return int(self.my_wire.speed_max)

            # outer position – calculate speed to span gap in one pulse train
            position_delta = positions[position_index] - positions[position_index - 1]
            speed = (position_delta / self.my_wire.scan_pulses) * self.my_wire.beam_rate
            return int(speed)

        def _get_step_positions() -> list:
            """Return sorted inner/outer positions for active profiles."""
            positions = []
            for profile in self.my_wire.active_profiles():
                for mode in ["inner", "outer"]:
                    attr_name = f"{profile}_wire_{mode}"
                    positions.append(getattr(self.my_wire, attr_name))
            return sorted(positions)

        # Start step scan sequence
        self.logger.info("Performing step scan mode")

        # Initialize wire for step scan (with retry logic, no continuous motion)
        self._initialize_wire_with_retry(wire_action="initialize")

        # Start buffer acquisition after successful wire initialization
        self.logger.info("Starting buffer acquisition for step scan...")
        self.my_buffer.start()

        # Get ordered positions and move to each
        positions = _get_step_positions()
        for i, position in enumerate(positions):
            _move_to_step_position(position, i, len(positions))

        # Retract wire
        self.logger.info("Retracting wire...")
        self.my_wire.speed = self.my_wire.speed_max
        self.my_wire.motor = 100

        # Wait for buffer acquisition to complete
        self.logger.info("Waiting for buffer acquisition to complete...")
        while not self.my_buffer.is_acquisition_complete():
            time.sleep(0.1)

    def _reserve_buffer(self) -> object:
        import slac_measurements.ws_buffer

        if self.my_buffer is None:
            return slac_measurements.ws_buffer.reserve_buffer(
                beampath=self.beampath,
                logger=self.logger,
                pulses=self.my_wire.scan_pulses,
                beam_rate=self.my_wire.beam_rate,
            )

    def _scan_with_wire(self, scan_type: str = "step") -> None:
        """
        Kick off motion for the wire and (optionally) the buffer.

        The behaviour depends on the requested ``scan_type``.  The default is
        ``step`` which drives the wire to each of the inner/outer positions one
        at a time while the buffer is already running.  The latter is useful for
        setups where the wire cannot use the
        built-in continuous scan command.

        Parameters
        ----------
        scan_type : str, optional
            ``"on_the_fly"`` or ``"step"``.  ``step`` behaviour is the
            historic default.
        """
        # Reserve a new buffer if necessary
        self.my_buffer = self._reserve_buffer()

        if scan_type == "on_the_fly":
            self._initialize_wire_with_retry(wire_action="start_scan")
        elif scan_type == "step":
            self._perform_step_scan()
        else:
            raise ValueError(f"Unsupported scan_type '{scan_type}'")

    def _start_timing_buffer(self) -> None:
        """
        Start a BSA buffer and wait for it to complete.  Post wire position to
        the log every second.
        """
        # Start buffer
        self.logger.info("Starting BSA buffer...")
        self.my_buffer.start()

        # Wait briefly before checking buffer 'ready'
        # Wire is already moving, data is already collecting...
        time.sleep(0.5)

        # Wait for buffer 'ready'
        i = 0
        while not self.my_buffer.is_acquisition_complete():
            # Check for completion every 0.1 s, post position 1s
            time.sleep(0.1)
            if i % 10 == 0:
                self.logger.info("Wire position: %s", self.my_wire.motor_rbv)
            i += 1

        self.logger.info(
            "BSA buffer %s acquisition complete after %s seconds",
            self.my_buffer.number,
            i / 10,
        )