import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

from typing_extensions import Self

from pydantic import model_validator
from slac_devices.wire import Wire
from slac_timing import Buffer
import slac_measurements.beam_profile
import slac_measurements.wires.collection.buffer
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)

_LOG_DIR = Path("/u1/lcls/physics/data/wire_scan/logs")
_LOGGER_NAME = "wire_scan_logger"
_ACQUISITION_TIMEOUT_MARGIN = 1.25
_ACQUISITION_TIMEOUT_MIN_EXTRA_S = 10.0
ScanMode = Literal["step", "otf"]


class BaseWireMeasurementCollection(
    slac_measurements.beam_profile.BeamProfileMeasurement,
    ABC,
):
    """
    Collects wire scan measurement data via motor motion and timing buffer.
    Raw data is returned for downstream analysis.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier for buffer and device selection.
        buffer: Timing buffer managing data acquisition.
        devices (dict): Device objects (wire, detectors) used in the scan.
        data (dict): Raw buffered data by device name.
        logger (logging.Logger): File-based measurement logger.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str
    buffer: Buffer | None = None
    devices: dict | None = None
    detectors: list | None = None
    data: dict | None = None
    logger: logging.Logger | None = None
    metadata: MeasurementMetadata | None = None

    def measure(self) -> WireMeasurementCollectionResult:
        """
        Execute wire scan: run mode-specific wire motion and acquire detector
        data from timing buffer.

        Two scan modes are supported:
            - ``otf`` : On-the-fly scan using the wire's built-in start_scan command and
              collect data while the wire moves continuously.
            - ``step`` : perform a discrete (step) scan by moving the motor to each
              inner/outer position in sequence.

        Returns
        -------
        WireMeasurementCollectionResult
            Raw data and metadata, including:
            - raw_data: Buffered position and detector values by device name
            - metadata: Timestamp, wire name, area, beampath, and detector list
        """

        def _prepare_runtime_state() -> None:
            """Prepare per-run state that depends on an active timing buffer."""

            self.buffer = self._reserve_buffer()
            self.devices = self._create_device_dictionary()
            self.metadata = self._create_metadata()

        def _release_buffer_safely() -> None:
            """Release timing buffer after scan completion."""

            buf = self.buffer
            if buf is not None:
                buffer_number = getattr(buf, "number", None)
                try:
                    self.logger.info("Releasing timing buffer %s.", buffer_number)
                    buf.release()
                except Exception:
                    self.logger.exception(
                        "Failed while releasing timing buffer %s.", buffer_number
                    )
                finally:
                    self.buffer = None

        try:
            _prepare_runtime_state()
            self._run_collection_scan()
            self.data = self._get_data_from_buffer()
            self.metadata.timestamp = datetime.now()
        finally:
            _release_buffer_safely()

        return WireMeasurementCollectionResult(
            raw_data=self.data,
            metadata=self.metadata,
        )

    def _create_device_dictionary(self) -> dict:
        """Create dictionary of required devices. Includes the wire device and detectors."""

        def _instantiate_device(name: str, area: str):
            """Instantiate a single device by name and area."""

            import slac_devices.reader
            import slac_measurements.tmit_loss

            if name == "TMITLOSS":
                return slac_measurements.tmit_loss.TMITLoss(
                    buffer=self.buffer,
                    beam_profile_device=self.beam_profile_device,
                    beampath=self.beampath,
                )

            create_by_prefix = {
                "LBLM": slac_devices.reader.create_lblm,
                "PMT": slac_devices.reader.create_pmt,
                "BPM": slac_devices.reader.create_bpm,
            }

            creator = next(
                (
                    f
                    for prefix, f in create_by_prefix.items()
                    if name.startswith(prefix)
                ),
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

        devices = {self.beam_profile_device.name: self.beam_profile_device}

        for ds in self.beam_profile_device.metadata.detectors:
            name, area = ds.split(":")
            detector = _instantiate_device(name, area)
            if detector is not None:
                devices[name] = detector

        # Add jitter correction BPMs if defined
        jitter_bpm_names = self.beam_profile_device.metadata.jitter_bpms
        if jitter_bpm_names:
            area = self.beam_profile_device.area
            for name in jitter_bpm_names:
                bpm = _instantiate_device(name, area)
                if bpm is not None:
                    devices[name] = bpm

        self.logger.info("Device dictionary built.")
        return devices

    def _create_metadata(self) -> MeasurementMetadata:
        """Create per-run metadata for the current scan."""

        def _get_default_detector() -> str:
            """Determine the default detector for analysis from wire metadata or device list."""

            default_detector = self.beam_profile_device.metadata.default_detector

            if not default_detector:
                if not self.detectors:
                    msg = (
                        "No detectors available from wire metadata; "
                        "cannot determine default detector."
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                return self.detectors[0]

            return default_detector.split(":", 1)[0]

        def _get_scan_ranges() -> dict:
            """Return dictionary of scan ranges for x, y, and u motors."""

            return {
                "x": self.beam_profile_device.x_range,
                "y": self.beam_profile_device.y_range,
                "u": self.beam_profile_device.u_range,
            }

        return MeasurementMetadata(
            wire_name=self.beam_profile_device.name,
            buffer_number=self.buffer.number,
            area=self.beam_profile_device.area,
            beampath=self.beampath,
            detectors=self.detectors,
            default_detector=_get_default_detector(),
            scan_ranges=_get_scan_ranges(),
            timestamp=None,
            active_profiles=self.beam_profile_device.active_profiles(),
            install_angle=self.beam_profile_device.install_angle,
            notes=None,
        )

    def _get_data_from_buffer(self) -> dict:
        """Collects wire scan and detector data after buffer completes."""

        def _get_buffer_collection_method(device_name: str) -> str | None:
            """Determine the buffer collection method for a given device based on its name."""

            if device_name == self.beam_profile_device.name:
                return "position_buffer"
            elif device_name.startswith("LBLM"):
                return "fast_buffer"
            elif device_name.startswith("PMT"):
                return "qdcraw_buffer"
            elif device_name.startswith("BPM"):
                return "bpm_buffer"
            else:
                return None

        def _collect_device_data(device_name: str):
            """Collect data for a given device."""

            device = self.devices[device_name]
            buffer_method = _get_buffer_collection_method(device_name)

            if buffer_method is None:
                return device.measure()

            if buffer_method == "bpm_buffer":
                return {
                    "x": device.x_buffer(self.buffer, retries=3, retry_delay=3.0),
                    "y": device.y_buffer(self.buffer, retries=3, retry_delay=3.0),
                }

            return getattr(device, buffer_method)(
                self.buffer, retries=3, retry_delay=3.0
            )

        self.logger.info("Getting data from timing buffer ...")
        data = {name: _collect_device_data(name) for name in self.devices.keys()}
        self.logger.info("Data retrieved from buffer. Scan complete.")
        return data

    def _reserve_buffer(self) -> object:
        """Reserve a timing buffer for the scan based on beampath and wire metadata."""

        if self.buffer is None:
            self.buffer = slac_measurements.wires.collection.buffer.reserve_buffer(
                beampath=self.beampath,
                logger=self.logger,
                pulses=self.beam_profile_device.scan_pulses,
                beam_rate=self.beam_profile_device.beam_rate,
            )

        return self.buffer

    def _calculate_acquisition_timeout_s(self) -> float:
        """Return timeout above minimum expected buffer acquisition time."""

        n_points = getattr(self.buffer, "n_measurements", None)
        if n_points is None or n_points <= 0:
            raise RuntimeError(
                f"Invalid buffer point count for timeout calculation: {n_points}"
            )

        min_expected_s = n_points / self.beam_profile_device.beam_rate
        return max(
            min_expected_s * _ACQUISITION_TIMEOUT_MARGIN,
            min_expected_s + _ACQUISITION_TIMEOUT_MIN_EXTRA_S,
        )

    @abstractmethod
    def _run_collection_scan(self) -> None:
        """Run mode-specific wire motion and buffer timing behavior."""

    @model_validator(mode="after")
    def _run_setup(self) -> Self:
        """Initialize construction-time state for a collection instance."""

        import slac_measurements.logger.file_logger

        if not _LOG_DIR.exists():
            raise FileNotFoundError(
                f"Log directory does not exist: {_LOG_DIR}. "
                "Create it or configure a valid existing path."
            )

        # Configure logger — compute filename now so long-running processes
        # get a fresh date-stamped file rather than the one frozen at import.
        log_filepath = _LOG_DIR / f"ws_log_{datetime.now().strftime('%Y%m%d')}.txt"
        self.logger = slac_measurements.logger.file_logger.custom_logger(
            log_file=str(log_filepath),
            name=_LOGGER_NAME,
        )
        self.logger.propagate = False

        # Get list of detector names from wire metadata
        self.detectors = [
            d.split(":")[0] for d in self.beam_profile_device.metadata.detectors
        ]
        return self


def create_wire_collection(
    *,
    scan_mode: ScanMode,
    beam_profile_device: Wire,
    beampath: str,
) -> BaseWireMeasurementCollection:
    """Instantiate the mode-specific wire collection class."""

    if scan_mode == "step":
        from slac_measurements.wires.collection.step import (
            StepWireMeasurementCollection,
        )

        return StepWireMeasurementCollection(
            beam_profile_device=beam_profile_device,
            beampath=beampath,
        )

    if scan_mode == "otf":
        from slac_measurements.wires.collection.otf import OTFWireMeasurementCollection

        return OTFWireMeasurementCollection(
            beam_profile_device=beam_profile_device,
            beampath=beampath,
        )

    raise ValueError(f"Unknown scan_mode '{scan_mode}'. Expected 'step' or 'otf'.")
