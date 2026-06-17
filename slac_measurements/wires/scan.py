from slac_devices.wire import Wire
import slac_measurements.beam_profile
from slac_measurements.wires.collection import ScanMode, create_wire_collection
from slac_measurements.wires.analysis import FittingMethod, WireMeasurementAnalysis
from slac_measurements.wires.analysis_results import (
    WireMeasurementAnalysisResult,
)
from slac_measurements.wires.collection_results import (
    WireMeasurementCollectionResult,
)
from typing import Optional


class WireBeamProfileMeasurement(slac_measurements.beam_profile.BeamProfileMeasurement):
    """Full wire scan: collect and analyze beam profile data."""

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str
    collection_result: Optional[WireMeasurementCollectionResult] = None

    def measure(
        self,
        scan_mode: ScanMode = "otf",
        fitting_method: FittingMethod = "gaussian",
        rms_detector: Optional[str] = None,
        jitter_correction: bool = False,
    ) -> WireMeasurementAnalysisResult:
        """
        Run a wire scan and return the analyzed result.

        Parameters
        ----------
        scan_mode : "otf" (default) or "step".
        fitting_method : "gaussian" (default), "asymmetric_gaussian", or "super_gaussian".
        rms_detector : Detector for RMS sizes; defaults to the collection metadata default.
        jitter_correction : If True, apply orbit-fit jitter correction during analysis.
            Requires jitter_correction_bpms defined in wire metadata.
        """

        collection = create_wire_collection(
            scan_mode=scan_mode,
            beam_profile_device=self.beam_profile_device,
            beampath=self.beampath,
        )
        self.collection_result = collection.measure()

        analysis = WireMeasurementAnalysis(
            collection_result=self.collection_result,
            fitting_method=fitting_method,
        )
        return analysis.analyze(
            rms_detector=rms_detector,
            jitter_correction=jitter_correction,
        )
