from slac_devices.wire import Wire
import slac_measurements.beam_profile
from slac_measurements.wires.ws_collection import WireMeasurementCollection
from slac_measurements.wires.ws_analysis import WireMeasurementAnalysis
from slac_measurements.wires.ws_analysis_results import (
    WireMeasurementAnalysisResult,
)
from typing import Literal


class WireBeamProfileMeasurement(
    slac_measurements.beam_profile.BeamProfileMeasurement
):
    """
    Orchestrates a full wire scan: accepts a wire device and beampath,
    instantiates a WireMeasurementCollection, runs the scan, and returns
    the analyzed result.

    Attributes:
        beam_profile_device (Wire): Wire device for the scan.
        beampath (str): Beamline identifier passed to the collection.
    """

    name: str = "Wire Beam Profile Measurement"
    beam_profile_device: Wire
    beampath: str

    def measure(
        self,
        scan_type: str = "step",
        fit_method: Literal[
            "gaussian", "asymmetric_gaussian", "super_gaussian"
        ] = "gaussian"
    ) -> WireMeasurementAnalysisResult:
        """
        Instantiate a WireMeasurementCollection, run the scan, analyze, and
        return the result.

        Parameters
        ----------
        scan_type : str
            ``"on_the_fly"`` or ``"step"`` (default).
        fit_method : str
            Fit model used by the downstream wire-scan analysis. Supported
            values are ``"gaussian"``, ``"asymmetric_gaussian"``, and
            ``"super_gaussian"``. Defaults to the measurement instance's
            configured ``fit_method``.

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results, RMS beam sizes, and organized profile data.
        """
        collection = WireMeasurementCollection(
            beam_profile_device=self.beam_profile_device,
            beampath=self.beampath,
        )
        collection_result = collection.measure(scan_type=scan_type)

        analysis = WireMeasurementAnalysis(
            collection_result=collection_result,
            fit_method=fit_method,
        )
        return analysis.analyze()
