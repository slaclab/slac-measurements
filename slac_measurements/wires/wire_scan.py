from slac_devices.wire import Wire
import slac_measurements.beam_profile
from slac_measurements.wires.ws_collection import WireMeasurementCollection
from slac_measurements.wires.ws_analysis import WireMeasurementAnalysis
from slac_measurements.wires.ws_analysis_results import (
    WireMeasurementAnalysisResult,
)
from slac_measurements.wires.ws_collection_results import (
    WireMeasurementCollectionResult,
)
from typing import Literal, Optional


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
    collection_result: Optional[WireMeasurementCollectionResult] = None

    def measure(
        self,
        scan_type: str = "step",
        fitting_method: Literal[
            "gaussian", "asymmetric_gaussian", "super_gaussian"
        ] = "gaussian",
    ) -> WireMeasurementAnalysisResult:
        """
        Instantiate a WireMeasurementCollection, run the scan, analyze, and
        return the result.

        Parameters
        ----------
        scan_type : str
            ``"on_the_fly"`` or ``"step"`` (default).
        fitting_method : str, optional
            Fit model used by the downstream wire-scan analysis. Supported
            values are ``"gaussian"``, ``"asymmetric_gaussian"``, and
            ``"super_gaussian"``.

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results, RMS beam sizes, and organized profile data.
        """
        collection = WireMeasurementCollection(
            beam_profile_device=self.beam_profile_device,
            beampath=self.beampath,
        )
        self.collection_result = collection.measure(scan_type=scan_type)
        return self.analyze(fitting_method=fitting_method)

    def analyze(self, fitting_method) -> WireMeasurementAnalysisResult:
        """
        Analyze the most recently collected wire-scan data.

        Parameters
        ----------
        fitting_method : str, optional
            Fit model used by wire-scan analysis. If omitted, uses the
            instance default ``self.fitting_method``.

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results, RMS beam sizes, and organized profile data.

        Raises
        ------
        RuntimeError
            If no collection data is available. Run ``measure()`` first.
        """
        if self.collection_result is None:
            msg = (
                "No collection_result available. "
                "Run measure() before analyze()."
            )
            raise RuntimeError(msg)

        analysis = WireMeasurementAnalysis(
            collection_result=self.collection_result,
            fitting_method=fitting_method,
        )
        return analysis.analyze()
