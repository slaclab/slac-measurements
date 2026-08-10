from unittest import TestCase
from unittest.mock import patch
from datetime import datetime
from tempfile import TemporaryDirectory
import numpy as np

from slac_devices.wire import Wire
from slac_measurements.wires.scan import WireBeamProfileMeasurement
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
    load_from_h5,
)


class WireBeamProfileMeasurementTest(TestCase):
    @staticmethod
    def _make_wire_device() -> Wire:
        # Create an instance without invoking device initialization side effects.
        return Wire.__new__(Wire)

    @staticmethod
    def _make_collection_result() -> WireMeasurementCollectionResult:
        metadata = MeasurementMetadata(
            wire_name="WIRE",
            buffer_number=7,
            area="TEST",
            beampath="TEST",
            detectors=["D1"],
            default_detector="D1",
            scan_ranges={"x": (0, 1)},
            timestamp=datetime.now(),
            active_profiles=["x"],
            install_angle=45.0,
        )
        return WireMeasurementCollectionResult(
            raw_data={"WIRE": np.array([0.0, 1.0])},
            metadata=metadata,
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    @patch("slac_measurements.wires.scan.create_wire_collection")
    def test_measure_uses_configured_fitting_method(
        self,
        mock_create_collection,
        mock_analysis_cls,
    ):
        mock_collection = mock_create_collection.return_value
        mock_collection.measure.return_value = "collection-result"

        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        result = measurement.measure(fitting_method="super_gaussian")

        self.assertEqual(result, "analysis-result")
        self.assertEqual(measurement.collection_result, "collection-result")
        mock_create_collection.assert_called_once()
        self.assertEqual(
            mock_create_collection.call_args.kwargs["scan_mode"],
            "otf",
        )
        self.assertIs(
            mock_create_collection.call_args.kwargs["beam_profile_device"],
            measurement.beam_profile_device,
        )
        self.assertEqual(
            mock_create_collection.call_args.kwargs["beampath"],
            "TEST",
        )
        mock_collection.measure.assert_called_once_with()
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="super_gaussian",
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    @patch("slac_measurements.wires.scan.create_wire_collection")
    def test_measure_allows_fitting_method_override(
        self,
        mock_create_collection,
        mock_analysis_cls,
    ):
        mock_collection = mock_create_collection.return_value
        mock_collection.measure.return_value = "collection-result"

        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        result = measurement.measure(fitting_method="asymmetric_gaussian")

        self.assertEqual(result, "analysis-result")
        mock_collection.measure.assert_called_once_with()
        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="asymmetric_gaussian",
        )

    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    def test_analysis_class_can_refit_collection_result(self, mock_analysis_cls):
        mock_analysis = mock_analysis_cls.return_value
        mock_analysis.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )
        measurement.collection_result = self._make_collection_result()

        analysis = mock_analysis_cls(
            collection_result=measurement.collection_result,
            fitting_method="super_gaussian",
        )
        result = analysis.analyze(rms_detector=None)

        self.assertEqual(result, "analysis-result")
        mock_analysis_cls.assert_called_once_with(
            collection_result=measurement.collection_result,
            fitting_method="super_gaussian",
        )
        mock_analysis.analyze.assert_called_once_with(rms_detector=None)

    # measure() constructs a mode-specific collection via factory, so we patch
    # it to keep this unit test isolated from collection setup behavior.
    @patch("slac_measurements.wires.scan.WireMeasurementAnalysis")
    @patch("slac_measurements.wires.scan.create_wire_collection")
    def test_measure_passes_rms_detector_override(
        self,
        mock_create_collection,
        mock_analysis_cls,
    ):
        mock_create_collection.return_value.measure.return_value = "collection-result"
        mock_analysis_cls.return_value.analyze.return_value = "analysis-result"

        measurement = WireBeamProfileMeasurement(
            beam_profile_device=self._make_wire_device(),
            beampath="TEST",
        )

        measurement.measure(
            fitting_method="gaussian",
            rms_detector="D2",
        )

        mock_analysis_cls.assert_called_once_with(
            collection_result="collection-result",
            fitting_method="gaussian",
        )
        mock_analysis_cls.return_value.analyze.assert_called_once_with(
            rms_detector="D2",
            jitter_correction=False,
        )

    def test_collection_result_round_trip_preserves_buffer_number(self):
        collection_result = self._make_collection_result()

        with TemporaryDirectory() as tmpdir:
            outpath = f"{tmpdir}/collection_result.h5"
            collection_result.save_to_h5(outpath)
            loaded = load_from_h5(outpath)

        self.assertEqual(loaded.metadata.buffer_number, 7)
