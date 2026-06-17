import numpy as np
from datetime import datetime
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from slac_measurements.wires.analysis import WireMeasurementAnalysis
from slac_measurements.wires.analysis_results import (
    DetectorFit,
    DetectorProfileMeasurement,
    FitResult,
    ProfileMeasurement,
    WireMeasurementAnalysisResult,
    load_from_h5,
)
from slac_measurements.wires.collection_results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)


def _make_detector_fit(
    sigma: float,
    *,
    mean: float = 0.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    curve: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> DetectorFit:
    return DetectorFit(
        mean=mean,
        sigma=sigma,
        amplitude=amplitude,
        offset=offset,
        curve=np.array([1.0, 2.0]) if curve is None else curve,
        positions=np.array([0.0, 1.0]) if positions is None else positions,
    )


def _make_fit_result(
    sigma_map: dict[str, dict[str, float]],
    *,
    curve: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> dict[str, FitResult]:
    return {
        profile: FitResult(
            detectors={
                detector: _make_detector_fit(
                    sigma,
                    curve=curve,
                    positions=positions,
                )
                for detector, sigma in detector_sigmas.items()
            }
        )
        for profile, detector_sigmas in sigma_map.items()
    }


class TestGetMonotonicIndices(TestCase):
    """Tests for the _get_monotonic_indices static method logic."""

    def test_monotonic_indices_returns_all_indices_if_already_monotonic(self):
        """Test that fully monotonic data returns all indices unchanged."""
        position_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = np.array([0, 1, 2, 3, 4])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        np.testing.assert_array_equal(result, indices)

    def test_monotonic_indices_skips_initial_wobble(self):
        """Test that initial encoder wobble is skipped until monotonic behavior begins."""
        # Wobble: 5.2, 4.8, 5.1, then stable from 4.9: 4.9, 5.0, 5.5, 6.0, 6.5, 7.0
        position_data = np.array([5.2, 4.8, 5.1, 4.9, 5.0, 5.5, 6.0, 6.5, 7.0])
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # Leading trough at run boundary is dropped.
        expected = np.array([4, 5, 6, 7, 8])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_with_flat_region_then_wobble(self):
        """Test handling when encoder settles with flat region before continuing."""
        # Data: flat region [5.0, 5.0, 5.0], then increases [5.1, 5.2, 5.3]
        position_data = np.array([5.0, 5.0, 5.0, 5.1, 5.2, 5.3])
        indices = np.array([0, 1, 2, 3, 4, 5])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # All indices should be included (flat is monotonically non-decreasing)
        expected = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_with_severe_wobble(self):
        """Test handling of more severe initial wobble (like real encoder noise)."""
        # Severe wobble at start: 10.002, 9.998, 10.001, 9.999, then stable: 10.0, 10.1, 10.2, 10.3, 10.4
        position_data = np.array(
            [10.002, 9.998, 10.001, 9.999, 10.0, 10.1, 10.2, 10.3, 10.4]
        )
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # Leading trough at run boundary is dropped.
        expected = np.array([4, 5, 6, 7, 8])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_single_point(self):
        """Test that single point returns itself."""
        position_data = np.array([5.0])
        indices = np.array([0])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        np.testing.assert_array_equal(result, indices)

    def test_monotonic_indices_empty_array(self):
        """Test that empty array returns empty array."""
        position_data = np.array([])
        indices = np.array([], dtype=int)

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        np.testing.assert_array_equal(result, indices)

    def test_monotonic_indices_two_points_increasing(self):
        """Test that two increasing points returns both."""
        position_data = np.array([5.0, 6.0])
        indices = np.array([0, 1])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        np.testing.assert_array_equal(result, indices)

    def test_monotonic_indices_two_points_decreasing(self):
        """Test that two decreasing points returns one point (tie resolved to later run)."""
        position_data = np.array([6.0, 5.0])
        indices = np.array([0, 1])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # Two runs of length 1 tie; implementation keeps the later run.
        expected = np.array([1])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_subset_of_full_array(self):
        """Test filtering when indices are a subset of full position array."""
        # Full array with many points
        position_data = np.array([0, 1, 2, 3, 4, 5.2, 4.8, 5.1, 4.9, 5.0, 5.5, 6.0])
        # Only considering indices 5-11 (the wobble region and beyond)
        indices = np.array([5, 6, 7, 8, 9, 10, 11])
        # Corresponding positions: [5.2, 4.8, 5.1, 4.9, 5.0, 5.5, 6.0]

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # Leading trough at run boundary is dropped.
        expected = np.array([9, 10, 11])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_all_decreasing(self):
        """Test behavior when all data is decreasing."""
        position_data = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        indices = np.array([0, 1, 2, 3, 4])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # All runs have length 1; implementation keeps the later run on ties.
        expected = np.array([4])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_with_flat_entire_sequence(self):
        """Test that flat (equal values) sequence is considered monotonic."""
        position_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        indices = np.array([0, 1, 2, 3, 4])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # All should be returned (equal values are monotonic non-decreasing)
        np.testing.assert_array_equal(result, indices)

    def test_monotonic_indices_mixed_wobble_and_plateau(self):
        """Test wobble followed by plateau followed by increase."""
        # Wobble: 5.2, 4.8, then plateau and increase: 5.0, 5.0, 5.0, 5.1, 5.2
        position_data = np.array([5.2, 4.8, 5.0, 5.0, 5.0, 5.1, 5.2])
        indices = np.array([0, 1, 2, 3, 4, 5, 6])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        # Leading trough at run boundary is dropped.
        expected = np.array([2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_selects_longest_segment(self):
        """Select the largest contiguous non-decreasing section."""
        position_data = np.array(
            [
                30000,
                30003,
                30002,
                30003,
                30002,
                30001,
                30010,
                30050,
                30100,
                30160,
                30200,
            ]
        )
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        expected = np.array([6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, expected)

    def test_monotonic_indices_with_late_small_reversal_chooses_longest_run(self):
        """A late reversal should split the run; longest segment is retained."""
        position_data = np.array([28900.0, 28910.0, 28920.0, 28919.0, 28930.0])
        indices = np.array([0, 1, 2, 3, 4])

        result = WireMeasurementAnalysis._get_monotonic_indices(position_data, indices)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)


class TestWireMeasurementAnalysisOtherMethods(TestCase):
    """Tests for additional helper methods in WireMeasurementAnalysis."""

    @staticmethod
    def _make_analysis(
        *,
        raw_data: dict[str, np.ndarray] | None = None,
        detectors: list[str] | None = None,
        active_profiles: list[str] | None = None,
        scan_ranges: dict[str, tuple[int, int]] | None = None,
        wire_name: str = "WIRE",
        default_detector: str = "D1",
    ) -> WireMeasurementAnalysis:
        metadata = MeasurementMetadata(
            wire_name=wire_name,
            area="TEST",
            beampath="TEST",
            detectors=detectors if detectors is not None else ["D1"],
            default_detector=default_detector,
            scan_ranges=scan_ranges if scan_ranges is not None else {"x": (0, 1)},
            timestamp=datetime.now(),
            active_profiles=active_profiles if active_profiles is not None else ["x"],
            install_angle=45.0,
        )
        collection_result = WireMeasurementCollectionResult(
            raw_data=raw_data
            if raw_data is not None
            else {wire_name: np.array([0.0, 1.0])},
            metadata=metadata,
        )

        return WireMeasurementAnalysis(collection_result=collection_result)

    def test_create_detector_measurement_uses_tmitloss_units(self):
        analysis = self._make_analysis()

        result = analysis._create_detector_measurement("TMITLOSS", np.array([1.0, 2.0]))

        self.assertEqual(result.units, "% beam loss")
        self.assertEqual(result.label, "TMITLOSS")
        np.testing.assert_array_equal(result.values, np.array([1.0, 2.0]))

    def test_create_detector_measurement_uses_counts_for_other_detectors(self):
        analysis = self._make_analysis()

        result = analysis._create_detector_measurement("D1", np.array([3.0, 4.0]))

        self.assertEqual(result.units, "counts")
        self.assertEqual(result.label, "D1")

    def test_get_profile_range_indices_raises_when_position_is_constant(self):
        analysis = self._make_analysis(
            raw_data={"WIRE": np.array([5.0, 5.0, 5.0])},
            active_profiles=["x"],
            scan_ranges={"x": (4, 6)},
        )

        with self.assertRaises(RuntimeError):
            analysis._get_profile_range_indices()

    def test_get_profile_range_indices_raises_when_scan_never_reaches_profile(self):
        analysis = self._make_analysis(
            raw_data={"WIRE": np.array([0.0, 1.0, 2.0, 3.0])},
            active_profiles=["x"],
            scan_ranges={"x": (10, 20)},
        )

        with self.assertRaises(RuntimeError):
            analysis._get_profile_range_indices()

    def test_get_profile_range_indices_filters_to_monotonic_run(self):
        position_data = np.array([5.2, 4.8, 5.1, 4.9, 5.0, 5.5, 6.0, 6.5, 7.0])
        analysis = self._make_analysis(
            raw_data={"WIRE": position_data},
            active_profiles=["x"],
            scan_ranges={"x": (4, 7)},
        )

        result = analysis._get_profile_range_indices()

        np.testing.assert_array_equal(result["x"], np.array([4, 5, 6, 7, 8]))

    def test_organize_data_by_profile_builds_positions_and_detector_slices(self):
        raw_data = {
            "WIRE": np.array([10.0, 11.0, 12.0]),
            "D1": np.array([100.0, 101.0, 102.0]),
            "TMITLOSS": np.array([1.0, 2.0, 3.0]),
        }
        analysis = self._make_analysis(
            raw_data=raw_data,
            detectors=["D1", "TMITLOSS", "MISSING"],
        )

        profile_indices = {"x": np.array([0, 2])}
        result = analysis._organize_data_by_profile(profile_indices)

        x_profile = result["x"]
        np.testing.assert_array_equal(x_profile.positions, np.array([10.0, 12.0]))
        self.assertIn("D1", x_profile.detectors)
        self.assertIn("TMITLOSS", x_profile.detectors)
        self.assertNotIn("MISSING", x_profile.detectors)
        np.testing.assert_array_equal(
            x_profile.detectors["D1"].values, np.array([100.0, 102.0])
        )
        np.testing.assert_array_equal(
            x_profile.detectors["TMITLOSS"].values, np.array([1.0, 3.0])
        )

    def test_get_rms_sizes_returns_expected_detector_sigmas(self):
        analysis = self._make_analysis(default_detector="D1")
        fit_result = _make_fit_result(
            {
                "x": {"D1": 1.25},
                "y": {"D1": 2.5},
            }
        )

        x_rms, y_rms = analysis._get_rms_sizes(fit_result)

        self.assertEqual(x_rms, 1.25)
        self.assertEqual(y_rms, 2.5)

    def test_get_rms_sizes_uses_requested_detector_override(self):
        analysis = self._make_analysis(
            default_detector="D1",
            detectors=["D1", "D2"],
        )
        fit_result = _make_fit_result(
            {
                "x": {"D1": 1.25, "D2": 3.5},
                "y": {"D1": 2.5, "D2": 4.5},
            }
        )

        x_rms, y_rms = analysis._get_rms_sizes(fit_result, detector="D2")

        self.assertEqual(x_rms, 3.5)
        self.assertEqual(y_rms, 4.5)

    def test_get_rms_sizes_handles_missing_profiles(self):
        analysis = self._make_analysis(default_detector="D1")
        fit_result = _make_fit_result({"x": {"D1": 1.1}})

        x_rms, y_rms = analysis._get_rms_sizes(fit_result)

        self.assertEqual(x_rms, 1.1)
        self.assertIsNone(y_rms)

    def test_get_rms_sizes_raises_for_unknown_detector(self):
        analysis = self._make_analysis(detectors=["D1", "D2"])

        with self.assertRaises(ValueError):
            analysis._get_rms_sizes({}, detector="D3")

    def test_analyze_orchestrates_helper_methods_and_returns_result(self):
        analysis = self._make_analysis()
        expected_indices = {"x": np.array([0, 1])}
        expected_profiles = {
            "x": ProfileMeasurement(
                positions=np.array([0.0, 1.0]),
                profile_indices=np.array([0, 1]),
                detectors={
                    "D1": DetectorProfileMeasurement(
                        values=np.array([10.0, 20.0]),
                        units="counts",
                        label="D1",
                    )
                },
            )
        }
        expected_fit_result = {
            "x": FitResult(
                detectors={"D1": _make_detector_fit(1.0, mean=0.5, amplitude=5.0)}
            )
        }

        with (
            patch.object(
                analysis, "_get_profile_range_indices", return_value=expected_indices
            ) as mock_idx,
            patch.object(
                analysis,
                "_organize_data_by_profile",
                return_value=expected_profiles,
            ) as mock_org,
            patch.object(
                analysis,
                "_fit_data_by_profile",
                return_value=expected_fit_result,
            ) as mock_fit,
            patch.object(
                analysis, "_get_rms_sizes", return_value=(1.0, 2.0)
            ) as mock_rms,
        ):
            result = analysis.analyze(rms_detector="D1")

        mock_idx.assert_called_once_with()
        mock_org.assert_called_once_with(expected_indices)
        mock_fit.assert_called_once_with(profile_measurements=expected_profiles)
        mock_rms.assert_called_once_with(expected_fit_result, detector="D1")
        self.assertEqual(set(result.fit_result.keys()), {"x"})
        self.assertEqual(result.fit_result["x"].detectors["D1"].sigma, 1.0)
        np.testing.assert_array_equal(
            np.asarray(result.rms_sizes), np.array([1.0, 2.0])
        )
        self.assertEqual(set(result.profiles.keys()), {"x"})


class TestWireMeasurementAnalysisResult(TestCase):
    @staticmethod
    def _make_result() -> WireMeasurementAnalysisResult:
        metadata = MeasurementMetadata(
            wire_name="WIRE",
            area="TEST",
            beampath="TEST",
            detectors=["D1", "D2"],
            default_detector="D1",
            scan_ranges={"x": (0, 1), "y": (0, 1)},
            timestamp=datetime.now(),
            active_profiles=["x", "y"],
            install_angle=45.0,
        )
        collection_result = WireMeasurementCollectionResult(
            raw_data={"WIRE": np.array([0.0, 1.0])},
            metadata=metadata,
        )
        fit_result = _make_fit_result(
            {
                "x": {"D1": 1.25, "D2": 3.5},
                "y": {"D1": 2.5, "D2": 4.5},
            }
        )

        return WireMeasurementAnalysisResult(
            fit_result=fit_result,
            collection_result=collection_result,
            profiles={},
            rms_sizes=np.array([1.25, 2.5]),
            metadata=metadata,
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._result_template = cls._make_result()

    def setUp(self):
        # Use a deep copy of one shared template so tests remain isolated.
        self.result = self._result_template.model_copy(deep=True)

    def test_save_to_h5_handles_object_rms_sizes(self):
        self.result.rms_sizes = np.array([1.25, None], dtype=object)

        with TemporaryDirectory() as tmpdir:
            outpath = f"{tmpdir}/analysis_result.h5"
            self.result.save_to_h5(outpath)
            loaded = load_from_h5(outpath)

        self.assertEqual(tuple(loaded.rms_sizes), (1.25, None))
