from unittest import TestCase

import numpy as np

from slac_measurements.wires.analysis.charge_normalization import (
    _extract_charge_data,
    _resolve_toroid,
    compute_charge_normalization,
)
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)


def _make_collection_result(raw_data, charge_toroids=None):
    metadata = MeasurementMetadata(
        wire_name="WS01",
        area="DL1",
        beampath="SC_HXR",
        detectors=["PMT1"],
        default_detector="PMT1",
        scan_ranges={"x": (0, 1000)},
        active_profiles=["x"],
        install_angle=45.0,
        charge_toroids=charge_toroids,
    )
    return WireMeasurementCollectionResult(raw_data=raw_data, metadata=metadata)


class ExtractChargeDataTest(TestCase):
    """Tests for _extract_charge_data helper."""

    def test_extracts_from_flat_ndarray(self):
        raw_data = {"IMBC1I": np.array([5e8, 6e8, 4e8])}
        charge = _extract_charge_data(raw_data, "IMBC1I")
        np.testing.assert_array_equal(charge, [5e8, 6e8, 4e8])

    def test_extracts_tmit_from_bpm_dict(self):
        raw_data = {
            "BPM2": {
                "x": np.zeros(3),
                "y": np.zeros(3),
                "tmit": np.array([1e9, 2e9, 3e9]),
            }
        }
        charge = _extract_charge_data(raw_data, "BPM2")
        np.testing.assert_array_equal(charge, [1e9, 2e9, 3e9])

    def test_raises_if_bpm_dict_missing_tmit(self):
        raw_data = {"BPM2": {"x": np.zeros(3), "y": np.zeros(3)}}
        with self.assertRaises(ValueError) as ctx:
            _extract_charge_data(raw_data, "BPM2")
        self.assertIn("tmit", str(ctx.exception))

    def test_raises_if_device_not_in_raw_data(self):
        with self.assertRaises(ValueError) as ctx:
            _extract_charge_data({}, "IMBC1I")
        self.assertIn("not found", str(ctx.exception))


class ResolveToroidTest(TestCase):
    """Tests for _resolve_toroid helper."""

    def test_explicit_toroid_found(self):
        raw_data = {"WS01": np.zeros(3), "IM01": np.array([1e8, 2e8, 3e8])}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])
        name = _resolve_toroid(result, "IM01")
        self.assertEqual(name, "IM01")

    def test_explicit_toroid_not_in_raw_data_raises(self):
        raw_data = {"WS01": np.zeros(3)}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])
        with self.assertRaises(ValueError) as ctx:
            _resolve_toroid(result, "IM01")
        self.assertIn("not found in raw_data", str(ctx.exception))

    def test_falls_back_to_first_available_from_metadata(self):
        raw_data = {"WS01": np.zeros(3), "IM02": np.array([1e8, 2e8, 3e8])}
        result = _make_collection_result(raw_data, charge_toroids=["IM01", "IM02"])
        name = _resolve_toroid(result, None)
        self.assertEqual(name, "IM02")

    def test_raises_when_no_toroid_available(self):
        raw_data = {"WS01": np.zeros(3)}
        result = _make_collection_result(raw_data, charge_toroids=["IM01", "IM02"])
        with self.assertRaises(ValueError) as ctx:
            _resolve_toroid(result, None)
        self.assertIn("No charge toroid data found", str(ctx.exception))

    def test_raises_when_no_charge_toroids_in_metadata(self):
        raw_data = {"WS01": np.zeros(3)}
        result = _make_collection_result(raw_data, charge_toroids=None)
        with self.assertRaises(ValueError) as ctx:
            _resolve_toroid(result, None)
        self.assertIn("No charge_toroids defined", str(ctx.exception))


class ComputeChargeNormalizationTest(TestCase):
    """Tests for the public compute_charge_normalization function."""

    def test_normalization_factors_correct(self):
        charge = np.array([2e8, 4e8, 6e8])
        raw_data = {"WS01": np.zeros(3), "IM01": charge}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        factors, valid_mask, name = compute_charge_normalization(result, toroid="IM01")

        expected_mean = 4e8
        np.testing.assert_allclose(factors, expected_mean / charge)
        np.testing.assert_array_equal(valid_mask, [True, True, True])
        self.assertEqual(name, "IM01")

    def test_low_charge_masked_out(self):
        charge = np.array([5e8, 1e6, 3e8])
        raw_data = {"WS01": np.zeros(3), "IM01": charge}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        factors, valid_mask, _ = compute_charge_normalization(result, toroid="IM01")

        self.assertFalse(valid_mask[1])
        self.assertEqual(factors[1], 1.0)
        expected_mean = np.mean([5e8, 3e8])
        np.testing.assert_allclose(factors[0], expected_mean / 5e8)
        np.testing.assert_allclose(factors[2], expected_mean / 3e8)

    def test_nan_charge_passes_mask_but_factor_is_one(self):
        charge = np.array([5e8, np.nan, 3e8])
        raw_data = {"WS01": np.zeros(3), "IM01": charge}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        factors, valid_mask, _ = compute_charge_normalization(result, toroid="IM01")

        self.assertTrue(valid_mask[1])
        self.assertEqual(factors[1], 1.0)
        # Mean computed from valid non-NaN: mean([5e8, 3e8]) = 4e8
        expected_mean = 4e8
        np.testing.assert_allclose(factors[0], expected_mean / 5e8)
        np.testing.assert_allclose(factors[2], expected_mean / 3e8)

    def test_explicit_toroid_selection(self):
        raw_data = {
            "WS01": np.zeros(3),
            "IM01": np.array([1e8, 1e8, 1e8]),
            "IM02": np.array([2e8, 4e8, 6e8]),
        }
        result = _make_collection_result(raw_data, charge_toroids=["IM01", "IM02"])

        factors, _, name = compute_charge_normalization(result, toroid="IM02")

        expected_mean = 4e8
        np.testing.assert_allclose(factors, expected_mean / np.array([2e8, 4e8, 6e8]))
        self.assertEqual(name, "IM02")

    def test_default_toroid_resolution(self):
        raw_data = {"WS01": np.zeros(3), "IM01": np.array([3e8, 3e8, 3e8])}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        factors, valid_mask, name = compute_charge_normalization(result)

        np.testing.assert_allclose(factors, 1.0)
        np.testing.assert_array_equal(valid_mask, [True, True, True])
        self.assertEqual(name, "IM01")

    def test_raises_when_no_toroid_data(self):
        raw_data = {"WS01": np.zeros(3)}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        with self.assertRaises(ValueError):
            compute_charge_normalization(result)

    def test_all_low_charge_produces_all_masked(self):
        charge = np.array([1e5, 1e6, 1e4])
        raw_data = {"WS01": np.zeros(3), "IM01": charge}
        result = _make_collection_result(raw_data, charge_toroids=["IM01"])

        factors, valid_mask, _ = compute_charge_normalization(result, toroid="IM01")

        np.testing.assert_array_equal(valid_mask, [False, False, False])
        np.testing.assert_array_equal(factors, [1.0, 1.0, 1.0])

    def test_bpm_type_toroid_with_tmit_key(self):
        raw_data = {
            "WS01": np.zeros(3),
            "BPM2": {
                "x": np.zeros(3),
                "y": np.zeros(3),
                "tmit": np.array([2e8, 4e8, 6e8]),
            },
        }
        result = _make_collection_result(raw_data, charge_toroids=["BPM2"])

        factors, valid_mask, name = compute_charge_normalization(result, toroid="BPM2")

        expected_mean = 4e8
        np.testing.assert_allclose(factors, expected_mean / np.array([2e8, 4e8, 6e8]))
        np.testing.assert_array_equal(valid_mask, [True, True, True])
        self.assertEqual(name, "BPM2")

    def test_returns_resolved_toroid_when_none_passed(self):
        raw_data = {
            "WS01": np.zeros(3),
            "IM02": np.array([2e8, 4e8, 6e8]),
        }
        result = _make_collection_result(raw_data, charge_toroids=["IM01", "IM02"])

        _, _, name = compute_charge_normalization(result, toroid=None)

        self.assertEqual(name, "IM02")
