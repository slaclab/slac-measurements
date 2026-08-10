import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np

from slac_measurements.wires.analysis.jitter_correction import (
    _compute_orbit_fit,
    _extract_bpm_data,
    compute_jitter,
    get_jitter_rmat,
)
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)

# meme is an optional dependency not installed in CI — provide a fake module
# so that the lazy `from meme.model import Model` inside get_jitter_rmat can
# be patched without ModuleNotFoundError.
_meme = ModuleType("meme")
_meme_model = ModuleType("meme.model")
_meme_model.Model = MagicMock()
_meme.model = _meme_model
sys.modules.setdefault("meme", _meme)
sys.modules.setdefault("meme.model", _meme_model)


class ExtractBpmDataTest(TestCase):
    def test_extracts_valid_bpm_entries(self):
        raw_data = {
            "WIRE:TEST": np.array([1.0, 2.0]),
            "BPM10": {"x": np.array([0.1, 0.2]), "y": np.array([0.3, 0.4])},
            "BPM11": {"x": np.array([0.5, 0.6]), "y": np.array([0.7, 0.8])},
        }

        bpm_x, bpm_y, names = _extract_bpm_data(raw_data)

        self.assertEqual(names, ["BPM10", "BPM11"])
        np.testing.assert_array_equal(bpm_x[0], [0.1, 0.2])
        np.testing.assert_array_equal(bpm_y[1], [0.7, 0.8])
        self.assertEqual(bpm_x.shape, (2, 2))

    def test_skips_non_dict_bpm_entries(self):
        raw_data = {
            "BPM10": np.array([1.0, 2.0]),
            "BPM11": {"x": np.array([0.1, 0.2]), "y": np.array([0.3, 0.4])},
        }

        bpm_x, bpm_y, names = _extract_bpm_data(raw_data)

        self.assertEqual(names, ["BPM11"])

    def test_skips_bpm_with_missing_xy(self):
        raw_data = {
            "BPM10": {"x": np.array([0.1, 0.2])},
            "BPM11": {"x": np.array([0.5, 0.6]), "y": np.array([0.7, 0.8])},
        }

        bpm_x, bpm_y, names = _extract_bpm_data(raw_data)

        self.assertEqual(names, ["BPM11"])

    def test_skips_all_nan_data(self):
        raw_data = {
            "BPM10": {
                "x": np.array([np.nan, np.nan]),
                "y": np.array([0.1, 0.2]),
            },
            "BPM11": {"x": np.array([0.5, 0.6]), "y": np.array([0.7, 0.8])},
        }

        bpm_x, bpm_y, names = _extract_bpm_data(raw_data)

        self.assertEqual(names, ["BPM11"])

    def test_raises_if_no_valid_bpms(self):
        raw_data = {"WIRE:TEST": np.array([1.0, 2.0])}

        with self.assertRaises(ValueError) as ctx:
            _extract_bpm_data(raw_data)
        self.assertIn("No valid BPM", str(ctx.exception))

    def test_sorts_bpm_names(self):
        raw_data = {
            "BPM20": {"x": np.array([0.1]), "y": np.array([0.2])},
            "BPM05": {"x": np.array([0.3]), "y": np.array([0.4])},
            "BPM12": {"x": np.array([0.5]), "y": np.array([0.6])},
        }

        _, _, names = _extract_bpm_data(raw_data)

        self.assertEqual(names, ["BPM05", "BPM12", "BPM20"])


class ComputeOrbitFitTest(TestCase):
    def test_zero_jitter_from_constant_bpm_readings(self):
        n_bpms, n_pulses = 3, 50
        bpm_x = np.ones((n_bpms, n_pulses)) * 5.0
        bpm_y = np.ones((n_bpms, n_pulses)) * 3.0
        rmat_x = np.array([[1.0, 0.5, 0.0]] * n_bpms)
        rmat_y = np.array([[1.0, 0.3, 0.0]] * n_bpms)

        jitter_x, jitter_y = _compute_orbit_fit(bpm_x, bpm_y, rmat_x, rmat_y)

        np.testing.assert_allclose(jitter_x, 0.0, atol=1e-10)
        np.testing.assert_allclose(jitter_y, 0.0, atol=1e-10)

    def test_reconstructs_known_position_jitter(self):
        n_bpms, n_pulses = 4, 100
        rng = np.random.default_rng(42)

        # Known orbit at wire: x position oscillates, x' is zero
        wire_x_m = rng.normal(0, 50e-6, n_pulses)  # 50 um rms jitter

        # R-matrix: BPM_x = R11 * x + R12 * x'
        rmat_x = np.zeros((n_bpms, 3))
        rmat_x[:, 0] = [1.2, 0.8, 1.5, 0.6]  # R11
        rmat_x[:, 1] = [0.3, 0.7, 0.1, 0.9]  # R12

        # Simulate BPM readings (mm) from orbit at wire (m)
        # BPM_x = R11 * x_wire (x' = 0, dispersion = 0)
        bpm_x_mm = rmat_x[:, 0:1] * wire_x_m[np.newaxis, :] * 1e3
        # Add mean offset (should be subtracted in the fit)
        bpm_x_mm += rng.normal(0, 1, (n_bpms, 1))

        rmat_y = np.eye(3)[: n_bpms // 2 + 2, :]  # dummy
        rmat_y = np.zeros((n_bpms, 3))
        rmat_y[:, 0] = 1.0
        bpm_y_mm = np.zeros((n_bpms, n_pulses))

        jitter_x, _ = _compute_orbit_fit(bpm_x_mm, bpm_y_mm, rmat_x, rmat_y)

        # Should reconstruct the original jitter (m -> um)
        expected_um = wire_x_m * 1e6
        # Remove mean from expected (fit works on deviations)
        expected_um -= expected_um.mean()
        jitter_x -= jitter_x.mean()
        np.testing.assert_allclose(jitter_x, expected_um, atol=1.0)

    def test_drops_dispersion_column_below_threshold(self):
        n_bpms, n_pulses = 3, 20
        bpm_x = np.random.default_rng(0).normal(0, 1, (n_bpms, n_pulses))
        bpm_y = np.zeros((n_bpms, n_pulses))

        # R16 below threshold — should be dropped
        rmat_x = np.array([[1.0, 0.5, 1e-5]] * n_bpms)
        rmat_y = np.array([[1.0, 0.3, 1e-5]] * n_bpms)

        # Should not raise (would fail if trying to fit 3 params with bad conditioning)
        jitter_x, jitter_y = _compute_orbit_fit(bpm_x, bpm_y, rmat_x, rmat_y)

        self.assertEqual(jitter_x.shape, (n_pulses,))
        self.assertEqual(jitter_y.shape, (n_pulses,))

    def test_includes_dispersion_column_above_threshold(self):
        n_bpms, n_pulses = 4, 30
        rng = np.random.default_rng(7)
        bpm_x = rng.normal(0, 1, (n_bpms, n_pulses))
        bpm_y = np.zeros((n_bpms, n_pulses))

        # R16 above threshold — should be included
        rmat_x = np.array([[1.0, 0.5, 0.1]] * n_bpms)
        rmat_y = np.array([[1.0, 0.3, 0.0]] * n_bpms)

        jitter_x, jitter_y = _compute_orbit_fit(bpm_x, bpm_y, rmat_x, rmat_y)

        self.assertEqual(jitter_x.shape, (n_pulses,))


class GetJitterRmatTest(TestCase):
    @patch("meme.model.Model")
    def test_extracts_correct_rmat_elements(self, mock_model_cls):
        rmat_6x6 = np.zeros((6, 6))
        rmat_6x6[0, 0] = 1.1  # R11
        rmat_6x6[0, 1] = 2.2  # R12
        rmat_6x6[0, 5] = 3.3  # R16
        rmat_6x6[2, 2] = 4.4  # R33
        rmat_6x6[2, 3] = 5.5  # R34
        rmat_6x6[2, 5] = 6.6  # R36

        mock_model = mock_model_cls.return_value
        mock_model.get_rmat.return_value = rmat_6x6

        rmat_x, rmat_y = get_jitter_rmat(
            "WIRE:TEST", ["BPM10", "BPM11"], "SC_HXR", "BLEM"
        )

        self.assertEqual(rmat_x.shape, (2, 3))
        np.testing.assert_array_equal(rmat_x[0], [1.1, 2.2, 3.3])
        np.testing.assert_array_equal(rmat_y[0], [4.4, 5.5, 6.6])
        mock_model_cls.assert_called_once_with(
            "SC_HXR", model_source="BLEM", use_design=False
        )
        self.assertEqual(mock_model.get_rmat.call_count, 2)


class ComputeJitterTest(TestCase):
    def _make_collection_result(self, raw_data):
        metadata = MeasurementMetadata(
            wire_name="WIRE:TEST:100",
            area="L3",
            beampath="SC_HXR",
            detectors=["D1"],
            default_detector="D1",
            scan_ranges={"x": (0, 1000)},
            timestamp=datetime.now(),
            active_profiles=["x"],
            install_angle=45.0,
        )
        return WireMeasurementCollectionResult(
            raw_data=raw_data,
            metadata=metadata,
        )

    @patch("slac_measurements.wires.analysis.jitter_correction.get_jitter_rmat")
    def test_raises_with_fewer_than_2_bpms(self, mock_rmat):
        raw_data = {
            "WIRE:TEST:100": np.array([1.0]),
            "BPM10": {"x": np.array([0.1, 0.2]), "y": np.array([0.3, 0.4])},
        }
        result = self._make_collection_result(raw_data)

        with self.assertRaises(ValueError) as ctx:
            compute_jitter(result, beampath="SC_HXR")
        self.assertIn("at least 2 BPMs", str(ctx.exception))

    @patch("slac_measurements.wires.analysis.jitter_correction.get_jitter_rmat")
    def test_end_to_end_with_mocked_rmat(self, mock_rmat):
        n_pulses = 50
        raw_data = {
            "WIRE:TEST:100": np.zeros(n_pulses),
            "BPM10": {
                "x": np.ones(n_pulses) * 2.0,
                "y": np.ones(n_pulses) * 1.0,
            },
            "BPM11": {
                "x": np.ones(n_pulses) * 3.0,
                "y": np.ones(n_pulses) * 2.0,
            },
        }
        result = self._make_collection_result(raw_data)

        rmat_x = np.array([[1.0, 0.5, 0.0], [0.8, 0.3, 0.0]])
        rmat_y = np.array([[1.0, 0.4, 0.0], [0.9, 0.2, 0.0]])
        mock_rmat.return_value = (rmat_x, rmat_y)

        jitter_x, jitter_y = compute_jitter(result, beampath="SC_HXR")

        # Constant BPM readings -> zero jitter
        np.testing.assert_allclose(jitter_x, 0.0, atol=1e-10)
        np.testing.assert_allclose(jitter_y, 0.0, atol=1e-10)
        mock_rmat.assert_called_once_with(
            "WIRE:TEST:100", ["BPM10", "BPM11"], "SC_HXR", "BLEM"
        )
