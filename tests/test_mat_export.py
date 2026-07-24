"""Tests for the MATLAB .mat export functionality."""

import numpy as np
import scipy.io
from datetime import datetime, timezone
from tempfile import NamedTemporaryFile
from unittest import TestCase

from slac_measurements.wires.analysis.mat_export import (
    analysis_result_to_mat,
    datetime_to_matlab_datenum,
)
from slac_measurements.wires.analysis.results import (
    DetectorFit,
    DetectorProfileMeasurement,
    FitResult,
    ProfileMeasurement,
    WireMeasurementAnalysisResult,
)
from slac_measurements.wires.collection.results import (
    MeasurementMetadata,
    WireMeasurementCollectionResult,
)


def _make_analysis_result(
    *,
    active_profiles=("x", "y"),
    include_bpms=True,
) -> WireMeasurementAnalysisResult:
    """Build a synthetic analysis result for testing."""
    n_pulses = 50
    positions_x = np.linspace(28000, 32000, n_pulses)
    positions_y = np.linspace(15000, 19000, n_pulses)
    signal_x = np.exp(-0.5 * ((positions_x - 30000) / 500) ** 2)
    signal_y = np.exp(-0.5 * ((positions_y - 17000) / 300) ** 2)

    raw_data = {
        "WIRE:LI21:285": positions_x,
        "PMT:LI21:100": signal_x * 1000,
        "PMT:LI21:200": signal_x * 800,
    }

    if include_bpms:
        raw_data["BPMS:LI21:301"] = {
            "x": np.random.randn(n_pulses) * 0.1,
            "y": np.random.randn(n_pulses) * 0.05,
        }
        raw_data["BPMS:LI21:401"] = {
            "x": np.random.randn(n_pulses) * 0.12,
            "y": np.random.randn(n_pulses) * 0.06,
        }

    scan_ranges = {"x": (28000, 32000)}
    profiles_list = list(active_profiles)
    if "y" in active_profiles:
        scan_ranges["y"] = (15000, 19000)

    metadata = MeasurementMetadata(
        wire_name="WIRE:LI21:285",
        area="TEST",
        beampath="CU_HXR",
        detectors=["PMT:LI21:100", "PMT:LI21:200"],
        default_detector="PMT:LI21:100",
        scan_ranges=scan_ranges,
        timestamp=datetime(2025, 3, 15, 14, 30, 0, tzinfo=timezone.utc),
        active_profiles=profiles_list,
        install_angle=45.0,
    )

    collection_result = WireMeasurementCollectionResult(
        raw_data=raw_data, metadata=metadata
    )

    fit_result = {}
    profiles = {}

    if "x" in active_profiles:
        fit_result["x"] = FitResult(
            detectors={
                "PMT:LI21:100": DetectorFit(
                    mean=30000.0,
                    sigma=353.5,
                    amplitude=1.0,
                    offset=0.0,
                    curve=signal_x,
                    positions=positions_x * np.sin(np.radians(45.0)),
                ),
                "PMT:LI21:200": DetectorFit(
                    mean=30000.0,
                    sigma=360.0,
                    amplitude=0.8,
                    offset=0.0,
                    curve=signal_x * 0.8,
                    positions=positions_x * np.sin(np.radians(45.0)),
                ),
            }
        )
        profiles["x"] = ProfileMeasurement(
            positions=positions_x,
            profile_indices=np.arange(n_pulses),
            detectors={
                "PMT:LI21:100": DetectorProfileMeasurement(
                    values=signal_x * 1000, units="counts", label="PMT:LI21:100"
                ),
                "PMT:LI21:200": DetectorProfileMeasurement(
                    values=signal_x * 800, units="counts", label="PMT:LI21:200"
                ),
            },
        )

    if "y" in active_profiles:
        fit_result["y"] = FitResult(
            detectors={
                "PMT:LI21:100": DetectorFit(
                    mean=17000.0,
                    sigma=212.1,
                    amplitude=1.0,
                    offset=0.0,
                    curve=signal_y,
                    positions=positions_y * np.cos(np.radians(45.0)),
                ),
            }
        )
        profiles["y"] = ProfileMeasurement(
            positions=positions_y,
            profile_indices=np.arange(n_pulses),
            detectors={
                "PMT:LI21:100": DetectorProfileMeasurement(
                    values=signal_y * 1000, units="counts", label="PMT:LI21:100"
                ),
            },
        )

    return WireMeasurementAnalysisResult(
        fit_result=fit_result,
        collection_result=collection_result,
        profiles=profiles,
        rms_sizes=np.array([353.5, 212.1]),
        metadata=metadata,
    )


class TestDatetimeToMatlabDatenum(TestCase):
    def test_unix_epoch(self):
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.assertAlmostEqual(datetime_to_matlab_datenum(dt), 719529.0, places=6)

    def test_known_date(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        expected = 739618.0
        self.assertAlmostEqual(datetime_to_matlab_datenum(dt), expected, places=6)

    def test_with_time(self):
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected = 739618.5
        self.assertAlmostEqual(datetime_to_matlab_datenum(dt), expected, places=6)

    def test_naive_datetime_treated_as_utc(self):
        dt_naive = datetime(2025, 1, 1, 0, 0, 0)
        dt_utc = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.assertAlmostEqual(
            datetime_to_matlab_datenum(dt_naive),
            datetime_to_matlab_datenum(dt_utc),
            places=6,
        )


class TestAnalysisResultToMat(TestCase):
    def setUp(self):
        self.result = _make_analysis_result()

    def _export_and_load(self, result=None, **kwargs):
        if result is None:
            result = self.result
        with NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        analysis_result_to_mat(result, path, **kwargs)
        return scipy.io.loadmat(path, squeeze_me=False)

    def _data(self, mat):
        return mat["data"][0, 0]

    def test_top_level_contains_data_struct(self):
        mat = self._export_and_load()
        self.assertIn("data", mat)

    def test_scalar_string_fields(self):
        mat = self._export_and_load()
        data = self._data(mat)
        self.assertEqual(str(data["name"][0]), "WIRE:LI21:285")
        self.assertEqual(str(data["wireName"][0]), "WIRE:LI21:285")
        self.assertEqual(str(data["wireMode"][0]), "wire")
        self.assertEqual(str(data["beampath"][0]), "CU_HXR")

    def test_timestamp_is_valid_datenum(self):
        mat = self._export_and_load()
        data = self._data(mat)
        ts = float(data["ts"][0, 0])
        self.assertGreater(ts, 700000)
        self.assertLess(ts, 800000)

    def test_wire_geometry_fields(self):
        mat = self._export_and_load()
        data = self._data(mat)
        self.assertAlmostEqual(float(data["wireAngle"][0, 0]), 45.0)

        wire_dir = data["wireDir"][0, 0]
        self.assertTrue(bool(wire_dir["x"][0, 0]))
        self.assertTrue(bool(wire_dir["y"][0, 0]))
        self.assertFalse(bool(wire_dir["u"][0, 0]))

    def test_wire_limit_shape(self):
        mat = self._export_and_load()
        data = self._data(mat)
        wire_limit = data["wireLimit"][0, 0]
        x_limit = wire_limit["x"].ravel()
        self.assertEqual(len(x_limit), 2)
        self.assertAlmostEqual(x_limit[0], 28000.0)
        self.assertAlmostEqual(x_limit[1], 32000.0)

    def test_device_lists(self):
        mat = self._export_and_load()
        data = self._data(mat)
        pmt_list = data["PMTList"].ravel()
        self.assertEqual(len(pmt_list), 2)
        self.assertEqual(str(pmt_list[0][0]), "PMT:LI21:100")

        bpm_list = data["BPMList"].ravel()
        self.assertEqual(len(bpm_list), 2)

    def test_raw_data_shapes(self):
        mat = self._export_and_load()
        data = self._data(mat)
        wire_data = np.asarray(data["wireData"])
        self.assertEqual(wire_data.ndim, 2)
        self.assertEqual(wire_data.shape[0], 1)
        self.assertEqual(wire_data.shape[1], 50)

        pmt_data = np.asarray(data["PMTData"])
        self.assertEqual(pmt_data.shape[0], 2)
        self.assertEqual(pmt_data.shape[1], 50)

    def test_bpm_data_shapes(self):
        mat = self._export_and_load()
        data = self._data(mat)
        bpmx = np.asarray(data["BPMXData"])
        bpmy = np.asarray(data["BPMYData"])
        self.assertEqual(bpmx.shape[0], 2)
        self.assertEqual(bpmx.shape[1], 50)
        self.assertEqual(bpmy.shape[0], 2)

    def test_pos_and_signal_present(self):
        mat = self._export_and_load()
        data = self._data(mat)
        pos = data["pos"][0, 0]
        signal = data["signal"][0, 0]
        pos_x = np.asarray(pos["x"]).ravel()
        sig_x = np.asarray(signal["x"]).ravel()
        self.assertEqual(len(pos_x), 50)
        self.assertEqual(len(sig_x), 50)
        self.assertTrue(np.all(pos_x > 0))

    def _get_beam(self, mat):
        data = self._data(mat)
        return data["beam"][0, 0][0, 0]

    def test_beam_struct_method(self):
        mat = self._export_and_load()
        beam = self._get_beam(mat)
        method = str(beam["method"].ravel()[0])
        self.assertEqual(method, "Gaussian")

    def test_beam_stats_shape(self):
        mat = self._export_and_load()
        beam = self._get_beam(mat)
        stats = np.asarray(beam["stats"]).ravel()
        self.assertEqual(len(stats), 6)

    def test_beam_profx_is_3xN(self):
        mat = self._export_and_load()
        beam = self._get_beam(mat)
        profx = np.asarray(beam["profx"])
        self.assertEqual(profx.shape[0], 3)
        self.assertEqual(profx.shape[1], 50)

    def test_beam_stats_contain_rms_values(self):
        mat = self._export_and_load()
        beam = self._get_beam(mat)
        stats = np.asarray(beam["stats"]).ravel()
        self.assertAlmostEqual(stats[2], 353.5, places=1)
        self.assertAlmostEqual(stats[3], 212.1, places=1)

    def test_select_pmt_is_one_based(self):
        mat = self._export_and_load()
        data = self._data(mat)
        self.assertEqual(float(data["selectPMT"][0, 0]), 1.0)

    def test_single_profile_active(self):
        result = _make_analysis_result(active_profiles=("x",))
        mat = self._export_and_load(result)
        data = self._data(mat)
        wire_dir = data["wireDir"][0, 0]
        self.assertTrue(bool(wire_dir["x"][0, 0]))
        self.assertFalse(bool(wire_dir["y"][0, 0]))

    def test_no_bpms(self):
        result = _make_analysis_result(include_bpms=False)
        mat = self._export_and_load(result)
        data = self._data(mat)
        bpm_list = data["BPMList"].ravel()
        self.assertEqual(len(bpm_list), 0)

    def test_beam_pv_structure(self):
        mat = self._export_and_load()
        data = self._data(mat)
        beam_pv = data["beamPV"]
        self.assertIsNotNone(beam_pv)

    def test_to_mat_method_on_result(self):
        with NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        returned_path = self.result.to_mat(path)
        self.assertEqual(returned_path, path)
        mat = scipy.io.loadmat(path)
        self.assertIn("data", mat)
