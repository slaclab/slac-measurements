from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from slac_measurements.wires.collection.beamless_step import run_beamless_step_scan
from slac_measurements.wires.collection.beamless_otf import run_beamless_otf_scan


class BeamlessStepScanTest(TestCase):
    def _make_device(self):
        device = MagicMock()
        device.name = "WIRE:TEST:100"
        device.area = "L3"
        device.x_range = [100, 500]
        device.y_range = [800, 1200]
        device.u_range = [0, 0]
        device.active_profiles.return_value = ["x", "y"]
        device.install_angle = 45.0
        device.x_wire_inner = 100
        device.x_wire_outer = 500
        device.y_wire_inner = 800
        device.y_wire_outer = 1200
        return device

    @patch("slac_measurements.wires.collection.beamless_step.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_step.move_to_step_position")
    @patch(
        "slac_measurements.wires.collection.beamless_step.initialize_step_with_retry"
    )
    def test_returns_collection_result_with_recorded_positions(
        self, mock_init, mock_move, mock_poll
    ):
        device = self._make_device()
        mock_poll.return_value = [100.0, 200.0, 500.0, 800.0, 1200.0]

        result = run_beamless_step_scan(device)

        mock_init.assert_called_once()
        np.testing.assert_array_equal(
            result.raw_data["WIRE:TEST:100"],
            np.array([100.0, 200.0, 500.0, 800.0, 1200.0]),
        )

    @patch("slac_measurements.wires.collection.beamless_step.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_step.move_to_step_position")
    @patch(
        "slac_measurements.wires.collection.beamless_step.initialize_step_with_retry"
    )
    def test_metadata_fields(self, mock_init, mock_move, mock_poll):
        device = self._make_device()
        mock_poll.return_value = [100.0, 500.0]

        result = run_beamless_step_scan(device)

        self.assertEqual(result.metadata.wire_name, "WIRE:TEST:100")
        self.assertEqual(result.metadata.area, "L3")
        self.assertIsNone(result.metadata.beampath)
        self.assertIsNone(result.metadata.detectors)
        self.assertEqual(result.metadata.scan_ranges["x"], (100, 500))
        self.assertEqual(result.metadata.scan_ranges["y"], (800, 1200))
        self.assertEqual(result.metadata.active_profiles, ["x", "y"])
        self.assertEqual(result.metadata.install_angle, 45.0)
        self.assertEqual(result.metadata.notes, "beamless step motion test")

    @patch("slac_measurements.wires.collection.beamless_step.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_step.move_to_step_position")
    @patch(
        "slac_measurements.wires.collection.beamless_step.initialize_step_with_retry"
    )
    def test_moves_to_all_step_positions(self, mock_init, mock_move, mock_poll):
        device = self._make_device()
        mock_poll.return_value = [100.0]

        run_beamless_step_scan(device)

        # 2 profiles x 2 positions (inner/outer) = 4 moves
        self.assertEqual(mock_move.call_count, 4)

    @patch("slac_measurements.wires.collection.beamless_step.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_step.move_to_step_position")
    @patch(
        "slac_measurements.wires.collection.beamless_step.initialize_step_with_retry"
    )
    def test_retracts_after_motion(self, mock_init, mock_move, mock_poll):
        device = self._make_device()
        mock_poll.return_value = [100.0]

        run_beamless_step_scan(device)

        device.retract.assert_called_once()


class BeamlessOTFScanTest(TestCase):
    def _make_device(self):
        device = MagicMock()
        device.name = "WIRE:TEST:200"
        device.area = "L2"
        device.x_range = [0, 1000]
        device.y_range = [500, 1500]
        device.u_range = [0, 0]
        device.active_profiles.return_value = ["x"]
        device.install_angle = 0.0
        return device

    @patch("slac_measurements.wires.collection.beamless_otf.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_otf.initialize_otf_with_retry")
    def test_returns_collection_result_with_recorded_positions(
        self, mock_init, mock_poll
    ):
        device = self._make_device()
        mock_poll.return_value = [0.0, 250.0, 500.0, 750.0, 1000.0]

        result = run_beamless_otf_scan(device)

        mock_init.assert_called_once()
        np.testing.assert_array_equal(
            result.raw_data["WIRE:TEST:200"],
            np.array([0.0, 250.0, 500.0, 750.0, 1000.0]),
        )

    @patch("slac_measurements.wires.collection.beamless_otf.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_otf.initialize_otf_with_retry")
    def test_metadata_fields(self, mock_init, mock_poll):
        device = self._make_device()
        mock_poll.return_value = [0.0, 500.0, 1000.0]

        result = run_beamless_otf_scan(device)

        self.assertEqual(result.metadata.wire_name, "WIRE:TEST:200")
        self.assertEqual(result.metadata.area, "L2")
        self.assertIsNone(result.metadata.beampath)
        self.assertIsNone(result.metadata.detectors)
        self.assertEqual(result.metadata.scan_ranges["x"], (0, 1000))
        self.assertEqual(result.metadata.scan_ranges["y"], (500, 1500))
        self.assertEqual(result.metadata.active_profiles, ["x"])
        self.assertEqual(result.metadata.install_angle, 0.0)
        self.assertEqual(result.metadata.notes, "beamless otf motion test")

    @patch("slac_measurements.wires.collection.beamless_otf.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_otf.initialize_otf_with_retry")
    def test_passes_poll_interval(self, mock_init, mock_poll):
        device = self._make_device()
        mock_poll.return_value = [0.0]

        run_beamless_otf_scan(device, poll_interval_s=0.1)

        mock_poll.assert_called_once_with(device, 0.1)

    @patch("slac_measurements.wires.collection.beamless_otf.poll_motor_rbv")
    @patch("slac_measurements.wires.collection.beamless_otf.initialize_otf_with_retry")
    def test_initializes_before_polling(self, mock_init, mock_poll):
        device = self._make_device()
        call_order = []
        mock_init.side_effect = lambda *a, **kw: call_order.append("init")
        mock_poll.side_effect = lambda *a, **kw: (call_order.append("poll") or [0.0])

        run_beamless_otf_scan(device)

        self.assertEqual(call_order, ["init", "poll"])
