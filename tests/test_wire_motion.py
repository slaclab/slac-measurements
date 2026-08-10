from unittest import TestCase
from unittest.mock import MagicMock, patch

from slac_measurements.wires.motion.step import (
    get_step_positions,
    initialize_step_with_retry,
    move_to_step_position,
)
from slac_measurements.wires.motion.otf import initialize_otf_with_retry
from slac_measurements.wires.motion.utils import poll_motor_rbv, SETTLE_COUNT


class InitializeStepWithRetryTest(TestCase):
    def _make_device(self, enabled=False):
        device = MagicMock()
        device.name = "TEST_WIRE"
        device.enabled = enabled
        return device

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_skips_if_already_enabled(self, mock_wait):
        device = self._make_device(enabled=True)
        initialize_step_with_retry(device)
        device.initialize.assert_not_called()
        mock_wait.assert_not_called()

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_initializes_and_succeeds_first_attempt(self, mock_wait):
        device = self._make_device(enabled=False)
        initialize_step_with_retry(device)
        device.initialize.assert_called_once()

    @patch("slac_measurements.utils.wait_until", side_effect=[False, True])
    def test_retries_on_failure(self, mock_wait):
        device = self._make_device(enabled=False)
        initialize_step_with_retry(device)
        self.assertEqual(device.initialize.call_count, 2)

    @patch("slac_measurements.utils.wait_until", return_value=False)
    def test_raises_after_max_attempts(self, mock_wait):
        device = self._make_device(enabled=False)
        with self.assertRaises(RuntimeError) as ctx:
            initialize_step_with_retry(device, max_attempts=3)
        self.assertIn("Failed to initialize", str(ctx.exception))
        self.assertEqual(device.initialize.call_count, 3)


class GetStepPositionsTest(TestCase):
    def test_returns_sorted_inner_outer_positions(self):
        device = MagicMock()
        device.active_profiles.return_value = ["x", "y"]
        device.x_wire_inner = 200
        device.x_wire_outer = 500
        device.y_wire_inner = 800
        device.y_wire_outer = 1100

        positions = get_step_positions(device)

        self.assertEqual(positions, [200, 500, 800, 1100])

    def test_single_profile(self):
        device = MagicMock()
        device.active_profiles.return_value = ["x"]
        device.x_wire_inner = 100
        device.x_wire_outer = 300

        positions = get_step_positions(device)

        self.assertEqual(positions, [100, 300])


class MoveToStepPositionTest(TestCase):
    def _make_device(self, motor_rbv=0):
        device = MagicMock()
        device.name = "TEST_WIRE"
        device.motor_rbv = motor_rbv
        device.speed_max = 5000
        device.scan_pulses = 350
        device.beam_rate = 120
        return device

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_moves_to_position_successfully(self, mock_wait):
        device = self._make_device()
        positions = [100, 500, 900, 1200]

        move_to_step_position(
            device,
            position=500,
            position_index=1,
            total_positions=4,
            positions=positions,
        )

        self.assertEqual(device.motor, 500)

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_even_index_uses_max_speed(self, mock_wait):
        device = self._make_device()
        positions = [100, 500, 900, 1200]

        move_to_step_position(
            device,
            position=100,
            position_index=0,
            total_positions=4,
            positions=positions,
        )

        self.assertEqual(device.speed, int(device.speed_max))

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_odd_index_calculates_speed_from_delta(self, mock_wait):
        device = self._make_device()
        device.scan_pulses = 350
        device.beam_rate = 120
        positions = [100, 500]

        move_to_step_position(
            device,
            position=500,
            position_index=1,
            total_positions=2,
            positions=positions,
        )

        expected_speed = int((400 / 350) * 120)
        self.assertEqual(device.speed, expected_speed)

    @patch("slac_measurements.utils.wait_until", return_value=False)
    def test_raises_if_position_not_reached(self, mock_wait):
        device = self._make_device(motor_rbv=0)
        positions = [100, 500]

        with self.assertRaises(RuntimeError) as ctx:
            move_to_step_position(
                device,
                position=500,
                position_index=1,
                total_positions=2,
                positions=positions,
            )
        self.assertIn("did not reach position", str(ctx.exception))


class InitializeOTFWithRetryTest(TestCase):
    def _make_device(self, homed=False, on_status=False):
        device = MagicMock()
        device.name = "TEST_WIRE"
        device.homed = homed
        device.on_status = on_status
        return device

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_succeeds_first_attempt(self, mock_wait):
        device = self._make_device()
        initialize_otf_with_retry(device)
        device.start_scan.assert_called_once()

    @patch("slac_measurements.utils.wait_until", side_effect=[False, False, True])
    def test_retries_until_success(self, mock_wait):
        device = self._make_device()
        initialize_otf_with_retry(device)
        self.assertEqual(device.start_scan.call_count, 3)

    @patch("slac_measurements.utils.wait_until", return_value=False)
    def test_raises_after_max_attempts(self, mock_wait):
        device = self._make_device()
        with self.assertRaises(RuntimeError) as ctx:
            initialize_otf_with_retry(device, max_attempts=2)
        self.assertIn("Failed to initialize", str(ctx.exception))
        self.assertEqual(device.start_scan.call_count, 2)

    @patch("slac_measurements.utils.wait_until", return_value=True)
    def test_always_calls_start_scan(self, mock_wait):
        """start_scan must be called even if homed/on_status are already True."""
        device = self._make_device(homed=True, on_status=True)
        initialize_otf_with_retry(device)
        device.start_scan.assert_called_once()


class PollMotorRBVTest(TestCase):
    @patch("time.sleep")
    def test_returns_positions_after_settling(self, mock_sleep):
        device = MagicMock()
        # Simulate: moving (large jumps), then settling (small changes)
        moving = [float(i * 100) for i in range(5)]
        settled = [500.0] * (SETTLE_COUNT + 1)
        all_positions = moving + settled
        device.motor_rbv = all_positions[0]
        type(device).motor_rbv = property(
            lambda self, pos=iter(all_positions): next(pos)
        )

        result = poll_motor_rbv(device, poll_interval_s=0)

        self.assertGreater(len(result), SETTLE_COUNT)
        self.assertAlmostEqual(result[-1], 500.0, places=0)

    @patch("time.sleep")
    def test_immediate_settle_if_not_moving(self, mock_sleep):
        device = MagicMock()
        # Wire is already stationary
        stationary = [1000.0] * (SETTLE_COUNT + 2)
        type(device).motor_rbv = property(lambda self, pos=iter(stationary): next(pos))

        result = poll_motor_rbv(device, poll_interval_s=0)

        self.assertEqual(len(result), SETTLE_COUNT + 1)
