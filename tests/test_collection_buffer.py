from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from slac_measurements.wires.collection.base import BaseWireMeasurementCollection
from slac_measurements.wires.collection.buffer import reserve_buffer


_MODULE = "slac_measurements.wires.collection.buffer"


@patch(f"{_MODULE}._get_username", return_value="tester")
@patch(f"{_MODULE}.create_buffer")
class ReserveBufferAreaTest(TestCase):
    def test_area_forwarded_to_create_buffer(self, mock_create, _user):
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120, area="LTUH")

        self.assertEqual(mock_create.call_args.kwargs["area"], "LTUH")

    def test_area_defaults_to_none(self, mock_create, _user):
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120)

        self.assertIsNone(mock_create.call_args.kwargs["area"])


def _collection(beam_rate, rate_limit_hz, n_measurements=1200):
    """Minimal stand-in for a collection; avoids the log-dir setup validator."""
    fake = SimpleNamespace(
        beam_profile_device=SimpleNamespace(beam_rate=beam_rate),
        buffer=SimpleNamespace(
            n_measurements=n_measurements, rate_limit_hz=rate_limit_hz
        ),
    )
    fake._effective_buffer_rate = (
        lambda: BaseWireMeasurementCollection._effective_buffer_rate(fake)
    )
    return fake


class EffectiveRateTest(TestCase):
    def test_no_rate_limit_uses_beam_rate(self):
        fake = _collection(beam_rate=120, rate_limit_hz=None)
        self.assertEqual(
            BaseWireMeasurementCollection._effective_buffer_rate(fake), 120
        )

    def test_rate_limit_caps_beam_rate(self):
        fake = _collection(beam_rate=120, rate_limit_hz=30.0)
        self.assertEqual(
            BaseWireMeasurementCollection._effective_buffer_rate(fake), 30.0
        )

    def test_rate_limit_above_beam_rate_uses_beam_rate(self):
        fake = _collection(beam_rate=10, rate_limit_hz=30.0)
        self.assertEqual(BaseWireMeasurementCollection._effective_buffer_rate(fake), 10)

    def test_buffer_without_rate_limit_attr_uses_beam_rate(self):
        fake = _collection(beam_rate=120, rate_limit_hz=None)
        fake.buffer = MagicMock(spec=["n_measurements"])
        self.assertEqual(
            BaseWireMeasurementCollection._effective_buffer_rate(fake), 120
        )

    def test_timeout_scales_with_rate_limit(self):
        unmasked = BaseWireMeasurementCollection._calculate_acquisition_timeout_s(
            _collection(beam_rate=120, rate_limit_hz=None)
        )
        masked = BaseWireMeasurementCollection._calculate_acquisition_timeout_s(
            _collection(beam_rate=120, rate_limit_hz=30.0)
        )
        # 1200 points: 10 s at 120 Hz -> 20 s; 40 s at 30 Hz -> 50 s.
        self.assertAlmostEqual(unmasked, 20.0)
        self.assertAlmostEqual(masked, 50.0)


@patch("slac_measurements.utils.wait_until", return_value=True)
class StepSpeedRateTest(TestCase):
    def _device(self):
        device = MagicMock()
        device.speed_max = 5000
        device.scan_pulses = 100
        device.beam_rate = 120
        return device

    def _move(self, device, **kwargs):
        from slac_measurements.wires.motion.step import move_to_step_position

        move_to_step_position(
            device,
            MagicMock(),
            position=1100,
            position_index=1,
            total_positions=2,
            positions=[100, 1100],
            **kwargs,
        )

    def test_default_uses_device_beam_rate(self, _wait):
        device = self._device()
        self._move(device)
        # 1000 um / 100 pulses * 120 Hz
        self.assertEqual(device.speed, 1200)

    def test_effective_rate_overrides_beam_rate(self, _wait):
        device = self._device()
        self._move(device, beam_rate=30.0)
        self.assertEqual(device.speed, 300)
