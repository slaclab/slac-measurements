from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from slac_measurements.wires.collection.base import BaseWireMeasurementCollection
from slac_measurements.wires.collection.buffer import (
    BufferError,
    _calculate_buffer_points,
    reserve_buffer,
)


_MODULE = "slac_measurements.wires.collection.buffer"
_BYKIK = "slac_timing.masks.bykik_inclusion_masks"


def _fake_buffer(rate_limit_hz=None):
    return MagicMock(rate_limit_hz=rate_limit_hz, number=3, n_measurements=0)


@patch(f"{_MODULE}._get_username", return_value="tester")
@patch(f"{_MODULE}.create_buffer")
@patch(_BYKIK)
class ReserveBufferTest(TestCase):
    def _setup(self, mock_bykik, mock_create, rate_limit_hz):
        mock_bykik.return_value = (["pockcel_perm"], rate_limit_hz)
        mock_create.return_value = _fake_buffer(rate_limit_hz)

    def _n_measurements(self, mock_create):
        return mock_create.call_args.kwargs["n_measurements"]

    def test_area_forwarded_to_create_buffer(self, mock_bykik, mock_create, _user):
        self._setup(mock_bykik, mock_create, None)
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120, area="LTUH")

        self.assertEqual(mock_create.call_args.kwargs["area"], "LTUH")
        mock_bykik.assert_called_once_with("CU_HXR", "LTUH")

    def test_area_defaults_to_none(self, mock_bykik, mock_create, _user):
        self._setup(mock_bykik, mock_create, None)
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120)

        self.assertIsNone(mock_create.call_args.kwargs["area"])

    def test_unmasked_size_unchanged(self, mock_bykik, mock_create, _user):
        self._setup(mock_bykik, mock_create, None)
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120, area="LTUH")

        self.assertEqual(
            self._n_measurements(mock_create), _calculate_buffer_points(350, 120)
        )

    def test_otf_masked_size_scaled_to_buffer_rate(
        self, mock_bykik, mock_create, _user
    ):
        self._setup(mock_bykik, mock_create, 30.0)
        reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120, area="LTUH")

        full = _calculate_buffer_points(350, 120)
        self.assertEqual(self._n_measurements(mock_create), -(-full // 4))

    def test_step_masked_size_uses_buffer_rate(self, mock_bykik, mock_create, _user):
        self._setup(mock_bykik, mock_create, 30.0)
        reserve_buffer(
            beampath="CU_HXR",
            pulses=350,
            beam_rate=120,
            area="LTUH",
            wire_follows_buffer_rate=True,
        )

        self.assertEqual(
            self._n_measurements(mock_create), _calculate_buffer_points(350, 30.0)
        )

    def test_rate_change_during_reservation_releases_and_raises(
        self, mock_bykik, mock_create, _user
    ):
        mock_bykik.return_value = (["pockcel_perm"], 30.0)
        buf = _fake_buffer(rate_limit_hz=None)
        mock_create.return_value = buf

        with self.assertRaises(BufferError):
            reserve_buffer(beampath="CU_HXR", pulses=350, beam_rate=120, area="LTUH")
        buf.release.assert_called_once()


class CalculateBufferPointsTest(TestCase):
    def test_historical_mode(self):
        self.assertEqual(_calculate_buffer_points(350, 120), 1453)

    def test_buffer_rate_scales_points(self):
        self.assertEqual(_calculate_buffer_points(350, 120, 30.0), 364)

    def test_buffer_rate_not_below_wire_rate_is_unscaled(self):
        self.assertEqual(_calculate_buffer_points(350, 120, 120), 1453)


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
