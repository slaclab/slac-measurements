import logging
import getpass
import numpy as np

from slac_timing import create_buffer, Buffer


_BUFFER_NAME = "SLAC Tools Wire Scan"
_MAX_BEAM_RATE = 16000
_MIN_BEAM_RATE = 10


class BufferError(Exception):
    pass


def _get_username() -> str:
    """Return the current username."""
    user = getpass.getuser()
    if user:
        return user

    raise BufferError("Could not determine current username for buffer reservation.")


def reserve_buffer(
    beampath: str,
    pulses: int,
    beam_rate: int,
    name: str = _BUFFER_NAME,
    logger: logging.Logger | None = None,
) -> Buffer:
    user = _get_username()
    if logger:
        logger.info("Reserving buffer...")

    buf = create_buffer(
        beampath=beampath,
        n_measurements=_calculate_buffer_points(pulses, beam_rate),
        user=user,
        name=name,
    )

    if logger:
        logger.info("Reserved timing buffer %s.", buf.number)

    return buf


def _calculate_buffer_points(pulses, rate) -> int:
    """
    Determine the number of buffer points for a wire scan.

    The beam rate and pulses per profile are used here to calculate the
    wire speed, which in turn defines how many BSA buffer points are needed
    to capture the full scan. The minimum safe wire speed is calculated
    separately and enforced by the motion IOC. The buffer size must be
    sufficient for data collection while staying under the 20,000-point
    operational limit.

    In the historical mode (120 Hz, 350 pulses), ~1,600 points are
    required; this function returns 1,595. In the expected high-rate mode
    (16 kHz, 5,000 pulses), the function estimates ~19,166 points, still
    within the system limit.

    Returns
    -------
    int
        Estimated number of buffer points to allocate for the scan.
    """
    if rate is None or rate <= 0:
        raise ValueError(f"Invalid beam rate: {rate}. Must be a positive number.")

    def _log_range():
        return np.log10(_MAX_BEAM_RATE) - np.log10(_MIN_BEAM_RATE)

    def _rate_factor(rate):
        return (np.log10(rate) - np.log10(_MIN_BEAM_RATE)) / _log_range()

    def _fudge(rate):
        return 1.5 - 0.4 * _rate_factor(rate)

    def _n_measurements(pulses, rate):
        return int(pulses * 3 * _fudge(rate) + rate / 6)

    return _n_measurements(pulses, rate)
