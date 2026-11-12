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


def effective_rate(beam_rate: float, rate_limit_hz: float | None) -> float:
    """Rate (Hz) at which a buffer records shots, given its mask rate limit."""
    if rate_limit_hz is None:
        return beam_rate
    return min(beam_rate, rate_limit_hz)


def reserve_buffer(
    beampath: str,
    pulses: int,
    beam_rate: int,
    name: str = _BUFFER_NAME,
    logger: logging.Logger | None = None,
    area: str | None = None,
    wire_follows_buffer_rate: bool = False,
) -> Buffer:
    """Reserve a timing buffer sized for a wire scan.

    ``area`` is the wire's machine area. For CU buffers it selects the BYKIK
    rate mask (applied in LTUH / LTUS only), which can lower the rate the
    buffer records at below ``beam_rate``.

    ``wire_follows_buffer_rate`` says how the wire speed is set:

    - False (OTF): the motion IOC paces the wire for ``beam_rate``. The scan
      lasts as long as it would unmasked, so the buffer is sized for that
      duration at the lower recording rate.
    - True (step): the wire speed is computed from the recording rate, so each
      profile still gets ``pulses`` shots and the buffer is sized for that rate.
    """
    import slac_timing.masks

    user = _get_username()
    if logger:
        logger.info("Reserving buffer...")

    # Read the BYKIK rate before reserving so the buffer can be sized for it.
    # create_buffer reads it again to choose the mask; checked below.
    _, expected_limit_hz = slac_timing.masks.bykik_inclusion_masks(beampath, area)
    buffer_rate = effective_rate(beam_rate, expected_limit_hz)
    wire_rate = buffer_rate if wire_follows_buffer_rate else beam_rate

    buf = create_buffer(
        beampath=beampath,
        n_measurements=_calculate_buffer_points(pulses, wire_rate, buffer_rate),
        user=user,
        name=name,
        area=area,
    )

    actual_limit_hz = getattr(buf, "rate_limit_hz", None)
    if actual_limit_hz != expected_limit_hz:
        buf.release()
        raise BufferError(
            f"BYKIK rate changed during reservation (expected limit "
            f"{expected_limit_hz} Hz, got {actual_limit_hz} Hz). Retry the scan."
        )

    if logger:
        logger.info(
            "Reserved timing buffer %s with %s points.",
            buf.number,
            buf.n_measurements,
        )
        if actual_limit_hz is not None:
            logger.info(
                "Buffer %s limited to %s Hz by BYKIK rate mask.",
                buf.number,
                actual_limit_hz,
            )

    return buf


def _calculate_buffer_points(pulses, rate, buffer_rate=None) -> int:
    """
    Determine the number of buffer points for a wire scan.

    ``rate`` is the rate the wire speed is paced for. ``buffer_rate`` is the
    rate the buffer records at (defaults to ``rate``). When the buffer records
    slower than the wire is paced, e.g. a 30 Hz BYKIK mask on a 120 Hz OTF
    scan, the scan duration is unchanged, so the point count is scaled by
    ``buffer_rate / rate``.

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

    n_points = _n_measurements(pulses, rate)
    if buffer_rate is None or buffer_rate >= rate:
        return n_points
    return int(np.ceil(n_points * buffer_rate / rate))
