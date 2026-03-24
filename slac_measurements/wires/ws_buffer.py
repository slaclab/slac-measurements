import edef
import logging
import os
import numpy as np


_BUFFER_NAME = "LCLS Tools Wire Scan"
_MAX_BEAM_RATE = 16000
_MIN_BEAM_RATE = 10


class BufferError(Exception):
    pass


def reserve_buffer(
    beampath: str,
    pulses: int,
    beam_rate: int,
    name: str = _BUFFER_NAME,
    destination_mode: str = "Inclusion",
    logger: logging.Logger = None,
):
    user = os.getlogin()
    if logger:
        logger.info("Reserving buffer...")

    if beampath.startswith("SC"):
        return _reserve_bsa_buffer(
            name=name,
            beampath=beampath,
            user=user,
            n_measurements=_calculate_buffer_points(pulses, beam_rate),
            destination_mode=destination_mode,
            logger=logger,
        )

    elif beampath.startswith("CU"):
        return _reserve_edef_buffer(
            name=name,
            user=user,
            n_measurements=_calculate_buffer_points(pulses, beam_rate),
            logger=logger,
        )

    else:
        raise BufferError(f"Unrecognized beampath: {beampath}")


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

    # 16000 max rate, 10 min rate
    def _log_range():
        return np.log10(_MAX_BEAM_RATE) - np.log10(_MIN_BEAM_RATE)

    def _rate_factor(rate):
        return (np.log10(rate) - np.log10(_MIN_BEAM_RATE)) / _log_range()

    def _fudge(rate):
        return 1.5 - 0.4 * _rate_factor(rate)  # Fudge the calculation by 1.1 to 1.5

    def _n_measurements(pulses, rate):
        return int(pulses * 3 * _fudge(rate) + rate / 6)

    return _n_measurements(pulses, rate)


def _reserve_bsa_buffer(
    name: str,
    beampath: str,
    user: str,
    n_measurements: int,
    destination_mode: str,
    logger: logging.Logger = None,
):
    if destination_mode not in ["Disable", "Exclusion", "Inclusion"]:
        raise BufferError(f"Invalid destination mode: {destination_mode}")

    buf = edef.BSABuffer(name=name, user=user)
    buf.n_measurements = n_measurements
    buf.destination_mode = destination_mode
    buf.clear_masks()
    buf.destination_masks = [beampath]
    if logger:
        logger.info("Reserved BSA Buffer %s.", buf.number)
    return buf


def _reserve_edef_buffer(
    name: str,
    user: str,
    n_measurements: int,
    logger: logging.Logger = None,
):
    buf = edef.EventDefinition(name=name, user=user)
    buf.n_measurements = n_measurements
    if logger:
        logger.info("Reserved eDef Buffer %s.", buf.number)
    return buf
