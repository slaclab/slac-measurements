import numpy as np


def wire_angle_scales(install_angle_deg: float) -> dict[str, float]:
    """Scale factors for converting between stage and beam coordinates.

    Parameters
    ----------
    install_angle_deg : float
        Wire installation angle in degrees.

    Returns
    -------
    dict[str, float]
        {"x": sin(angle), "y": cos(angle), "u": 1.0}
    """
    rad = np.radians(install_angle_deg)
    return {"x": np.sin(rad), "y": np.cos(rad), "u": 1.0}


def stage_to_beam(
    positions: np.ndarray, profile: str, install_angle_deg: float
) -> np.ndarray:
    """Convert wire stage positions to beam coordinates for a profile."""
    scales = wire_angle_scales(install_angle_deg)
    return positions * abs(scales[profile])


def beam_to_stage(
    positions: np.ndarray, profile: str, install_angle_deg: float
) -> np.ndarray:
    """Convert beam coordinates back to wire stage positions for a profile."""
    scales = wire_angle_scales(install_angle_deg)
    return positions / abs(scales[profile])


def xy_to_stage(
    jitter_x: np.ndarray, jitter_y: np.ndarray, install_angle_deg: float
) -> np.ndarray:
    """Project x/y beam jitter onto the wire stage direction."""
    scales = wire_angle_scales(install_angle_deg)
    return jitter_x * scales["x"] + jitter_y * scales["y"]
