from __future__ import annotations

from typing import NamedTuple


def resolve_beampath_key(beampath: str) -> str:
    """Derive the CU/SC timing key from a beampath string."""
    return "CU" if beampath.startswith("CU") else "SC"


def pick_by_beampath(
    detector_container: str | list[str] | dict[str, str] | dict[str, list[str]],
    beampath: str,
    fallback: str | list[str] | None = None,
) -> str | list[str] | None:
    """If *detector_container* is a dict keyed by CU/SC, return the entry for the active beampath; otherwise pass through."""
    if isinstance(detector_container, dict):
        timing = resolve_beampath_key(beampath)
        return detector_container.get(timing, fallback)
    return detector_container


class DetectorConfig(NamedTuple):
    """Result of resolving Wire metadata detectors for a beampath."""

    names: list[str]
    default: str


def resolve_detectors(
    detectors: list[str] | dict[str, list[str]],
    default_detector: str | dict[str, str],
    beampath: str,
) -> DetectorConfig:
    """Resolve detector timing dicts and determine the default detector
    for a given beampath.

    Parameters
    ----------
    detectors
        Detector list from Wire metadata (flat or timing-keyed).
    default_detector
        Default detector from Wire metadata (plain or timing-keyed).
    beampath
        Active beampath string (e.g. ``"CU_HXR"``).

    Returns
    -------
    DetectorConfig
        ``names`` — detector MAD names for the active beampath.
        ``default`` — default detector MAD name.
    """
    beampath_key = resolve_beampath_key(beampath)

    if isinstance(detectors, dict):
        names = list(detectors.get(beampath_key, []))
    else:
        names = list(detectors)

    if isinstance(default_detector, dict):
        default = default_detector.get(beampath_key, "")
    else:
        default = default_detector

    if not default:
        if not names:
            raise RuntimeError(
                "No detectors available; cannot determine default detector."
            )
        default = names[0]

    return DetectorConfig(names=names, default=default)
