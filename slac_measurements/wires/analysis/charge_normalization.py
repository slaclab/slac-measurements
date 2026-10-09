import numpy as np

from slac_measurements.wires.collection.results import WireMeasurementCollectionResult


_LOW_CHARGE_THRESHOLD = 1e7


def compute_charge_normalization(
    collection_result: WireMeasurementCollectionResult,
    toroid: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute per-pulse charge normalization factors and validity mask.

    Normalizes detector signals to the mean charge level, removing
    pulse-to-pulse charge jitter. Pulses below the low-charge threshold
    are masked out of downstream fitting.

    Parameters
    ----------
    collection_result : WireMeasurementCollectionResult
        Raw collection result containing toroid TMIT data in raw_data.
    toroid : str, optional
        Toroid device name to use for normalization. If not provided,
        defaults to the first charge toroid from metadata that has data
        in raw_data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        (normalization_factors, valid_mask, resolved_toroid):
        - normalization_factors: per-pulse scale factor (mean_charge / charge[i]).
          Set to 1.0 for invalid or NaN pulses.
        - valid_mask: boolean array, True where charge > threshold or isnan.
        - resolved_toroid: the toroid device name actually used.
    """
    toroid_name = _resolve_toroid(collection_result, toroid)
    charge = _extract_charge_data(collection_result.raw_data, toroid_name)

    valid_mask = (charge > _LOW_CHARGE_THRESHOLD) | np.isnan(charge)

    valid_finite = valid_mask & ~np.isnan(charge)
    mean_charge = np.nanmean(charge[valid_finite]) if np.any(valid_finite) else 1.0

    factors = np.where(valid_finite, mean_charge / charge, 1.0)

    return factors, valid_mask, toroid_name


def _resolve_toroid(
    collection_result: WireMeasurementCollectionResult,
    toroid: str | None,
) -> str:
    """Resolve which toroid device to use for charge normalization.

    Parameters
    ----------
    collection_result : WireMeasurementCollectionResult
        Collection result with raw_data and metadata.
    toroid : str or None
        Explicit toroid name, or None to auto-resolve.

    Returns
    -------
    str
        Resolved toroid device name.

    Raises
    ------
    ValueError
        If the specified toroid is not in raw_data, or no charge toroid
        data is available.
    """
    raw_data = collection_result.raw_data

    if toroid is not None:
        if toroid not in raw_data:
            raise ValueError(
                f"Specified charge toroid '{toroid}' not found in raw_data. "
                f"Available keys: {list(raw_data.keys())}"
            )
        return toroid

    charge_toroids = getattr(collection_result.metadata, "charge_toroids", None)
    if not charge_toroids:
        raise ValueError(
            "No charge_toroids defined in metadata and no explicit toroid specified."
        )

    for name in charge_toroids:
        if name in raw_data:
            return name

    raise ValueError(
        f"No charge toroid data found in raw_data. "
        f"Expected one of {charge_toroids}, "
        f"available keys: {list(raw_data.keys())}"
    )


def _extract_charge_data(raw_data: dict, toroid_name: str) -> np.ndarray:
    """Extract 1D TMIT charge array from raw_data.

    Handles two formats:
    - IM-type toroids: raw_data[name] is a 1D ndarray of TMIT values
    - BPM-type toroids: raw_data[name] is a dict with a "tmit" key

    Parameters
    ----------
    raw_data : dict
        Raw data dictionary from collection result.
    toroid_name : str
        Device name key in raw_data.

    Returns
    -------
    np.ndarray
        1D array of per-pulse charge (TMIT) values.

    Raises
    ------
    ValueError
        If the toroid data cannot be extracted.
    """
    if toroid_name not in raw_data:
        raise ValueError(f"Toroid '{toroid_name}' not found in raw_data.")

    data = raw_data[toroid_name]

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, dict):
        if "tmit" in data:
            return np.asarray(data["tmit"])
        raise ValueError(
            f"BPM-type toroid '{toroid_name}' has no 'tmit' key in raw_data. "
            f"Available keys: {list(data.keys())}"
        )

    raise ValueError(f"Unexpected data type for toroid '{toroid_name}': {type(data)}")
