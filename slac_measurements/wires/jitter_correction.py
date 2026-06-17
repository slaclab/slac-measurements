import numpy as np

from slac_measurements.wires.collection_results import WireMeasurementCollectionResult


_DISPERSION_THRESHOLD = 1e-4


def compute_jitter(
    collection_result: WireMeasurementCollectionResult,
    beampath: str,
    physics_model: str = "BLEM",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pulse beam jitter at the wire from BPM data.

    Uses BPM position data and transport matrices to reconstruct beam
    position jitter at the wire location via least-squares orbit fit.
    The dispersion (energy) term is automatically included when the
    R16/R36 elements are non-negligible.

    Parameters
    ----------
    collection_result : WireMeasurementCollectionResult
        Raw collection result containing wire positions and BPM x/y buffer data.
    beampath : str
        Beam path identifier for the model (e.g., "SC_HXR", "SC_DIAG0").
    physics_model : str
        Model source for R-matrix retrieval. Default "BLEM".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (jitter_x, jitter_y) per-pulse beam position jitter at the wire
        in um, one value per buffer pulse.

    Raises
    ------
    ValueError
        If no BPM data is found in the collection result or if fewer than
        2 BPMs have valid data.
    """
    raw_data = collection_result.raw_data

    bpm_x_data, bpm_y_data, bpm_names = _extract_bpm_data(raw_data)

    if len(bpm_names) < 2:
        raise ValueError(
            f"Jitter correction requires at least 2 BPMs with valid data, "
            f"found {len(bpm_names)}."
        )

    wire_name = collection_result.metadata.wire_name

    rmat_x, rmat_y = get_jitter_rmat(wire_name, bpm_names, beampath, physics_model)

    return _compute_orbit_fit(bpm_x_data, bpm_y_data, rmat_x, rmat_y)


def get_jitter_rmat(
    wire_name: str,
    bpm_names: list[str],
    beampath: str,
    physics_model: str = "BLEM",
) -> tuple[np.ndarray, np.ndarray]:
    """Get R-matrices from wire to each BPM for orbit fitting.

    Always fetches R16/R36 (dispersion) columns. The orbit fit will
    automatically drop them if they are below threshold.

    Parameters
    ----------
    wire_name : str
        Wire device name (MAD element name).
    bpm_names : list[str]
        BPM element names (MAD names, e.g., ["BPM10", "BPM11"]).
    beampath : str
        Beam path identifier for the model.
    physics_model : str
        Model source. Default "BLEM".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rmat_x, rmat_y) where each is [N_bpms x 3] containing
        [R11, R12, R16] and [R33, R34, R36] respectively.
    """
    from meme.model import Model

    model = Model(beampath, model_source=physics_model, use_design=False)

    rmat_x = np.zeros((len(bpm_names), 3))
    rmat_y = np.zeros((len(bpm_names), 3))

    for i, bpm_name in enumerate(bpm_names):
        rmat_6x6 = model.get_rmat(from_device=wire_name, to_device=bpm_name)

        rmat_x[i, 0] = rmat_6x6[0, 0]  # R11
        rmat_x[i, 1] = rmat_6x6[0, 1]  # R12
        rmat_x[i, 2] = rmat_6x6[0, 5]  # R16

        rmat_y[i, 0] = rmat_6x6[2, 2]  # R33
        rmat_y[i, 1] = rmat_6x6[2, 3]  # R34
        rmat_y[i, 2] = rmat_6x6[2, 5]  # R36

    return rmat_x, rmat_y


def _extract_bpm_data(
    raw_data: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract BPM x/y position data from raw_data dictionary.

    BPM entries are keyed by device name (starting with 'BPM') and contain
    a dict with 'x' and 'y' buffer arrays.

    Returns
    -------
    tuple
        (bpm_x_array, bpm_y_array, bpm_names) where arrays are
        [N_bpms x N_pulses] and bpm_names are device names.
    """
    bpm_keys = sorted(k for k in raw_data if k.startswith("BPM"))

    bpm_names = []
    x_arrays = []
    y_arrays = []

    for name in bpm_keys:
        bpm_data = raw_data[name]

        if not isinstance(bpm_data, dict):
            continue

        x_arr = bpm_data.get("x")
        y_arr = bpm_data.get("y")

        if x_arr is None or y_arr is None:
            continue
        if np.all(np.isnan(x_arr)) or np.all(np.isnan(y_arr)):
            continue

        bpm_names.append(name)
        x_arrays.append(x_arr)
        y_arrays.append(y_arr)

    if not bpm_names:
        raise ValueError("No valid BPM position data found in collection result.")

    bpm_x_data = np.array(x_arrays)  # [N_bpms x N_pulses]
    bpm_y_data = np.array(y_arrays)  # [N_bpms x N_pulses]

    return bpm_x_data, bpm_y_data, bpm_names


def _compute_orbit_fit(
    bpm_x_data: np.ndarray,
    bpm_y_data: np.ndarray,
    rmat_x: np.ndarray,
    rmat_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute beam position at wire from BPM readings via least-squares orbit fit.

    The dispersion column (R16/R36) is automatically excluded from the
    fit if its maximum absolute value is below _DISPERSION_THRESHOLD,
    matching the legacy MATLAB behavior in beamAnalysis_orbitFit.

    Parameters
    ----------
    bpm_x_data : np.ndarray
        BPM x positions [N_bpms x N_pulses] in mm.
    bpm_y_data : np.ndarray
        BPM y positions [N_bpms x N_pulses] in mm.
    rmat_x : np.ndarray
        X-plane transport matrix [N_bpms x 3] (R11, R12, R16).
    rmat_y : np.ndarray
        Y-plane transport matrix [N_bpms x 3] (R33, R34, R36).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (jitter_x, jitter_y) - reconstructed beam position jitter at wire
        for each pulse, in um (matching wire position units).
    """
    # Drop dispersion column if negligible
    if np.max(np.abs(rmat_x[:, 2])) < _DISPERSION_THRESHOLD:
        rmat_x = rmat_x[:, :2]
    if np.max(np.abs(rmat_y[:, 2])) < _DISPERSION_THRESHOLD:
        rmat_y = rmat_y[:, :2]

    # Compute deviations from mean (mm)
    delta_x = bpm_x_data - bpm_x_data.mean(axis=1, keepdims=True)
    delta_y = bpm_y_data - bpm_y_data.mean(axis=1, keepdims=True)

    # Convert mm -> m for orbit fit
    delta_x_m = delta_x * 1e-3
    delta_y_m = delta_y * 1e-3

    # Least-squares fit: rmat @ orbit = delta_bpm for each pulse
    # orbit_x shape: [n_params x N_pulses], orbit_x[0] is x position at wire
    orbit_x, _, _, _ = np.linalg.lstsq(rmat_x, delta_x_m, rcond=None)
    orbit_y, _, _, _ = np.linalg.lstsq(rmat_y, delta_y_m, rcond=None)

    # Position at wire (first element of orbit vector), convert m -> um
    jitter_x_um = orbit_x[0, :] * 1e6
    jitter_y_um = orbit_y[0, :] * 1e6

    return jitter_x_um, jitter_y_um
