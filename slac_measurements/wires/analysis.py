import importlib
import numpy as np
from pydantic import ConfigDict, PrivateAttr
from scipy.ndimage import median_filter
from skimage.filters import threshold_triangle
import warnings
from typing import Literal

import slac_measurements.beam_profile
from slac_measurements.wires.coordinates import stage_to_beam, beam_to_stage
from slac_measurements.wires.jitter_correction import compute_jitter
from slac_measurements.wires.analysis_results import (
    DetectorFit,
    DetectorProfileMeasurement,
    FitResult,
    ProfileMeasurement,
    WireMeasurementAnalysisResult,
)

FittingMethod = Literal["gaussian", "asymmetric_gaussian", "super_gaussian"]


class WireMeasurementAnalysis(slac_measurements.beam_profile.BeamProfileAnalysis):
    """
    Organizes wire-scan data by profile, fits curves, and extracts
    centroid, RMS size, and amplitude per detector.

    Attributes:
        collection_result: Raw measurement data from wire scan.
        fitting_method: Fitting method to use for profile analysis.
                       Options: 'gaussian', 'asymmetric_gaussian', 'super_gaussian'.
                       Default: 'gaussian'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fitting_method: FittingMethod = "gaussian"
    _jitter_x: np.ndarray | None = PrivateAttr(default=None)
    _jitter_y: np.ndarray | None = PrivateAttr(default=None)

    def analyze(
        self,
        rms_detector: str | None = None,
        jitter_correction: bool = False,
        physics_model: str = "BLEM",
    ) -> WireMeasurementAnalysisResult:
        """
        Fit profiles and extract RMS beam sizes.

        Parameters
        ----------
        rms_detector : str, optional
            Detector used for RMS extraction; defaults to
            ``collection_result.metadata.default_detector``.
        jitter_correction : bool
            If True, compute orbit-fit jitter correction from BPM data
            and subtract per-profile in beam coordinates before fitting.
        physics_model : str
            Model source for R-matrix retrieval. Default "BLEM".

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results, RMS sizes, and organized profile data.
        """

        if jitter_correction:
            beampath = self.collection_result.metadata.beampath
            self._jitter_x, self._jitter_y = compute_jitter(
                self.collection_result, beampath, physics_model
            )

        profile_indices = self._get_profile_range_indices()
        profile_measurements = self._organize_data_by_profile(profile_indices)

        fit_result = self._fit_data_by_profile(
            profile_measurements=profile_measurements
        )
        rms_sizes = self._get_rms_sizes(fit_result, detector=rms_detector)

        metadata = self.collection_result.metadata
        metadata.rms_detector = (
            rms_detector if rms_detector is not None else metadata.default_detector
        )

        jitter_rms = None
        if jitter_correction and self._jitter_x is not None:
            jitter_rms = (
                float(np.std(self._jitter_x)),
                float(np.std(self._jitter_y)),
            )

        return WireMeasurementAnalysisResult(
            fit_result=fit_result,
            rms_sizes=rms_sizes,
            collection_result=self.collection_result,
            metadata=metadata,
            profiles=profile_measurements,
            fitting_method=self.fitting_method,
            jitter_corrected=jitter_correction,
            jitter_rms=jitter_rms,
        )

    def _create_detector_measurement(
        self, device_name: str, data_slice: np.ndarray
    ) -> DetectorProfileMeasurement:
        """Create a DetectorProfileMeasurement object for a given device and data slice."""

        def _get_units_for_device(device_name: str) -> str:
            """Get the appropriate units for a given device based on its name."""
            if device_name == "TMITLOSS":
                return "% beam loss"
            return "counts"

        return DetectorProfileMeasurement(
            values=data_slice,
            units=_get_units_for_device(device_name),
            label=device_name,
        )

    def _fit_data_by_profile(self, profile_measurements) -> dict[str, FitResult]:
        """Fit detector data for each profile using the configured method."""

        profiles = list(profile_measurements.keys())
        detectors = list(self.collection_result.metadata.detectors)

        fit_result = {
            profile: self._fit_profile(profile_measurements, profile, detectors)
            for profile in profiles
        }

        return fit_result

    def _fit_profile(
        self, profile_measurements, profile: str, detectors: list
    ) -> FitResult:
        """Fit all detectors for one profile and return fit results."""

        def _convert_stage_to_beam_coords(
            profile: str, positions: np.ndarray
        ) -> np.ndarray:
            """Convert stage positions to beam coordinates for a given profile."""
            return stage_to_beam(
                positions, profile, self.collection_result.metadata.install_angle
            )

        def _fit_detector_in_profile(
            x_beam: np.ndarray, detector_signal: np.ndarray, profile: str
        ) -> DetectorFit:
            """Fit one detector signal for a profile and return DetectorFit."""
            # Dynamically import the fitting module based on fitting_method
            fitting_module = importlib.import_module(
                f"slac_measurements.fitting.{self.fitting_method}"
            )

            peak_window = _peak_window(x=x_beam, y=detector_signal)

            # Get fit parameters - use_prior parameter only exists for asymmetric_gaussian
            if self.fitting_method == "asymmetric_gaussian":
                fp = fitting_module.fit(
                    pos=peak_window[0], data=peak_window[1], use_prior=False
                )
            else:
                fp = fitting_module.fit(pos=peak_window[0], data=peak_window[1])

            # Convert mean from beam coordinates back to stage coordinates
            mean_stage = beam_to_stage(
                fp["mean"], profile, self.collection_result.metadata.install_angle
            )

            # Generate fit curve (in beam coordinates)
            # Build kwargs dynamically to handle different fit types
            curve_params = {k: v for k, v in fp.items() if k != "error"}
            fit_curve = fitting_module.curve(x=peak_window[0], **curve_params)

            return DetectorFit(
                mean=mean_stage,
                sigma=fp["sigma"],
                amplitude=fp["amp"],
                offset=fp["off"],
                curve=fit_curve,
                positions=peak_window[0],
            )

        def _peak_window(
            x: np.ndarray, y: np.ndarray, n_stds: float = 8, filter_size: int = 5
        ) -> tuple:
            """
            Extract a peak-centered window from 1D detector data.

            Parameters:
                x (np.ndarray): Position data.
                y (np.ndarray): Detector signal values.
                n_stds (float): Window half-width in RMS units (default 8).
                filter_size (int): Median filter size (default 5).

            Returns:
                tuple: (windowed_x, windowed_y)
            """

            x = np.asarray(x)
            y = np.asarray(y)

            # Smooth the signal
            y_filtered = median_filter(y, size=filter_size)

            # Apply triangle threshold
            threshold = threshold_triangle(y_filtered)
            y_thresholded = np.clip(y_filtered - threshold, 0, None)

            # Find centroid and RMS of thresholded signal
            if y_thresholded.sum() == 0:
                # Fallback to simple peak finding if no signal above threshold
                msg = "No signal above threshold. Using simple peak finding for window."
                warnings.warn(msg, UserWarning, stacklevel=2)
                i = np.argmax(y)
                center = x[i]
                rms = (x[-1] - x[0]) / 4  # Default quarter-range
            else:
                # Weighted centroid
                weights = y_thresholded
                center = np.sum(x * weights) / weights.sum()
                # Weighted RMS
                rms = np.sqrt(np.sum(weights * (x - center) ** 2) / weights.sum())

            # Define window as center ± n_stds * rms
            left_bound = center - n_stds * rms
            right_bound = center + n_stds * rms

            # Find indices
            left = np.searchsorted(x, left_bound, side="left")
            right = np.searchsorted(x, right_bound, side="right")

            # Clip to valid range
            left = max(0, left)
            right = min(len(y) - 1, right)

            return x[left : right + 1], y[left : right + 1]

        profile_data = profile_measurements[profile]
        x_stage = profile_data.positions
        x_beam = _convert_stage_to_beam_coords(profile, x_stage)

        if self._jitter_x is not None:
            x_beam = self._apply_jitter_correction(
                x_beam, profile, profile_data.profile_indices
            )

        detector_fits = {}
        for detector_name in detectors:
            if detector_name not in profile_data.detectors:
                continue

            detector_fits[detector_name] = _fit_detector_in_profile(
                x_beam, profile_data.detectors[detector_name].values, profile
            )

        return FitResult(detectors=detector_fits)

    def _apply_jitter_correction(
        self, x_beam: np.ndarray, profile: str, indices: np.ndarray
    ) -> np.ndarray:
        """Subtract per-pulse beam jitter in beam coordinates.

        Matches the legacy MATLAB behavior: jitter is subtracted after
        stage-to-beam conversion, so x-profile subtracts jitter_x and
        y-profile subtracts jitter_y directly in beam-space (um).
        """
        if self._jitter_x is None or self._jitter_y is None:
            return x_beam

        jx = self._jitter_x[indices]
        jy = self._jitter_y[indices]

        if profile == "x":
            return x_beam - jx
        elif profile == "y":
            return x_beam - jy
        else:
            install_angle = self.collection_result.metadata.install_angle
            rad = np.radians(install_angle)
            return x_beam - (jx * np.sin(rad) + jy * np.cos(rad))

    @staticmethod
    def _get_monotonic_indices(
        position_data: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Return the longest contiguous non-decreasing segment of indices."""

        if len(indices) <= 1:
            return indices

        pos = position_data[indices]

        best_start = 0
        best_end = 1
        run_start = 0

        for i in range(1, len(pos)):
            if pos[i] < pos[i - 1]:
                run_end = i
                if (run_end - run_start) >= (best_end - best_start):
                    best_start = run_start
                    best_end = run_end
                run_start = i

        run_end = len(pos)
        if (run_end - run_start) >= (best_end - best_start):
            best_start = run_start
            best_end = run_end

        # If the selected run starts immediately after a rollback, drop the
        # first point (local trough) so the kept segment starts with forward motion.
        if (
            best_start > 0
            and pos[best_start] < pos[best_start - 1]
            and (best_end - best_start) > 1
        ):
            best_start += 1

        return indices[best_start:best_end]

    def _get_profile_range_indices(self) -> dict[str, np.ndarray]:
        """Finds sequential scan indices within each profile's position range."""

        def _get_indices_in_range(
            position_data: np.ndarray, min_pos: float, max_pos: float
        ) -> np.ndarray:
            """Return indices of position data within a given range."""

            return np.where((position_data >= min_pos) & (position_data <= max_pos))[0]

        def _validate_position_data(position_data: np.ndarray) -> None:
            """Validates the position data to ensure it is suitable for analysis."""

            if position_data.min() == position_data.max():
                msg = (
                    "Min and max position are the same. Check scan data and collection."
                )
                raise RuntimeError(msg)

        def _get_profile_range(profile: str) -> tuple:
            """Get the (min, max) range for a given profile."""

            return self.collection_result.metadata.scan_ranges[profile]

        def _check_range_in_position(
            position_data: np.ndarray, profile: str, profile_range: tuple
        ) -> None:
            """Check if the position data covers the expected range for a profile."""

            if position_data.max() < profile_range[0]:
                msg = (
                    f"Scan did not reach expected {profile} profile range "
                    f"{profile_range}. Check scan data and collection."
                )
                raise RuntimeError(msg)

        position_data = self.collection_result.raw_data[
            self.collection_result.metadata.wire_name
        ]

        # Single validation pass
        _validate_position_data(position_data)

        profile_indices = {}
        for p in self.collection_result.metadata.active_profiles:
            profile_range = _get_profile_range(p)
            _check_range_in_position(position_data, p, profile_range)

            indices = _get_indices_in_range(
                position_data, profile_range[0], profile_range[1]
            )

            monotonic_indices = self._get_monotonic_indices(position_data, indices)

            profile_indices[p] = monotonic_indices

        return profile_indices

    def _get_rms_sizes(
        self, fit_result: dict, detector: str | None = None
    ) -> tuple[float | None, float | None]:
        """
        Extract x/y RMS sizes from fit results.

        Parameters:
            fit_result (dict): Fit results from fit_data_by_profile().
            detector (str | None): Detector to use; defaults to metadata default.

        Returns:
            tuple[float | None, float | None]: (x_rms, y_rms) in meters,
            or (None, None) if x/y are unavailable.
        """

        available_detectors = self.collection_result.metadata.detectors
        selected_detector = (
            self.collection_result.metadata.default_detector
            if detector is None
            else detector
        )

        if selected_detector not in available_detectors:
            raise ValueError(
                f"Detector '{selected_detector}' is not available in "
                f"metadata.detectors={available_detectors}."
            )

        x_rms = None
        y_rms = None

        if "x" in fit_result:
            x_fit = fit_result["x"].detectors[selected_detector]
            x_rms = x_fit.sigma

        if "y" in fit_result:
            y_fit = fit_result["y"].detectors[selected_detector]
            y_rms = y_fit.sigma

        return (x_rms, y_rms)

    def _organize_data_by_profile(
        self, profile_indices: dict[str, np.ndarray]
    ) -> dict[str, ProfileMeasurement]:
        """
        Organizes detector data by scan profile for each device.

        Returns:
            dict: Nested dict with profiles as keys and device
                  data per profile.
        """

        def _create_profile_measurement(
            positions: np.ndarray, detectors: dict, profile_indices: np.ndarray
        ) -> ProfileMeasurement:
            return ProfileMeasurement(
                positions=positions,
                detectors=detectors,
                profile_indices=profile_indices,
            )

        profile_measurements = {}
        devices = [
            *self.collection_result.metadata.detectors,
            self.collection_result.metadata.wire_name,
        ]
        for profile, index in profile_indices.items():
            detectors = {}
            positions = None
            for d_n in devices:
                if d_n not in self.collection_result.raw_data:
                    continue
                data_slice = self.collection_result.raw_data[d_n][index]

                if d_n == self.collection_result.metadata.wire_name:
                    positions = data_slice
                else:
                    detectors[d_n] = self._create_detector_measurement(d_n, data_slice)

            profile_measurements[profile] = _create_profile_measurement(
                positions, detectors, index
            )

        return profile_measurements
