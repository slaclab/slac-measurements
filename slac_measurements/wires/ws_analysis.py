import numpy as np
from pydantic import ConfigDict
import warnings
from typing import Literal

import slac_measurements.beam_profile
from slac_measurements.wires.ws_analysis_results import (
    DetectorFit,
    DetectorProfileMeasurement,
    FitResult,
    ProfileMeasurement,
    WireMeasurementAnalysisResult,
)


_MIN_POS_FUDGE = 1.001  # 0.1% fudge factor to ensure we capture all relevant data points in profile range


class WireMeasurementAnalysis(slac_measurements.beam_profile.BeamProfileAnalysis):
    """
    Analyzes wire scan data: organizes by profile, fits beam profile curves,
    extracts beam parameters.

    Takes raw wire measurement data and performs curve fitting to extract
    centroid, RMS size, and amplitude for each detector and profile.

    Attributes:
        collection_result: Raw measurement data from wire scan.
        fitting_method: Fitting method to use for profile analysis.
                       Options: 'gaussian', 'asymmetric_gaussian', 'super_gaussian'.
                       Default: 'gaussian'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fitting_method: Literal["gaussian", "asymmetric_gaussian", "super_gaussian"] = "gaussian"

    def analyze(self) -> WireMeasurementAnalysisResult:
        """
        Organize data by profile, fit beam profile curves, extract RMS sizes.

        Uses the configured fitting method (set via fitting_method parameter)
        to fit each detector's data. Available methods: 'gaussian',
        'asymmetric_gaussian', 'super_gaussian'.

        Returns
        -------
        WireMeasurementAnalysisResult
            Fit results per profile and detector, RMS beam sizes, and
            organized data.
        """
        profile_indices = self._get_profile_range_indices()
        profile_measurements = self._organize_data_by_profile(profile_indices)

        fit_result = self._fit_data_by_profile(profile_measurements=profile_measurements)
        rms_sizes = self._get_rms_sizes(fit_result)

        return WireMeasurementAnalysisResult(
            fit_result=fit_result,
            rms_sizes=rms_sizes,
            collection_result=self.collection_result,
            metadata=self.collection_result.metadata,
            profiles=profile_measurements,
        )

    def _create_detector_measurement(
        self, device_name: str, data_slice: np.ndarray
    ) -> DetectorProfileMeasurement:
        """
        Create a DetectorProfileMeasurement object for a given device and
        data slice.
        """
        def _get_units_for_device(device_name: str) -> str:
            """Get the appropriate units for a given device based on its name."""
            if device_name == "TMITLOSS":
                return "%% beam loss"
            return "counts"

        return DetectorProfileMeasurement(
            values=data_slice, units=_get_units_for_device(device_name), label=device_name
        )

    def _fit_data_by_profile(self, profile_measurements) -> dict:
        """
        Fit detector data for each profile and device using the configured fitting method.

        Applies curve fitting to x, y, and u projections for all detectors
        in the measurement result.

        Returns:
            dict: Fit results organized by profile and detector.
        """
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
        """
        Fit all detectors within a single profile.

        Parameters:
            profile (str): Profile name ('x', 'y', or 'u').
            detectors (list): List of detector names.

        Returns:
            FitResult: Fit results for all detectors in the profile.
        """
        def _convert_stage_to_beam_coords(profile: str, positions: np.ndarray) -> np.ndarray:
            """Convert stage positions to beam coordinates for a given profile."""
            scale = _extract_wire_angle()
            return positions * abs(scale[profile])

        def _extract_wire_angle() -> dict:
            """
            Extract the wire install angle (in radians) for coordinate conversion.
            """
            rad = np.deg2rad(self.collection_result.metadata.install_angle)
            return {"x": np.sin(rad), "y": np.cos(rad), "u": 1.0}

        def _fit_detector_in_profile(x_beam: np.ndarray, detector_signal: np.ndarray, profile: str) -> DetectorFit:
            """
            Fit a single detector signal within a profile using the configured fitting method.

            Parameters:
                x_beam (np.ndarray): Position data in beam coordinates.
                detector_signal (np.ndarray): Detector signal values.
                profile (str): Profile name ('x', 'y', or 'u').

            Returns:
                DetectorFit: Fit parameters (mean in stage coords, others in beam coords) and curve.
            """
            import importlib

            # Dynamically import the fitting module based on fitting_method
            fitting_module = importlib.import_module(
                f"slac_measurements.fitting.{self.fitting_method}"
            )

            peak_window = _peak_window(x=x_beam, y=detector_signal)

            # Get fit parameters - use_prior parameter only exists for asymmetric_gaussian
            if self.fitting_method == "asymmetric_gaussian":
                fp = fitting_module.fit(pos=peak_window[0], data=peak_window[1], use_prior=False)
            else:
                fp = fitting_module.fit(pos=peak_window[0], data=peak_window[1])

            # Convert mean from beam coordinates back to stage coordinates
            scale = _extract_wire_angle()
            mean_stage = fp["mean"] / abs(scale[profile])

            # Generate fit curve (in beam coordinates)
            # Build kwargs dynamically to handle different fit types
            curve_kwargs = {
                "x": peak_window[0],
                "mean": fp["mean"],
                "sigma": fp["sigma"],
                "amp": fp["amp"],
                "off": fp["off"],
            }

            # Add method-specific parameters
            if self.fitting_method == "asymmetric_gaussian" and "skew" in fp:
                curve_kwargs["skew"] = fp["skew"]
            elif self.fitting_method == "super_gaussian" and "n" in fp:
                curve_kwargs["n"] = fp["n"]

            fit_curve = fitting_module.curve(**curve_kwargs)

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
            Extract peak window from 1D detector data using statistical windowing.

            Applies median filtering, triangle thresholding, and n-sigma windowing
            around the signal centroid.

            Parameters:
                x (np.ndarray): Position data.
                y (np.ndarray): Detector signal values.
                n_stds (float): Number of standard deviations for windowing.
                                Default is 8.
                filter_size (int): Median filter kernel size. Default is 5.

            Returns:
                tuple: (windowed_x, windowed_y, (left_idx, right_idx))
            """
            from scipy.ndimage import median_filter
            from skimage.filters import threshold_triangle

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

            return x[left : right + 1], y[left : right + 1], (left, right)

        profile_data = profile_measurements[profile]
        x_stage = profile_data.positions
        x_beam = _convert_stage_to_beam_coords(profile, x_stage)

        detector_fits = {}
        for detector_name in detectors:
            if detector_name not in profile_data.detectors:
                continue

            detector_fits[detector_name] = _fit_detector_in_profile(
                x_beam, profile_data.detectors[detector_name].values, profile
            )

        return FitResult(detectors=detector_fits)

    def _get_monotonic_indices(
        self, position_data: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """
        Return indices of position data that are monotonically non-decreasing.
        """
        def _mono_array(pos: np.ndarray) -> np.ndarray:
            """
            Boolean mask of monotonically non-decreasing data points
            Mask of values where difference between neighbors is > 0.
            """
            mono = True
            mono_mask = np.array(
                # Data point [i-1] is less than subsequent data point [i]
                # and that relationship was True for the previous pair
                # for all points
                [mono := (pos[i - 1] <= pos[i] and mono) for i in range(1, len(pos))],
                dtype=bool,
            )
            mono_mask = np.concatenate(([True], mono_mask))
            return mono_mask

        pos = position_data[indices]
        mono_mask = _mono_array(pos)
        return indices[mono_mask]

    def _get_profile_range_indices(self) -> dict:
        """
        Finds sequential scan indices within each profile's position range.

        Returns:
            dict: Profile keys ('x', 'y', 'u') with lists of index arrays.
        """
        def _get_indices_in_range(position_data: np.ndarray, min_pos: float, max_pos: float) -> np.ndarray:
            """Return indices of position data within a given range."""
            # When setting motor positions, encoder can wobble at inner range
            # bound, so we apply a small fudge factor to ensure we capture all relevant data points
            x = (min_pos * _MIN_POS_FUDGE)
            return np.where((position_data >= x) & (position_data <= max_pos))[0]

        def _validate_position_data(position_data: np.ndarray) -> None:
            """
            Validates the position data to ensure it is suitable for analysis.
            """
            if position_data.min() == position_data.max():
                msg = (
                    "Min and max position are the same. Check scan data "
                    "and collection. Exiting scan."
                )
                raise RuntimeError(msg)

        def _get_profile_range(profile: str) -> tuple:
            """Get the (min, max) range for a given profile."""
            # Ranges are stored in the collection metadata by profile name
            return self.collection_result.metadata.scan_ranges[profile]

        def _check_range_in_position(position_data: np.ndarray, profile: str, profile_range: tuple) -> None:
            """
            Check if the position data covers the expected range for a profile.
            """
            if position_data.max() < profile_range[0]:
                msg = (
                    f"Scan did not reach expected {profile} profile range "
                    f"{profile_range}. Check scan data and collection. "
                    f"Exiting scan."
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

    def _get_rms_sizes(self, fit_result: dict) -> tuple | None:
        """
        Extract RMS beam sizes from fit results.

        Computes RMS sizes from x and y profile fits using the
        default detector.

        Parameters:
            fit_result (dict): Fit results from fit_data_by_profile().

        Returns:
            tuple or None: (x_rms, y_rms) in meters, or None if both profiles
            not present.
        """
        if "x" in fit_result and "y" in fit_result:
            default_det = self.collection_result.metadata.default_detector
            x_fit = fit_result["x"].detectors[default_det]
            y_fit = fit_result["y"].detectors[default_det]

            rms_sizes = (x_fit.sigma, y_fit.sigma)
        else:
            rms_sizes = None
        return rms_sizes

    def _organize_data_by_profile(self, profile_indices) -> dict:
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
                positions=positions, detectors=detectors, profile_indices=profile_indices
            )

        profile_measurements = {}
        devices = self.collection_result.metadata.detectors
        devices.append(self.collection_result.metadata.wire_name)
        for profile, index in profile_indices.items():
            detectors = {}
            positions = None
            for d_n in self.collection_result.metadata.detectors:
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
